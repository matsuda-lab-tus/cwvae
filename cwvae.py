import torch
import torch.nn as nn
import torch.distributions as dist
from cnns import Encoder, Decoder
from cells import RSSMCell

class CWVAE(nn.Module):
    def __init__(
        self,
        levels,
        tmp_abs_factor,
        state_sizes,
        embed_size,
        obs_embed_size,
        enc_dense_layers,
        enc_dense_embed_size,
        channels_mult,
        device,
        cell_type,
        min_stddev,
        mean_only_cell=False,
        reset_states=False,
    ):
        super(CWVAE, self).__init__()
        self._levels = levels
        self._tmp_abs_factor = tmp_abs_factor
        self._state_sizes = state_sizes
        self._embed_size = embed_size
        self._obs_embed_size = obs_embed_size
        self._cell_type = cell_type
        self._min_stddev = min_stddev
        self._mean_only_cell = mean_only_cell
        self._reset_states = reset_states
        self.device = device

        # エンコーダーとデコーダーの設定
        self.encoder = Encoder(
            levels,
            tmp_abs_factor,
            dense_layers=enc_dense_layers,
            embed_size=enc_dense_embed_size,
            channels_mult=channels_mult,
        ).to(device)

        self.decoder = Decoder(
            output_channels=3,
            embed_size=self._state_sizes["deter"],
            channels_mult=channels_mult,
            final_activation=nn.Tanh(),
        ).to(device)

        # RSSMセルをレベルごとに作成
        self.cells = nn.ModuleList()
        for level in range(self._levels):
            if self._cell_type == 'RSSMCell':
                cell = RSSMCell(
                    stoch_size=self._state_sizes["stoch"],
                    deter_size=self._state_sizes["deter"],
                    embed_size=self._embed_size,
                    obs_embed_size=self._obs_embed_size,
                    reset_states=self._reset_states,
                    min_stddev=self._min_stddev,
                    mean_only=self._mean_only_cell,
                ).to(device)
            else:
                raise NotImplementedError(f"Unknown cell type {self._cell_type}")
            self.cells.append(cell)

        # 潜在変数を埋め込み空間に変換する線形層
        self.stoch_to_embed = nn.Linear(self._state_sizes["stoch"], self._embed_size).to(device)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.GRUCell):
            for name, param in m.named_parameters():
                if 'weight' in name:
                    nn.init.kaiming_uniform_(param, nonlinearity='relu')
                elif 'bias' in name:
                    nn.init.zeros_(param)

    def decode_prior_multistep(self, prior_multistep):
        """
        prior_multistep: Tensor of shape [batch, steps, stoch_size]
        Returns: Tensor of shape [batch, steps, channels, height, width]
        """
        embed = self.stoch_to_embed(prior_multistep)  # [B, T, embed_size]
        decoded = self.decoder(embed)  # [B, T, C, H, W]
        return decoded

    def hierarchical_unroll(self, inputs, actions=None, use_observations=None, initial_state=None):
        level_top = self._levels - 1

        if initial_state is None:
            initial_state = [None] * self._levels

        if not isinstance(use_observations, list):  # use_observationsがリストでない場合、すべてのレベルで同じ値を使用
            use_observations = [use_observations] * self._levels

        context = torch.zeros(
            inputs[level_top].size(0),
            inputs[level_top].size(1),
            self.cells[-1]._detstate_size,
            device=inputs[level_top].device
        )

        prior_list = []
        posterior_list = []
        last_state_all_levels = []

        for level in range(level_top, -1, -1):
            obs_inputs = inputs[level]

            # バッチ次元の確認
            if obs_inputs.size(0) != context.size(0):
                obs_inputs = obs_inputs.transpose(0, 1)

            # obs_inputs と context のバッチサイズ確認
            if obs_inputs.size(0) != context.size(0):
                raise ValueError(f"Batch size mismatch at level {level}: obs_inputs.size(0) = {obs_inputs.size(0)}, context.size(0) = {context.size(0)}")

            if level == level_top:
                reset_state = torch.ones(
                    obs_inputs.size(0), obs_inputs.size(1), 1, device=obs_inputs.device)
            else:
                reset_state = reset_state.unsqueeze(2).repeat(1, 1, self._tmp_abs_factor, 1)
                reset_state = reset_state.view(
                    reset_state.size(0),
                    reset_state.size(1) * self._tmp_abs_factor,
                    reset_state.size(3)
                )

            expected_seq_len = obs_inputs.size(1)

            # contextのシーケンス長を合わせる
            if level != level_top:
                context = context.unsqueeze(2).repeat(1, 1, self._tmp_abs_factor, 1)
                context = context.view(
                    context.size(0),
                    context.size(1) * self._tmp_abs_factor,
                    context.size(3)
                )

            # context の長さを調整
            context = context[:, :expected_seq_len, :]

            if level == 0 and actions is not None:
                context = torch.cat([context, actions], dim=-1)

            initial = self.cells[level].zero_state(obs_inputs.size(0), obs_inputs.device)
            if initial_state[level] is not None:
                initial = initial_state[level]

            prior, posterior, posterior_last_step = manual_scan(
                self.cells[level],
                obs_inputs,
                context,
                reset_state,
                use_observations[level],  # 各レベルの use_observations を使用
                initial,
            )

            last_state_all_levels.insert(0, posterior_last_step)
            context = posterior["det_out"]

            prior_list.insert(0, prior)
            posterior_list.insert(0, posterior)

        output_bot_level = context
        return output_bot_level, last_state_all_levels, prior_list, posterior_list

    def open_loop_unroll(self, inputs, ctx_len, actions=None, use_observations=None, initial_state=None):
        if use_observations is None:
            use_observations = [True] * self._levels  # use_observationsがNoneの場合、すべてのレベルでTrueを使用

        ctx_len_backup = ctx_len
        pre_inputs = []
        post_inputs = []
        for lvl in range(self._levels):
            pre_inputs.append(inputs[lvl][:, :ctx_len, :])
            post_inputs.append(torch.zeros_like(inputs[lvl][:, ctx_len:, :]))
            ctx_len = ctx_len // self._tmp_abs_factor
        ctx_len = ctx_len_backup

        actions_pre = actions_post = None
        if actions is not None:
            actions_pre = actions[:, :ctx_len, :]
            actions_post = actions[:, ctx_len:, :]

        _, pre_last_state_all_levels, pre_priors, pre_posteriors = self.hierarchical_unroll(
            pre_inputs, actions=actions_pre, use_observations=use_observations, initial_state=initial_state
        )
        outputs_bot_level, _, post_priors, _ = self.hierarchical_unroll(
            post_inputs, actions=actions_post, use_observations=[False] * self._levels, initial_state=pre_last_state_all_levels
        )
        return pre_posteriors, pre_priors, post_priors, outputs_bot_level

    def _log_prob_obs(self, samples, mean, stddev):
        """
        Returns the log probability of the observed samples under a normal distribution
        defined by the mean and stddev.
        """
        mvn = dist.Normal(mean, stddev)
        log_prob = mvn.log_prob(samples)  # ログ確率を計算
        return log_prob.sum(dim=[-3, -2, -1])  # 各ピクセルのログ確率を合計して返す

    def _gaussian_KLD(self, dist1, dist2):
        mvn1 = dist.Normal(dist1["mean"], dist1["stddev"])
        mvn2 = dist.Normal(dist2["mean"], dist2["stddev"])
        return dist.kl_divergence(mvn1, mvn2).sum(dim=-1)

    def compute_losses(self, obs, obs_decoded, priors, posteriors, dec_stddev=0.1, kl_grad_post_perc=None, free_nats=None, beta=None):
        # dec_stddev をテンソルに変換
        if isinstance(dec_stddev, (int, float)):
            dec_stddev = torch.full_like(obs_decoded, dec_stddev)

        # 観測されたデータとデコードされたデータのネガティブ対数尤度の計算
        nll_term = -self._log_prob_obs(obs, obs_decoded, dec_stddev).mean()

        # KLダイバージェンスの計算
        kl_term = torch.tensor(0.0).to(obs.device)
        kld_all_levels = []

        for i in range(self._levels):
            kld_level = self._gaussian_KLD(posteriors[i], priors[i])

            if free_nats is not None:
                kld_level = torch.clamp(kld_level - free_nats, min=0.0)
            if beta is not None:
                if isinstance(beta, list):
                    kld_level = beta[i] * kld_level
                else:
                    kld_level = beta * kld_level

            kl_term += kld_level.mean()
            kld_all_levels.append(kld_level)

        # ELBOの計算（負の対数尤度 + KLダイバージェンス）
        neg_elbo = nll_term + kl_term
        loss = neg_elbo / obs.size(1)

        return {
            "loss": loss,
            "nll_term": nll_term,
            "kl_term": kl_term,
            "kld_all_levels": kld_all_levels
        }

def manual_scan(cell, obs_inputs, context, reset_state, use_observation, initial):
    priors = []
    posteriors = []
    prev_out = {"state": initial}
    seq_len = obs_inputs.size(1)

    for t in range(seq_len):
        inputs = (
            obs_inputs[:, t],
            context[:, t],
            reset_state[:, t],
        )
        outputs = cell(prev_out, inputs, use_observation)
        priors.append(outputs["out"][0])
        posteriors.append(outputs["out"][1])
        prev_out = outputs

    prior = {k: torch.stack([p[k] for p in priors], dim=1) for k in priors[0]}
    posterior = {k: torch.stack([p[k] for p in posteriors], dim=1) for k in posteriors[0]}
    posterior_last_step = prev_out["state"]
    return prior, posterior, posterior_last_step


def build_model(cfg, open_loop=True):
    device = cfg['device']

    # モデルのインスタンス作成
    model = CWVAE(
        levels=cfg['levels'],
        tmp_abs_factor=cfg['tmp_abs_factor'],
        state_sizes={"stoch": cfg['cell_stoch_size'], "deter": cfg['cell_deter_size']},
        embed_size=cfg['cell_embed_size'],
        obs_embed_size=cfg['enc_dense_embed_size'],
        enc_dense_layers=cfg['enc_dense_layers'],
        enc_dense_embed_size=cfg['enc_dense_embed_size'],
        channels_mult=cfg['channels_mult'],
        device=device,
        cell_type=cfg['cell_type'],
        min_stddev=cfg['cell_min_stddev'],
        mean_only_cell=cfg['cell_mean_only'],
        reset_states=cfg['cell_reset_state'],
    ).to(device)

    model.apply(model.init_weights)

    obs = torch.zeros([cfg['batch_size'], cfg['seq_len'], cfg['channels'], 64, 64]).to(device)
    obs_encoded = model.encoder(obs)

    if len(obs_encoded) != cfg['levels']:
        raise ValueError(f"Encoder output does not match expected levels. Expected {cfg['levels']}, but got {len(obs_encoded)}.")

    outputs_bot, last_state_all_levels, priors, posteriors = model.hierarchical_unroll(obs_encoded)
    obs_decoded = model.decoder(outputs_bot)

    losses = model.compute_losses(
        obs,
        obs_decoded,
        priors,
        posteriors,
        dec_stddev=cfg['dec_stddev'],
        free_nats=cfg['free_nats'],
        beta=cfg['beta'],
    )

    if open_loop:
        ctx_len = cfg['open_loop_ctx']
        pre_posteriors, pre_priors, post_priors, outputs_bot_level = model.open_loop_unroll(
            obs_encoded, ctx_len=ctx_len, use_observations=cfg.get('use_obs', True)
        )
        prior_multistep_decoded = model.decode_prior_multistep(post_priors[0]["mean"])
        open_loop_obs_decoded = {
            "posterior_recon": model.decoder(pre_posteriors[0]["det_out"]),
            "prior_multistep": prior_multistep_decoded,
            "gt_multistep": obs[:, ctx_len:, ...],
        }
    else:
        open_loop_obs_decoded = None

    return {
        "training": {
            "obs": obs,
            "encoder": model.encoder,
            "decoder": model.decoder,
            "obs_encoded": obs_encoded,
            "obs_decoded": obs_decoded,
            "priors": priors,
            "posteriors": posteriors,
            "loss": losses["loss"],
            "nll_term": losses["nll_term"],
            "kl_term": losses["kl_term"],
            "kld_all_levels": losses["kld_all_levels"],
        },
        "meta": {"model": model},
        "open_loop_obs_decoded": open_loop_obs_decoded,
    }

