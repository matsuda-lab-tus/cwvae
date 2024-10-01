import torch
import torch.nn as nn
import torch.distributions as dist
from cnns import Encoder, Decoder
from cells import RSSMCell
import tools

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
            embed_size=enc_dense_embed_size,
            channels_mult=channels_mult,
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
                )
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
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def decode_prior_multistep(self, prior_multistep):
        """
        デコードを行う関数
        prior_multistep: Tensor of shape [batch, steps, stoch_size]
        Returns: Tensor of shape [batch, steps, channels, height, width]
        """
        print(f"[DEBUG] Input to decode_prior_multistep shape: {prior_multistep.shape}")

        batch, steps, stoch_size = prior_multistep.shape
        embed = self.stoch_to_embed(prior_multistep)

        # 形状確認
        print(f"[DEBUG] Embed shape before reshaping: {embed.shape}")

        # リシェイプ
        embed = embed.view(batch, steps, self._embed_size)

        # 形状確認
        print(f"[DEBUG] Embed shape after reshaping: {embed.shape}")

        # デコード
        decoded = self.decoder(embed)

        # デコード結果の形状確認
        print(f"[DEBUG] Decoded output shape: {decoded.shape}")

        return decoded

    def hierarchical_unroll(self, inputs, actions=None, use_observations=None, initial_state=None):
        level_top = self._levels - 1

        # 初期状態の設定
        if initial_state is None:
            initial_state = [None] * self._levels

        # デバッグ用ログ
        print(f"[DEBUG] Level top: {level_top}")
        print(f"[DEBUG] Inputs length: {len(inputs)}")

        for i, inp in enumerate(inputs):
            print(f"[DEBUG] Input shape at level {i}: {inp.shape}")

        # レベル数が不足していないかチェック
        if len(inputs) <= level_top:
            raise IndexError(f"inputs does not have enough levels. Expected at least {level_top + 1}, but got {len(inputs)}")

        # contextの初期化
        context = torch.zeros(
            inputs[level_top].size(0),
            inputs[level_top].size(1),
            self.cells[-1]._detstate_size
        ).to(inputs[level_top].device)

        prior_list = []
        posterior_list = []
        last_state_all_levels = []

        for level in range(level_top, -1, -1):
            obs_inputs = inputs[level]
            print(f"[DEBUG] Input shape in CWVAE level {level}: {obs_inputs.shape}")
            if level == level_top:
                reset_state = torch.ones(obs_inputs.size(0), obs_inputs.size(1), 1).to(obs_inputs.device)
            else:
                reset_state = reset_state.repeat(1, self._tmp_abs_factor, 1)

            context = context[:, :obs_inputs.size(1), :]
            if level == 0 and actions is not None:
                context = torch.cat([context, actions], dim=-1)

            # 初期状態の設定
            initial = self.cells[level].zero_state(obs_inputs.size(0), obs_inputs.device)
            if initial_state[level] is not None:
                initial["sample"] = initial_state[level]["sample"]
                initial["det_state"] = initial_state[level]["det_state"]

            # manual_scanを呼び出してpriorとposteriorを取得
            prior, posterior, posterior_last_step = manual_scan(
                self.cells[level],
                obs_inputs,
                context,
                reset_state,
                use_observations[level] if use_observations is not None else True,
                initial,
            )

            last_state_all_levels.insert(0, posterior_last_step)
            context = posterior["det_out"]

            prior_list.insert(0, prior)
            posterior_list.insert(0, posterior)

            # デバッグ用のcontextの形状確認
            print(f"[DEBUG] Context shape at level {level}: {context.shape}")

            if level != 0:
                context = context.unsqueeze(2).expand(-1, -1, self._tmp_abs_factor, -1)
                context = context.reshape(context.size(0), -1, context.size(3))

        output_bot_level = context
        return output_bot_level, last_state_all_levels, prior_list, posterior_list

    def _gaussian_KLD(self, dist1, dist2):
        mvn1 = dist.Normal(dist1["mean"], dist1["stddev"])
        mvn2 = dist.Normal(dist2["mean"], dist2["stddev"])
        return dist.kl_divergence(mvn1, mvn2).sum(dim=-1)

    def _log_prob_obs(self, samples, mean, stddev):
        mvn = dist.Normal(mean, stddev)
        return mvn.log_prob(samples).sum(dim=-1)

    def compute_losses(self, obs, obs_decoded, priors, posteriors, dec_stddev=0.1, free_nats=None, beta=None):
        nll_term = -self._log_prob_obs(obs, obs_decoded, dec_stddev).mean()
        kl_term = torch.tensor(0.0).to(obs.device)
        kld_all_levels = []

        for i in range(self._levels):
            kld_level = self._gaussian_KLD(posteriors[i], priors[i])
            kl_term += kld_level.mean()
            kld_all_levels.append(kld_level)

        neg_elbo = nll_term + kl_term
        loss = neg_elbo / obs.size(1)

        print(f"[DEBUG] Loss calculated: {loss.item()}")

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
    posterior_last_step = posterior

    return prior, posterior, posterior_last_step

def build_model(cfg, open_loop=True):
    obs = torch.zeros([cfg.batch_size, cfg.seq_len, cfg.channels, 64, 64]).to(cfg.device)

    encoder = Encoder(
        cfg.levels,
        cfg.tmp_abs_factor,
        dense_layers=cfg.enc_dense_layers,
        embed_size=cfg.enc_dense_embed_size,
        channels_mult=cfg.channels_mult,
    ).to(cfg.device)

    decoder = Decoder(
        output_channels=3,
        embed_size=cfg.enc_dense_embed_size,
        channels_mult=cfg.channels_mult,
    ).to(cfg.device)

    obs_encoded_mu, obs_encoded_logvar = encoder(obs)
    print(f"[DEBUG] Encoded output length: {len(obs_encoded_mu)}")

    for i, enc_mu in enumerate(obs_encoded_mu):
        print(f"[DEBUG] Encoded output at level {i}: {enc_mu.shape}")
    
    if len(obs_encoded_mu) != cfg.levels:
        raise ValueError(f"Encoder output does not match expected levels. Expected {cfg.levels}, but got {len(obs_encoded_mu)}.")

    model = CWVAE(
        levels=cfg.levels,
        tmp_abs_factor=cfg.tmp_abs_factor,
        state_sizes={"stoch": cfg.cell_stoch_size, "deter": cfg.cell_deter_size},
        embed_size=cfg.cell_embed_size,
        obs_embed_size=cfg.enc_dense_embed_size,
        enc_dense_layers=cfg.enc_dense_layers,
        enc_dense_embed_size=cfg.enc_dense_embed_size,
        channels_mult=cfg.channels_mult,
        device=cfg.device,
        cell_type=cfg.cell_type,
        min_stddev=cfg.cell_min_stddev,
        mean_only_cell=cfg.cell_mean_only,
        reset_states=cfg.cell_reset_state,
    ).to(cfg.device)

    outputs_bot, _, priors, posteriors = model.hierarchical_unroll(obs_encoded_mu)
    obs_decoded = decoder(outputs_bot)

    obs_decoded = obs_decoded.view(cfg.batch_size, cfg.seq_len, cfg.channels, 64, 64)

    print(f"[DEBUG] obs_decoded shape: {obs_decoded.shape}")

    losses = model.compute_losses(
        obs,
        obs_decoded,
        priors,
        posteriors,
        dec_stddev=cfg.dec_stddev,
        free_nats=cfg.free_nats,
        beta=cfg.beta,
    )

    if open_loop:
        prior_multistep_decoded = model.decode_prior_multistep(priors[0]["mean"])
        open_loop_obs_decoded = {
            "prior_multistep": prior_multistep_decoded,
            "gt_multistep": obs_decoded,
        }
    else:
        open_loop_obs_decoded = None

    return {
        "training": {
            "obs": obs,
            "encoder": encoder,
            "decoder": decoder,
            "obs_encoded": obs_encoded_mu,
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
