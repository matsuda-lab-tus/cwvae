import torch
import torch.nn as nn
import torch.distributions as dist
from cnns import Encoder, Decoder
from cells import RSSMCell
import math

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
        self.device = device  # デバイスを保存

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

        # RSSMセルをレベルごとに作成し、デバイスに移動
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
                ).to(device)  # デバイスに移動
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
        デコードを行う関数
        prior_multistep: Tensor of shape [batch, steps, stoch_size]
        Returns: Tensor of shape [batch, steps, channels, height, width]
        """
        print(f"[DEBUG] Input to decode_prior_multistep shape: {prior_multistep.shape}")

        batch, steps, stoch_size = prior_multistep.shape
        embed = self.stoch_to_embed(prior_multistep)  # [B, T, embed_size]

        # 形状確認
        print(f"[DEBUG] Embed shape before reshaping: {embed.shape}")

        # リシェイプ
        embed = embed.view(batch, steps, self._embed_size)  # [B, T, embed_size]

        # 形状確認
        print(f"[DEBUG] Embed shape after reshaping: {embed.shape}")

        # デコード
        decoded = self.decoder(embed)  # [B, T, 3, 32, 32]

        # デコード結果の形状確認
        print(f"[DEBUG] Decoded output shape: {decoded.shape}")

        return decoded

    def hierarchical_unroll(self, inputs, actions=None, use_observations=None, initial_state=None):
        level_top = self._levels - 1

        # 初期状態の設定
        if initial_state is None:
            initial_state = [None] * self._levels

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

            expected_seq_len = obs_inputs.size(1)
            actual_context_len = context.size(1)

            # コンテキストを繰り返して期待されるシーケンス長を満たす
            if actual_context_len < expected_seq_len:
                repeats = math.ceil(expected_seq_len / actual_context_len)
                context = context.repeat(1, repeats, 1)
                context = context[:, :expected_seq_len, :]
                actual_context_len = context.size(1)
                print(f"[DEBUG] Context repeated {repeats} times to match expected_seq_len")

            # シーケンス長のチェック
            if actual_context_len < expected_seq_len:
                raise IndexError(f"Context sequence length {actual_context_len} is less than expected {expected_seq_len} at level {level}")

            # 必要に応じてコンテキストをスライス
            context = context[:, :expected_seq_len, :]
            print(f"[DEBUG] Context shape after slicing: {context.shape}")

            if level == 0 and actions is not None:
                context = torch.cat([context, actions], dim=-1)
                print(f"[DEBUG] Actions concatenated to context. New context shape: {context.shape}")

            # 初期状態の設定
            initial = self.cells[level].zero_state(obs_inputs.size(0), obs_inputs.device)
            if initial_state[level] is not None:
                initial["sample"] = initial_state[level]["sample"]
                initial["det_state"] = initial_state[level]["det_state"]
                print(f"[DEBUG] Initial state set from initial_state for level {level}")

            # `manual_scan` を呼び出して prior と posterior を取得
            prior, posterior, posterior_last_step = manual_scan(
                self.cells[level],
                obs_inputs,
                context,
                reset_state,
                use_observations[level] if use_observations is not None else True,
                initial,
            )
            print(f"[DEBUG] manual_scan returned prior shape: {prior['mean'].shape}, posterior shape: {posterior['mean'].shape}")

            last_state_all_levels.insert(0, posterior_last_step)
            context = posterior["det_out"]

            prior_list.insert(0, prior)
            posterior_list.insert(0, posterior)

            # デバッグ用の context の形状確認
            print(f"[DEBUG] Context shape at level {level}: {context.shape}")

            if level != 0:
                # コンテキストの繰り返しを適切に行う（すでに繰り返し済み）
                pass

        output_bot_level = context
        print(f"[DEBUG] hierarchical_unroll output_bot_level shape: {output_bot_level.shape}")
        return output_bot_level, last_state_all_levels, prior_list, posterior_list

    def _gaussian_KLD(self, dist1, dist2):
        mvn1 = dist.Normal(dist1["mean"], dist1["stddev"])
        mvn2 = dist.Normal(dist2["mean"], dist2["stddev"])
        return dist.kl_divergence(mvn1, mvn2).sum(dim=-1)

    def compute_losses(self, obs, obs_decoded, priors, posteriors, dec_stddev=0.1, free_nats=None, beta=None):
        # 損失計算前にデバッグ情報を表示
        print(f"[DEBUG] obs shape: {obs.shape}")  # [50, 100, 3, 64, 64]
        print(f"[DEBUG] obs_decoded shape: {obs_decoded.shape}")  # [50, 100, 3, 64, 64]

        # dec_stddev をテンソルに変換
        if isinstance(dec_stddev, (int, float)):
            dec_stddev = torch.full_like(obs_decoded, dec_stddev)
            print(f"[DEBUG] dec_stddev is a scalar. Converted to tensor with shape: {dec_stddev.shape}")
        elif isinstance(dec_stddev, torch.Tensor):
            print(f"[DEBUG] dec_stddev is already a tensor with shape: {dec_stddev.shape}")
        else:
            raise TypeError(f"Unsupported type for dec_stddev: {type(dec_stddev)}")

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

    def _log_prob_obs(self, samples, mean, stddev):
        # mean と stddev の形状が [B, T, 3, 64, 64] であることを確認
        print(f"[DEBUG] samples shape: {samples.shape}")  # [50, 100, 3, 64, 64]
        print(f"[DEBUG] mean shape: {mean.shape}")  # [50, 100, 3, 64, 64]
        print(f"[DEBUG] stddev shape: {stddev.shape}")  # [50, 100, 3, 64, 64]
        
        mvn = dist.Normal(mean, stddev)
        return mvn.log_prob(samples).sum(dim=-1)

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

    print(f"[DEBUG] manual_scan returned 3 values: prior, posterior, posterior_last_step")
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

    # 重みの初期化
    model.apply(model.init_weights)

    # 入力データの準備
    obs = torch.zeros([cfg['batch_size'], cfg['seq_len'], cfg['channels'], 64, 64]).to(device)

    # エンコーダーの出力を取得
    obs_encoded = model.encoder(obs)
    print(f"[DEBUG] Encoded output length: {len(obs_encoded)}")

    for i, enc_mu in enumerate(obs_encoded):
        print(f"[DEBUG] Encoded output at level {i}: {enc_mu.shape}")

    if len(obs_encoded) != cfg['levels']:
        raise ValueError(f"Encoder output does not match expected levels. Expected {cfg['levels']}, but got {len(obs_encoded)}.")

    # 階層的にデータを処理
    outputs = model.hierarchical_unroll(obs_encoded)
    print(f"[DEBUG] hierarchical_unroll returned {len(outputs)} values")
    outputs_bot, last_state_all_levels, priors, posteriors = outputs
    print(f"[DEBUG] outputs_bot shape: {outputs_bot.shape}")
    print(f"[DEBUG] last_state_all_levels length: {len(last_state_all_levels)}")
    print(f"[DEBUG] priors length: {len(priors)}")
    print(f"[DEBUG] posteriors length: {len(posteriors)}")

    obs_decoded = model.decoder(outputs_bot)
    print(f"[DEBUG] obs_decoded shape: {obs_decoded.shape}")

    # 損失の計算
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
        prior_multistep_decoded = model.decode_prior_multistep(priors[0]["mean"])
        print(f"[DEBUG] prior_multistep_decoded shape: {prior_multistep_decoded.shape}")
        open_loop_obs_decoded = {
            "prior_multistep": prior_multistep_decoded,
            "gt_multistep": obs_decoded,
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
