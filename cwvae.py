# 必要なライブラリをインポートします
import torch  # PyTorchライブラリを使います
import torch.nn as nn  # ニューラルネットワークの基本機能を使います
import torch.distributions as dist  # 確率分布を扱う機能を使います
import cnns
from cells import RSSMCell  # 状態を表すセルをインポート
from tools import scan  # ツール関数を使うためにインポート

class CWVAE(nn.Module):
    def __init__(
        self,
        levels,
        tmp_abs_factor,
        state_sizes,
        embed_size,
        cell_type,
        lr,
        min_stddev,
        mean_only_cell=False,
        reset_states=False,
        var_scope="CWVAE",
    ):
        super(CWVAE, self).__init__()

        # パラメータを保存
        self.cell_type = cell_type
        self._levels = levels
        self._state_size = state_sizes["stoch"]
        self._detstate_size = state_sizes["deter"]
        self._embed_size = embed_size
        self._var_scope = var_scope
        self.lr = lr
        self._min_stddev = min_stddev
        self._tmp_abs_factor = tmp_abs_factor
        self._reset_states = reset_states

        # RSSMセルのリストを初期化
        self.cells = nn.ModuleList()
        for i_lvl in range(self._levels):
            if self.cell_type == "RSSMCell":
                # RSSMCellを作成
                cell = RSSMCell(
                    state_size=self._state_size,
                    detstate_size=self._detstate_size,
                    embed_size=self._embed_size,
                    reset_states=self._reset_states,
                    min_stddev=self._min_stddev,
                    mean_only=mean_only_cell,
                    var_scope="cell_" + str(i_lvl),
                )
            else:
                raise ValueError(f"Cell type {self.cell_type} not supported")
            self.cells.append(cell)

        # 確率的状態を埋め込むための線形層
        self.stoch_to_embed = nn.Linear(self._state_size, self._embed_size)

    # モデルの重みを初期化する関数です
    def init_weights(self, m):
        # 線形層の場合、重みをxavierの方法で初期化します
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)  # 重みをXavierの方法で初期化
            if m.bias is not None:
                nn.init.zeros_(m.bias)  # バイアスをゼロで初期化
        # 畳み込み層の場合、kaimingの方法で初期化します
        elif isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.kaiming_normal_(
                m.weight, mode="fan_out", nonlinearity="relu"
            )  # Kaimingの方法で初期化
            if m.bias is not None:
                nn.init.zeros_(m.bias)  # バイアスをゼロで初期化

    def hierarchical_unroll(self, inputs, actions=None, use_observations=None, initial_state=None):
        """
        各階層を通じて情報を伝えながら予測を行います。
        """

        # 観察を使用するかどうかを決定
        if use_observations is None:
            use_observations = [True] * self._levels
        elif isinstance(use_observations, bool):
            use_observations = [use_observations] * self._levels

        if initial_state is None:
            initial_state = [self.cells[level].zero_state(inputs[level].size(0), inputs[level].device) for level in range(self._levels)]

        level_top = self._levels - 1
        inputs_top = inputs[level_top]

        # 最上階層にゼロで初期化されたコンテキストを設定
        context = torch.zeros(
            inputs_top.size(0),
            inputs_top.size(1),
            self.cells[-1]._detstate_size + self.cells[-1]._state_size,
            device=inputs_top.device,
        )

        # リセット状態の初期化
        if level_top >= 1:
            inputs_top_ = inputs[level_top - 1]
            temp_zeros = torch.zeros(inputs_top_.size(0), inputs_top_.size(1), 1, device=inputs_top_.device)
            temp_ones = torch.ones_like(temp_zeros)
            _reset_state = torch.cat([temp_zeros, temp_ones], dim=-1)
            _reset_state = _reset_state.view(inputs_top_.size(0), -1, 1)
            _reset_state = _reset_state[:, :inputs_top_.size(1), :]
        else:
            _reset_state = None

        # 予測と後方推定を保存するリストを初期化
        prior_list = []
        posterior_list = []
        last_state_all_levels = []

        for level in range(level_top, -1, -1):
            obs_inputs = inputs[level]

            # 最上階層の場合、リセット状態を設定
            if level == level_top:
                reset_state, reset_state_next = (
                    torch.ones(obs_inputs.size(0), obs_inputs.size(1), 1, device=obs_inputs.device),
                    _reset_state,
                )
            else:
                reset_state, reset_state_next = (
                    reset_state,
                    reset_state.unsqueeze(2).repeat(1, 1, self._tmp_abs_factor, 1).view(
                        reset_state.size(0), -1, reset_state.size(-1)
                    ),
                )

                context = context.unsqueeze(2).repeat(1, 1, self._tmp_abs_factor, 1)
                context = context.view(context.size(0), -1, context.size(-1))

            # 観測数に合わせてリセット状態とコンテキストを調整
            reset_state = reset_state[:, :obs_inputs.size(1), :]
            context = context[:, :obs_inputs.size(1), :]

            # アクションをボトムレベルのコンテキストに追加
            if level == 0 and actions is not None:
                context = torch.cat([context, actions], dim=-1)

            # 初期状態を設定
            initial = initial_state[level]

            # `scan` 関数を使用して各ステップの出力と状態を取得
            (prior, posterior), posterior_last_step = scan(
                self.cells[level],
                (obs_inputs, context, reset_state),
                use_observations[level],
                initial,
            )

            # 最後の状態と予測をリストに保存
            last_state_all_levels.insert(0, posterior_last_step)
            context = torch.cat([posterior["sample"], posterior["det_state"]], dim=-1)
            prior_list.insert(0, prior)
            posterior_list.insert(0, posterior)

        return context, last_state_all_levels, prior_list, posterior_list


    # 予測された状態からデコードする関数
    def decode_prior_multistep(self, prior_multistep):
        embed = self.stoch_to_embed(prior_multistep)  # 隠れ状態を埋め込みに変換
        decoded = self.decoder(embed)  # デコーダーでデコード
        return decoded  # デコードされた結果を返す

    # 予測と観察の違いから損失を計算します。
    def compute_losses(
        self,
        obs,
        obs_decoded,
        priors,
        posteriors,
        dec_stddev=0.1,
        kl_grad_post_perc=None,
        free_nats=None,
        beta=None,
    ):
        """
        予測と観察の違いから損失を計算します。
        """
        dec_stddev = torch.full_like(
            obs_decoded, dec_stddev
        )  # デコードの標準偏差を設定
        nll_term = -self._log_prob_obs(
            obs, obs_decoded, dec_stddev
        ).mean()  # 負の対数尤度を計算

        kl_term = torch.tensor(0.0).to(obs.device)  # KLダイバージェンスの初期化
        kld_all_levels = []  # 各階層のKLDを保存するリスト

        for i in range(self._levels):  # 各階層ごとに
            kld_level = self._gaussian_KLD(posteriors[i], priors[i])  # KLDを計算
            if free_nats is not None:
                kld_level = torch.clamp(
                    kld_level - free_nats, min=0.0
                )  # フリーナッツの適用
            if beta is not None:
                kld_level *= beta[i] if isinstance(beta, list) else beta  # ベータの適用
            kl_term += kld_level.mean()  # KLDを合計
            kld_all_levels.append(kld_level)  # 各階層のKLDを追加

        neg_elbo = nll_term + kl_term  # ネガティブELBOの計算
        loss = neg_elbo / obs.size(1)  # 損失を計算

        return {
            "loss": loss,
            "nll_term": nll_term,
            "kl_term": kl_term,
            "kld_all_levels": kld_all_levels,
        }

    # 観察されたフレームを使って、未来のフレームを予測する関数です
    def open_loop_unroll(self, inputs, ctx_len, actions=None, use_observations=None):
        # 観察を使用するかどうかを設定します。指定がなければすべてのレベルで使用するようにします。
        if use_observations is None:
            use_observations = [True] * self._levels

        # コンテキスト長のバックアップを取ります
        ctx_len_backup = ctx_len
        pre_inputs = []
        post_inputs = []

        # 各レベルで、観察部分と予測部分にデータを分けます
        for lvl in range(self._levels):
            pre_inputs.append(inputs[lvl][:, :ctx_len, ...])  # 観察部分
            post_inputs.append(torch.zeros_like(inputs[lvl][:, ctx_len:, ...]))  # 予測部分をゼロで初期化
            ctx_len = ctx_len // self._tmp_abs_factor  # 次のレベルの時間間隔を調整
        ctx_len = ctx_len_backup  # 元のコンテキスト長に戻します

        # アクションがあれば、観察部分と予測部分に分けます
        actions_pre = actions_post = None
        if actions is not None:
            actions_pre = actions[:, :ctx_len, :]
            actions_post = actions[:, ctx_len:, :]

        # 観察部分での予測（pre_unroll）を実行
        _, pre_last_state_all_levels, pre_priors, pre_posteriors = self.hierarchical_unroll(
            pre_inputs, actions=actions_pre, use_observations=use_observations
        )

        # 観察状態を使い、予測部分（post_unroll）を観察なしで実行
        _, _, post_priors, _ = self.hierarchical_unroll(
            post_inputs, actions=actions_post, use_observations=[False] * self._levels, initial_state=pre_last_state_all_levels
        )

        # 観察結果（pre）と予測結果（post）を返します
        return pre_posteriors, pre_priors, post_priors


    # 観察されたデータ（サンプル）と予測されたデータの違いを計算するためのもの

    def _log_prob_obs(self, samples, mean, stddev):
        """
        Returns the log probability density of samples in the given distribution.
        The last dim of the samples is taken as the one to sum over.
        """
        # サンプルの最後の次元を平坦化
        if samples.dim() > 3:  # チャネル次元が存在する場合
            batch_size, seq_len = samples.shape[:2]
            new_shape = (batch_size, seq_len, -1)  # 最後の次元を1次元に平坦化
            samples = samples.view(new_shape)
            mean = mean.view(new_shape)
            if isinstance(stddev, torch.Tensor):
                stddev = stddev.view(new_shape)
        
        # `Independent`と`Normal`を用いた分布の定義
        dist_normal = dist.Independent(dist.Normal(mean, stddev), reinterpreted_batch_ndims=1)
        log_prob = dist_normal.log_prob(samples)  # 各サンプルの対数尤度を計算
        return log_prob


    # 2つの「ガウス分布（正規分布）」の間の違いを計算して、それを数値で表している
    def _gaussian_KLD(self, dist1, dist2):
        # 対角共分散行列を持つ分布を作成
        scale_tril1 = torch.diag_embed(dist1["stddev"])
        scale_tril2 = torch.diag_embed(dist2["stddev"])
        mvn1 = dist.MultivariateNormal(dist1["mean"], scale_tril=scale_tril1)
        mvn2 = dist.MultivariateNormal(dist2["mean"], scale_tril=scale_tril2)
        # 計算したKLダイバージェンスを全部足し合わせる
        return dist.kl_divergence(mvn1, mvn2).sum(
            dim=-1
        )  # KLダイバージェンスを計算して返す


# 動画の中で、時間が進むごとにどんな変化が起きていくか」を計算するための仕組みを作っている
# 時間ごとに少しずつ変わるものを計算して、それを集める役割
# def manual_scan(cell, obs_inputs, context, reset_state, use_observation, initial):
#     priors = []  # 予測を保存するリスト
#     posteriors = []  # 後方推定を保存するリスト
#     prev_out = initial  # 前回の出力を初期状態で設定
#     seq_len = obs_inputs.size(1)  # 入力の長さを取得

#     for t in range(seq_len):  # 各タイムステップに対して
#         inputs = (
#             obs_inputs[:, t],  # 現在の観察入力
#             context[:, t, :cell._state_size + cell._detstate_size],  # context のサイズを統一
#             reset_state[:, t],  # 現在のリセット状態
#         )
#         # print(f"context at step {t}: {context[:, t, :cell._state_size + cell._detstate_size].size()}")
#         outputs = cell(prev_out, inputs, use_observation)  # セルに入力を渡す
#         priors.append(outputs["out"][0])  # 予測を追加
#         posteriors.append(outputs["out"][1])  # 後方推定を追加
#         prev_out = outputs  # 前回の出力を更新

#     # すべての予測と後方推定をスタックして返す
#     prior = {k: torch.stack([p[k] for p in priors], dim=1) for k in priors[0]}
#     posterior = {
#         k: torch.stack([p[k] for p in posteriors], dim=1) for k in posteriors[0]
#     }
#     posterior_last_step = prev_out["state"]  # 最後の状態を取得
#     return prior, posterior, posterior_last_step  # 予測と後方推定、最後の状態を返す


# モデルを構築する関数
import torch
import torch.nn as nn

def build_model(cfg, open_loop=True):
    # 観測入力の初期化
    obs = torch.zeros((cfg.batch_size, cfg.seq_len, cfg.channels, 64, 64), dtype=torch.float32)  # 仮の形状

    # EncoderとDecoderの初期化
    encoder = cnns.Encoder(
        cfg.levels,
        cfg.tmp_abs_factor,
        dense_layers=cfg.enc_dense_layers,
        embed_size=cfg.enc_dense_embed_size,
        channels_mult=cfg.channels_mult,
    )
    decoder = cnns.Decoder(cfg.channels, channels_mult=cfg.channels_mult)

    # エンコードされた観測値を取得
    obs_encoded = encoder(obs)

    # CWVAEモデルのインスタンスを作成
    model = CWVAE(
        levels=cfg.levels,
        tmp_abs_factor=cfg.tmp_abs_factor,
        state_sizes=dict(stoch=cfg.cell_stoch_size, deter=cfg.cell_deter_size),
        embed_size=cfg.cell_embed_size,
        cell_type=cfg.cell_type,
        lr=cfg.lr,
        min_stddev=cfg.cell_min_stddev,
        mean_only_cell=cfg.cell_mean_only,
        reset_states=cfg.cell_reset_state,
    )

    # 階層的なアンロールとデコード
    initial_state = [model.cells[level].zero_state(cfg.batch_size, obs.device) for level in range(cfg.levels)]
    outputs_bot, _, priors, posteriors = model.hierarchical_unroll(obs_encoded, initial_state=initial_state)
    obs_decoded = decoder(outputs_bot)

    # 損失の計算
    loss = model.compute_losses(
        obs,
        obs_decoded,
        priors,
        posteriors,
        dec_stddev=cfg.dec_stddev,
        kl_grad_post_perc=cfg.kl_grad_post_perc,
        free_nats=cfg.free_nats,
        beta=cfg.beta,
    )

    # 結果を辞書として格納
    out = {
        "training": {
            "obs": obs,
            "encoder": encoder,
            "decoder": decoder,
            "obs_encoded": obs_encoded,
            "obs_decoded": obs_decoded,
            "priors": priors,
            "posteriors": posteriors,
            "loss": loss,
        },
        "meta": {"model": model},
    }

    # Open loopの実行
    if open_loop:
        posteriors_recon, priors_onestep, priors_multistep = model.open_loop_unroll(
            obs_encoded, cfg.open_loop_ctx, use_observations=cfg.use_obs
        )
        obs_decoded_posterior_recon = decoder(posteriors_recon[0]["output"])
        obs_decoded_prior_onestep = decoder(priors_onestep[0]["output"])
        obs_decoded_prior_multistep = decoder(priors_multistep[0]["output"])
        gt_multistep = obs[:, cfg.open_loop_ctx:, ...]

        out.update(
            {
                "open_loop_obs_decoded": {
                    "posterior_recon": obs_decoded_posterior_recon,
                    "prior_onestep": obs_decoded_prior_onestep,
                    "prior_multistep": obs_decoded_prior_multistep,
                    "gt_multistep": gt_multistep,
                }
            }
        )
    return out



# def forward(self, obs):
#     """
#     観察データを受け取り、エンコード、階層的なアンロール、デコードを行う。
#     :param obs: 入力観察データ (batch_size, seq_len, channels, height, width)
#     :return: 再構成された観察データ、損失情報
#     """
#     # 観察データをエンコーダーでエンコード
#     obs_encoded = self.encoder(obs)

#     # 階層的にアンロールして予測を行う
#     outputs_bot, last_state_all_levels, priors, posteriors = (
#         self.hierarchical_unroll(obs_encoded)
#     )

#     # デコーダーで再構成された観察データを取得
#     obs_decoded = self.decoder(outputs_bot)[0]

#     # 損失を計算
#     losses = self.compute_losses(
#         obs=obs, obs_decoded=obs_decoded, priors=priors, posteriors=posteriors
#     )

#     # 再構成された観察データと損失を返す
#     return obs_decoded, losses