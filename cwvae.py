# 必要なライブラリをインポートします
import torch  # PyTorchライブラリを使います
import torch.nn as nn  # ニューラルネットワークの基本機能を使います
import torch.distributions as dist  # 確率分布を扱う機能を使います
from cnns import Encoder, Decoder  # エンコーダーとデコーダーを使うためにインポート
from cells import RSSMCell  # 状態を表すセルをインポート

# CWVAEという新しいクラスを定義します。このクラスはnn.Moduleから継承します。
class CWVAE(nn.Module):
    def __init__(  # クラスの初期化関数
        self,
        levels,  # 階層の数
        tmp_abs_factor,  # 時間の絶対的な因子
        state_sizes,  # 状態のサイズ
        embed_size,  # 埋め込みサイズ
        obs_embed_size,  # 観察の埋め込みサイズ
        enc_dense_layers,  # エンコーダーの密な層の数
        enc_dense_embed_size,  # エンコーダーの埋め込みサイズ
        channels_mult,  # チャンネルの倍率
        device,  # 使用するデバイス（CPUまたはGPU）
        cell_type,  # 使用するセルのタイプ
        min_stddev,  # 最小の標準偏差
        mean_only_cell=False,  # 平均のみのセルを使うか
        reset_states=False,  # 状態をリセットするか
    ):
        super(CWVAE, self).__init__()  # 親クラスを初期化

        # 各種パラメータを保存
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

        # エンコーダーとデコーダーの定義
        self.encoder = Encoder(
            levels,
            tmp_abs_factor,
            dense_layers=enc_dense_layers,
            embed_size=enc_dense_embed_size,
            channels_mult=channels_mult,
        ).to(device)  # エンコーダーをデバイスに移動

        self.decoder = Decoder(
            output_channels=3,  # 出力チャンネル数（色の数）
            embed_size=self._state_sizes["deter"],  # 埋め込みサイズ
            channels_mult=channels_mult,
            final_activation=nn.Tanh(),  # 最後にTanh関数を使う
        ).to(device)  # デコーダーをデバイスに移動

        # 各階層のRSSMセルを作成
        self.cells = nn.ModuleList()  # モジュールリストを作成
        for level in range(self._levels):  # 各階層に対して
            if self._cell_type == 'RSSMCell':  # セルのタイプがRSSMCellの場合
                # RSSMCellを作成し、必要なパラメータを設定
                cell = RSSMCell(
                    stoch_size=self._state_sizes["stoch"],  # 隠れ状態のサイズ
                    deter_size=self._state_sizes["deter"],  # 決定的状態のサイズ
                    embed_size=self._embed_size,  # 埋め込みサイズ
                    obs_embed_size=self._obs_embed_size,  # 観察の埋め込みサイズ
                    reset_states=self._reset_states,  # 状態をリセットするか
                    min_stddev=self._min_stddev,  # 最小の標準偏差
                    mean_only=self._mean_only_cell,  # 平均のみのセルを使うか
                ).to(device)  # セルをデバイスに移動
            else:
                raise NotImplementedError(f"Unknown cell type {self._cell_type}")  # 不明なセルタイプのエラー
            self.cells.append(cell)  # セルをリストに追加

        # 確率的状態を埋め込むための線形層
        self.stoch_to_embed = nn.Linear(self._state_sizes["stoch"], self._embed_size).to(device)  # 隠れ状態から埋め込みを作成

    # モデルの重みを初期化する関数です
    def init_weights(self, m):  
        # 線形層の場合、重みをxavierの方法で初期化します
        if isinstance(m, nn.Linear):  
            nn.init.xavier_uniform_(m.weight)  # 重みをXavierの方法で初期化
            if m.bias is not None:
                nn.init.zeros_(m.bias)  # バイアスをゼロで初期化
        # 畳み込み層の場合、kaimingの方法で初期化します
        elif isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')  # Kaimingの方法で初期化
            if m.bias is not None:
                nn.init.zeros_(m.bias)  # バイアスをゼロで初期化

    # 階層的に情報を伝えながら予測を行う関数
    def hierarchical_unroll(self, inputs, actions=None, use_observations=None, initial_state=None):
        """
        各階層を通じて情報を伝えながら予測を行います。
        """
        # 観察を使用するかどうかを決定します
        if use_observations is None:
            use_observations = [True] * self._levels  # すべての階層で観察を使用する
        elif isinstance(use_observations, bool):
            use_observations = [use_observations] * self._levels  # boolの場合、すべての階層に同じ値を設定

        level_top = self._levels - 1  # 最上階層のインデックス
        # コンテキストの初期化
        context = torch.zeros(
            inputs[level_top].size(0), inputs[level_top].size(1), self.cells[-1]._detstate_size, device=inputs[level_top].device
        )

        # 予測と後方推定を保存するリストを初期化
        prior_list = []  # 予測リスト
        posterior_list = []  # 後方推定リスト
        last_state_all_levels = []  # 各階層の最後の状態

        for level in range(level_top, -1, -1):  # 最上階層から最下階層へ
            obs_inputs = inputs[level]  # 入力を取得

            # 最上階層の場合、リセット状態を設定
            if level == level_top:
                reset_state = torch.ones(obs_inputs.size(0), obs_inputs.size(1), 1, device=obs_inputs.device)
            else:  # 下の階層の場合、リセット状態を展開
                reset_state = reset_state.unsqueeze(2).repeat(1, 1, self._tmp_abs_factor, 1).view(
                    reset_state.size(0), -1, reset_state.size(-1)
                )
                context = context.unsqueeze(2).repeat(1, 1, self._tmp_abs_factor, 1).view(
                    context.size(0), -1, context.size(-1)
                )

            initial = self.cells[level].zero_state(obs_inputs.size(0), obs_inputs.device)  # 初期状態を設定
            # セルを使って予測を行う
            prior, posterior, posterior_last_step = manual_scan(
                self.cells[level],  # 現在のセル
                obs_inputs,  # 観察入力
                context,  # コンテキスト
                reset_state,  # リセット状態
                use_observations[level],  # 観察を使用するか
                initial,  # 初期状態
            )

            last_state_all_levels.insert(0, posterior_last_step)  # 最後の状態を追加
            context = posterior["det_out"]  # コンテキストを更新

            prior_list.insert(0, prior)  # 予測を追加
            posterior_list.insert(0, posterior)  # 後方推定を追加

        return context, last_state_all_levels, prior_list, posterior_list  # 結果を返す

    # 予測された状態からデコードする関数
    def decode_prior_multistep(self, prior_multistep):
        embed = self.stoch_to_embed(prior_multistep)  # 隠れ状態を埋め込みに変換
        decoded = self.decoder(embed)  # デコーダーでデコード
        return decoded  # デコードされた結果を返す

    # 予測と観察の違いから損失を計算します。
    def compute_losses(self, obs, obs_decoded, priors, posteriors, dec_stddev=0.1, kl_grad_post_perc=None, free_nats=None, beta=None):
        """
        予測と観察の違いから損失を計算します。
        """
        dec_stddev = torch.full_like(obs_decoded, dec_stddev)  # デコードの標準偏差を設定
        nll_term = -self._log_prob_obs(obs, obs_decoded, dec_stddev).mean()  # 負の対数尤度を計算

        kl_term = torch.tensor(0.0).to(obs.device)  # KLダイバージェンスの初期化
        kld_all_levels = []  # 各階層のKLDを保存するリスト

        for i in range(self._levels):  # 各階層ごとに
            kld_level = self._gaussian_KLD(posteriors[i], priors[i])  # KLDを計算
            if free_nats is not None:
                kld_level = torch.clamp(kld_level - free_nats, min=0.0)  # フリーナッツの適用
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
    def open_loop_unroll(self, inputs, ctx_len, actions=None, use_observations=None, initial_state=None):
        # 観察を使用するかどうかを決定します
        if use_observations is None:
            use_observations = [True] * self._levels  # すべての階層で観察を使用する設定

        # もともとのコンテキスト長をバックアップします
        ctx_len_backup = ctx_len
        pre_inputs = []  # 観察データ用のリスト
        post_inputs = []  # 予測データ用のリスト

        # 各階層ごとに観察部分と予測部分にデータを分けます
        for lvl in range(self._levels):
            pre_inputs.append(inputs[lvl][:, :ctx_len, :])  # 観察部分を追加
            post_inputs.append(torch.zeros_like(inputs[lvl][:, ctx_len:, :]))  # 予測部分をゼロで初期化
            ctx_len = ctx_len // self._tmp_abs_factor  # 次の階層用に時間を縮めます
        ctx_len = ctx_len_backup  # バックアップしたコンテキスト長を戻します

        # アクションを観察と予測の部分に分けます
        actions_pre = actions_post = None
        if actions is not None:
            actions_pre = actions[:, :ctx_len, :]  # アクションの観察部分
            actions_post = actions[:, ctx_len:, :]  # アクションの予測部分

        # 観察データで最初の予測を行います
        _, pre_last_state_all_levels, pre_priors, pre_posteriors = self.hierarchical_unroll(
            pre_inputs, actions=actions_pre, use_observations=use_observations, initial_state=initial_state
        )

        # 観察した状態を使って、観察なしで未来のフレームを予測します
        outputs_bot_level, _, post_priors, _ = self.hierarchical_unroll(
            post_inputs, actions=actions_post, use_observations=[False] * self._levels, initial_state=pre_last_state_all_levels
        )

        # 観察部分の結果と予測部分の結果を返します
        return pre_posteriors, pre_priors, post_priors, outputs_bot_level

    # 観察されたデータ（サンプル）と予測されたデータの違いを計算するためのもの
    def _log_prob_obs(self, samples, mean, stddev):
        """
        Returns the log probability of the observed samples under a normal distribution
        defined by the mean and stddev.
        """
        # 正規分布を作成
        mvn = dist.Normal(mean, stddev) 
        log_prob = mvn.log_prob(samples)  # 実際の画像が、この正規分布にどれくらい似ているかを計算
        return log_prob.sum(dim=[-3, -2, -1])  # 各ピクセルのログ確率を合計して返す

    # 2つの「ガウス分布（正規分布）」の間の違いを計算して、それを数値で表している
    def _gaussian_KLD(self, dist1, dist2):
        # dist1の「平均」と「標準偏差」を使って、mvn1というガウス分布を作る
        mvn1 = dist.Normal(dist1["mean"], dist1["stddev"])
        # dist2の「平均」と「標準偏差」を使って、mvn2というガウス分布を作る
        mvn2 = dist.Normal(dist2["mean"], dist2["stddev"])
        # 計算したKLダイバージェンスを全部足し合わせる
        return dist.kl_divergence(mvn1, mvn2).sum(dim=-1)  # KLダイバージェンスを計算して返す

# 動画の中で、時間が進むごとにどんな変化が起きていくか」を計算するための仕組みを作っている
# 時間ごとに少しずつ変わるものを計算して、それを集める役割
def manual_scan(cell, obs_inputs, context, reset_state, use_observation, initial):
    priors = []  # 予測を保存するリスト
    posteriors = []  # 後方推定を保存するリスト
    prev_out = {"state": initial}  # 前回の出力を初期状態で設定
    seq_len = obs_inputs.size(1)  # 入力の長さを取得

    for t in range(seq_len):  # 各タイムステップに対して
        inputs = (
            obs_inputs[:, t],  # 現在の観察入力
            context[:, t],  # 現在のコンテキスト
            reset_state[:, t],  # 現在のリセット状態
        )
        outputs = cell(prev_out, inputs, use_observation)  # セルに入力を渡す
        priors.append(outputs["out"][0])  # 予測を追加
        posteriors.append(outputs["out"][1])  # 後方推定を追加
        prev_out = outputs  # 前回の出力を更新

    # すべての予測と後方推定をスタックして返す
    prior = {k: torch.stack([p[k] for p in priors], dim=1) for k in priors[0]}
    posterior = {k: torch.stack([p[k] for p in posteriors], dim=1) for k in posteriors[0]}
    posterior_last_step = prev_out["state"]  # 最後の状態を取得
    return prior, posterior, posterior_last_step  # 予測と後方推定、最後の状態を返す

# モデルを構築する関数
def build_model(cfg, open_loop=True):
    device = cfg['device']  # デバイスを取得

    # CWVAEモデルを初期化
    model = CWVAE(
        levels=cfg['levels'],  # 階層の数
        tmp_abs_factor=cfg['tmp_abs_factor'],  # 時間の絶対的な因子
        state_sizes={"stoch": cfg['cell_stoch_size'], "deter": cfg['cell_deter_size']},  # 状態のサイズ
        embed_size=cfg['cell_embed_size'],  # 埋め込みサイズ
        obs_embed_size=cfg['enc_dense_embed_size'],  # 観察の埋め込みサイズ
        enc_dense_layers=cfg['enc_dense_layers'],  # エンコーダーの密な層の数
        enc_dense_embed_size=cfg['enc_dense_embed_size'],  # エンコーダーの埋め込みサイズ
        channels_mult=cfg['channels_mult'],  # チャンネルの倍率
        device=device,  # デバイスを設定
        cell_type=cfg['cell_type'],  # セルのタイプ
        min_stddev=cfg['cell_min_stddev'],  # 最小の標準偏差
        mean_only_cell=cfg['cell_mean_only'],  # 平均のみのセルを使うか
        reset_states=cfg['cell_reset_state'],  # 状態をリセットするか
    ).to(device)  # モデルをデバイスに移動

    model.apply(model.init_weights)  # モデルの重みを初期化

    # 初期観察データを作成
    obs = torch.zeros([cfg['batch_size'], cfg['seq_len'], cfg['channels'], 64, 64]).to(device)
    obs_encoded = model.encoder(obs)  # エンコーダーで観察をエンコード

    # エンコーダーの出力が期待する階層数と一致するかを確認
    if len(obs_encoded) != cfg['levels']:
        raise ValueError(f"Encoder output does not match expected levels. Expected {cfg['levels']}, but got {len(obs_encoded)}.")

    # 階層的に情報を伝えて予測を行う
    outputs_bot, last_state_all_levels, priors, posteriors = model.hierarchical_unroll(obs_encoded)
    obs_decoded = model.decoder(outputs_bot)[0]  # デコーダーでデコード

    # 損失を計算
    losses = model.compute_losses(
        obs,
        obs_decoded,
        priors,
        posteriors,
        dec_stddev=cfg['dec_stddev'],  # デコードの標準偏差
        free_nats=cfg['free_nats'],  # フリーナッツ
        beta=cfg['beta'],  # ベータ
    )

    if open_loop:  # オープンループの場合
        ctx_len = cfg['open_loop_ctx']  # コンテキスト長を取得
        pre_posteriors, pre_priors, post_priors, _ = model.open_loop_unroll(
            obs_encoded, ctx_len=ctx_len, use_observations=cfg.get('use_obs', True)  # オープンループでの予測
        )
        prior_multistep_decoded = model.decode_prior_multistep(post_priors[0]["mean"])  # マルチステップの予測をデコード
        open_loop_obs_decoded = {
            "posterior_recon": model.decoder(pre_posteriors[0]["det_out"]),  # 後方推定をデコード
            "prior_multistep": prior_multistep_decoded,  # マルチステップ予測
            "gt_multistep": obs[:, ctx_len:, ...],  # グラウンドトゥルース
        }
    else:
        open_loop_obs_decoded = None  # オープンループでない場合はNoneを設定

    # モデルの情報を返す
    return {
        "training": {
            "obs": obs,  # 観察データ
            "encoder": model.encoder,  # エンコーダー
            "decoder": model.decoder,  # デコーダー
            "obs_encoded": obs_encoded,  # エンコードされた観察データ
            "obs_decoded": obs_decoded,  # デコードされた観察データ
            "priors": priors,  # 予測
            "posteriors": posteriors,  # 後方推定
            "loss": losses["loss"],  # 損失
            "nll_term": losses["nll_term"],  # 負の対数尤度
            "kl_term": losses["kl_term"],  # KLダイバージェンス
            "kld_all_levels": losses["kld_all_levels"],  # 各階層のKLD
        },
        "meta": {"model": model},  # モデルのメタ情報
        "open_loop_obs_decoded": open_loop_obs_decoded,  # オープンループのデコード結果
    }
