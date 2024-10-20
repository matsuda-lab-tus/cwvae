import torch  # PyTorchライブラリをインポート
import torch.nn as nn  # ニューラルネットワークの基本機能を使います
import torch.nn.functional as F  # ニューラルネットワークのための便利な関数を使います

# 「再帰型状態空間モデル（Recurrent State Space Model）」のセルを定義します
class RSSMCell(nn.Module):  # nn.Module は、ニューラルネットワークの基本的な部品を表すPyTorchのクラス
    def __init__(
        self,
        stoch_size,  # ランダムな部分のサイズ
        deter_size,  # 決定的な部分のサイズ
        embed_size,  # データを変換する際の埋め込みサイズ
        obs_embed_size,  # 観測データを埋め込むためのサイズ
        reset_states=False,  # 状態をリセットするかどうか
        min_stddev=0.0001,  # 標準偏差の最小値
        mean_only=False,  # 平均値だけを使うかどうか
    ):
        super(RSSMCell, self).__init__()  # 親クラスを初期化
        # 各サイズをインスタンス変数に保存
        self._state_size = stoch_size
        self._detstate_size = deter_size
        self._embed_size = embed_size
        self._obs_embed_size = obs_embed_size  

        self._min_stddev = min_stddev  # 最小標準偏差を設定
        self._mean_only = mean_only  # 平均のみを使用するかのフラグ
        self._reset_states = reset_states  # 状態リセットのフラグ

        # 決定論的部分のためのGRUセルを定義
        self._cell = nn.GRUCell(input_size=self._embed_size, hidden_size=self._detstate_size)  # GRUセルを作成

        # 事前分布（Prior）を計算するための全結合層
        self.prior_h1_dense = nn.Linear(self._state_size + self._detstate_size, self._embed_size)  # 1層目
        self.prior_h2_dense = nn.Linear(self._embed_size, self._embed_size)  # 2層目
        self.prior_mean_dense = nn.Linear(self._embed_size, self._state_size)  # 平均を計算する層
        self.prior_stddev_dense = nn.Linear(self._embed_size, self._state_size)  # 標準偏差を計算する層

        # 観測入力のための埋め込み層
        self.obs_embed_dense = nn.Linear(4096, self._embed_size)  # 観測データを埋め込む層（入力サイズは4096）

        # 事後分布（Posterior）を計算するための全結合層
        self.posterior_h1_dense = nn.Linear(self._detstate_size + self._embed_size, self._embed_size)  # 1層目
        self.posterior_h2_dense = nn.Linear(self._embed_size, self._embed_size)  # 2層目
        self.posterior_mean_dense = nn.Linear(self._embed_size, self._state_size)  # 平均を計算する層
        self.posterior_stddev_dense = nn.Linear(self._embed_size, self._state_size)  # 標準偏差を計算する層

    # 前の状態と新しい入力（観測データやコンテキスト）を使って事前分布を計算
    def forward(self, prev_out, inputs, use_observation): 
        """
        前の出力と現在の入力を受け取り、事前分布と事後分布を計算します。

        Args:
            prev_out (dict): 前の状態を含む辞書。キーは "state"。
            inputs (list): 入力のリスト。inputs[0] は観測データ、inputs[1] はコンテキスト。
            use_observation (bool): 観測を使用するかどうか。

        Returns:
            dict: 新しい状態と事前・事後分布の情報を含む辞書。
        """
        # prev_outは辞書 {"state": ...} であると想定
        prev_state = prev_out["state"]  # 前の状態を取得
        context = inputs[1]  # コンテキストを取得

        # Contextのサイズを調整
        if context.dim() == 3 and context.size(1) == 1:
            context = context.squeeze(1)  # [batch, context_size]に変形

        # Prior（事前分布）の計算
        prior = self._prior(prev_state, context)  # 事前分布を計算

        # Posterior（事後分布）の計算
        if use_observation:  # 観測データがあれば事後分布を計算
            posterior = self._posterior(inputs[0], prior, context)
        else:
            posterior = prior  # 観測がない場合は事前分布を使用

        # 'state' に 'posterior' 全体を設定
        return {"out": [prior, posterior], "state": posterior}  # 事前分布と事後分布を返す

    def _prior(self, prev_state, context):
        """
        事前分布を計算します。

        Args:
            prev_state (dict): 前の状態。キーは "sample" と "det_state"。
            context (Tensor): コンテキスト情報。

        Returns:
            dict: 事前分布の情報を含む辞書。
        """
        # 前の状態とコンテキストを結合
        inputs = torch.cat([prev_state["sample"], context], dim=-1)  # [batch, 100 + 800]
        hl = F.relu(self.prior_h1_dense(inputs))  # 1層目を通す
        hl = F.relu(self.prior_h2_dense(hl))     # 2層目を通す

        mean = self.prior_mean_dense(hl)         # 平均を計算
        # 標準偏差を計算し、最小値を適用
        stddev = F.softplus(self.prior_stddev_dense(hl) + 0.54) + self._min_stddev  # [batch, state_size]

        if self._mean_only:  # 平均のみを使う場合
            sample = mean
        else:
            sample = mean + stddev * torch.randn_like(stddev)  # サンプリングを行う

        # GRUCellに[batch, embed_size]のhlを渡す
        det_state = self._cell(hl, prev_state["det_state"])  # GRUセルを通して状態を更新

        return {
            "mean": mean,  # 事前分布の平均
            "stddev": stddev,  # 事前分布の標準偏差
            "sample": sample,  # 事前分布からのサンプリング結果
            "det_out": det_state,  # 決定的な出力
            "det_state": det_state,  # 決定的な状態
            "output": torch.cat([sample, det_state], dim=-1),  # サンプリング結果と決定的な出力を結合
        }

    def _posterior(self, obs_inputs, prior, context):
        """
        事後分布を計算します。

        Args:
            obs_inputs (Tensor): 観測データの入力。
            prior (dict): 事前分布の情報を含む辞書。
            context (Tensor): コンテキスト情報。

        Returns:
            dict: 事後分布の情報を含む辞書。
        """
        # 観測データの埋め込み
        embedded_obs = F.relu(self.obs_embed_dense(obs_inputs))  # 埋め込みを計算

        # Priorの決定論的出力と埋め込み観測データを結合
        inputs = torch.cat([prior["det_out"], embedded_obs], dim=-1)  # [batch, 800 + 800]
        hl = F.relu(self.posterior_h1_dense(inputs))  # 1層目を通す
        hl = F.relu(self.posterior_h2_dense(hl))     # 2層目を通す

        mean = self.posterior_mean_dense(hl)         # 事後分布の平均を計算
        stddev = F.softplus(self.posterior_stddev_dense(hl) + 0.54) + self._min_stddev  # 標準偏差を計算

        if self._mean_only:  # 平均のみを使う場合
            sample = mean
        else:
            sample = mean + stddev * torch.randn_like(stddev)  # サンプリングを行う

        # GRUCellに[batch, embed_size]のhlを渡す
        det_state = self._cell(hl, prior["det_state"])  # GRUセルを通して状態を更新

        return {
            "mean": mean,  # 事後分布の平均
            "stddev": stddev,  # 事後分布の標準偏差
            "sample": sample,  # 事後分布からのサンプリング結果
            "det_out": det_state,  # 決定的な出力
            "det_state": det_state,  # 決定的な状態
            "output": torch.cat([sample, det_state], dim=-1),  # サンプリング結果と決定的な出力を結合
        }

    def zero_state(self, batch_size, device):  # ゼロから始めるための初期状態を作成
        """
        初期状態をゼロで設定します。

        Args:
            batch_size (int): バッチサイズ。
            device (torch.device): デバイス情報。

        Returns:
            dict: 初期状態を含む辞書。
        """
        return {
            "sample": torch.zeros(batch_size, self._state_size).to(device),  # サンプルの初期状態
            "det_state": torch.zeros(batch_size, self._detstate_size).to(device)  # 決定的状態の初期状態
        }

    def init_weights(self, module):  # モデルが学習を始めやすいように重みを初期化
        """
        モジュールの重みを初期化します。

        Args:
            module (nn.Module): 重みを初期化するモジュール。
        """
        if isinstance(module, nn.Linear):  # 線形層の場合
            nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')  # Kaiming初期化
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)  # バイアスを0で初期化
        elif isinstance(module, nn.ConvTranspose2d):  # 転置畳み込み層の場合
            nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')  # Kaiming初期化
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)  # バイアスを0で初期化
        elif isinstance(module, nn.GRUCell):  # GRUセルの場合
            for name, param in module.named_parameters():  # 各パラメータを確認
                if 'weight' in name:
                    nn.init.kaiming_uniform_(param, nonlinearity='relu')  # Kaiming初期化
                elif 'bias' in name:
                    nn.init.constant_(param, 0)  # バイアスを0で初期化
