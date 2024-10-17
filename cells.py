import torch
import torch.nn as nn
import torch.nn.functional as F

class RSSMCell(nn.Module):
    def __init__(
        self,
        stoch_size,
        deter_size,
        embed_size,
        obs_embed_size,  # 追加された引数
        reset_states=False,
        min_stddev=0.0001,
        mean_only=False,
    ):
        super(RSSMCell, self).__init__()
        self._state_size = stoch_size
        self._detstate_size = deter_size
        self._embed_size = embed_size
        self._obs_embed_size = obs_embed_size  # 追加された属性

        self._min_stddev = min_stddev
        self._mean_only = mean_only
        self._reset_states = reset_states

        # 決定論的部分のためのGRUセル
        self._cell = nn.GRUCell(input_size=self._embed_size, hidden_size=self._detstate_size)

        # Prior（事前分布）のための全結合層
        self.prior_h1_dense = nn.Linear(
            self._state_size + self._detstate_size, self._embed_size
        )
        self.prior_h2_dense = nn.Linear(self._embed_size, self._embed_size)
        self.prior_mean_dense = nn.Linear(self._embed_size, self._state_size)
        self.prior_stddev_dense = nn.Linear(self._embed_size, self._state_size)

        # 観測入力のための埋め込み層
        self.obs_embed_dense = nn.Linear(4096, self._embed_size)  # 入力サイズは4096

        # Posterior（事後分布）のための全結合層
        self.posterior_h1_dense = nn.Linear(
            self._detstate_size + self._embed_size, self._embed_size
        )
        self.posterior_h2_dense = nn.Linear(self._embed_size, self._embed_size)
        self.posterior_mean_dense = nn.Linear(self._embed_size, self._state_size)
        self.posterior_stddev_dense = nn.Linear(self._embed_size, self._state_size)

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
        prev_state = prev_out["state"]
        context = inputs[1]

        # Contextのサイズを調整してPriorおよびPosteriorの計算に使えるようにする
        if context.dim() == 3 and context.size(1) == 1:
            context = context.squeeze(1)  # [batch, context_size]

        # Prior（事前分布）の計算
        prior = self._prior(prev_state, context)

        # Posterior（事後分布）の計算
        if use_observation:
            posterior = self._posterior(inputs[0], prior, context)
        else:
            posterior = prior

        # 修正ポイント: 'state' に 'posterior' 全体を設定
        return {"out": [prior, posterior], "state": posterior}

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
        hl = F.relu(self.prior_h1_dense(inputs))  # [batch, embed_size]
        hl = F.relu(self.prior_h2_dense(hl))     # [batch, embed_size]

        mean = self.prior_mean_dense(hl)         # [batch, state_size]
        stddev = F.softplus(self.prior_stddev_dense(hl) + 0.54) + self._min_stddev  # [batch, state_size]

        if self._mean_only:
            sample = mean
        else:
            sample = mean + stddev * torch.randn_like(stddev)  # [batch, state_size]

        # GRUCellに[batch, embed_size]のhlを渡す
        det_state = self._cell(hl, prev_state["det_state"])  # [batch, deter_size]

        return {
            "mean": mean,
            "stddev": stddev,
            "sample": sample,
            "det_out": det_state,
            "det_state": det_state,
            "output": torch.cat([sample, det_state], dim=-1),
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
        embedded_obs = F.relu(self.obs_embed_dense(obs_inputs))  # [batch, embed_size]

        # Priorの決定論的出力と埋め込み観測データを結合
        inputs = torch.cat([prior["det_out"], embedded_obs], dim=-1)  # [batch, 800 + 800]
        hl = F.relu(self.posterior_h1_dense(inputs))  # [batch, embed_size]
        hl = F.relu(self.posterior_h2_dense(hl))     # [batch, embed_size]

        mean = self.posterior_mean_dense(hl)         # [batch, state_size]
        stddev = F.softplus(self.posterior_stddev_dense(hl) + 0.54) + self._min_stddev  # [batch, state_size]

        if self._mean_only:
            sample = mean
        else:
            sample = mean + stddev * torch.randn_like(stddev)  # [batch, state_size]

        # GRUCellに[batch, embed_size]のhlを渡す
        det_state = self._cell(hl, prior["det_state"])  # [batch, deter_size]

        return {
            "mean": mean,
            "stddev": stddev,
            "sample": sample,
            "det_out": det_state,
            "det_state": det_state,
            "output": torch.cat([sample, det_state], dim=-1),
        }

    def zero_state(self, batch_size, device):
        """
        初期状態をゼロで設定します。
        
        Args:
            batch_size (int): バッチサイズ。
            device (torch.device): デバイス情報。
        
        Returns:
            dict: 初期状態を含む辞書。
        """
        return {
            "sample": torch.zeros(batch_size, self._state_size).to(device),
            "det_state": torch.zeros(batch_size, self._detstate_size).to(device)
        }

    def init_weights(self, module):
        """
        モジュールの重みを初期化します。
        
        Args:
            module (nn.Module): 重みを初期化するモジュール。
        """
        if isinstance(module, nn.Linear):
            nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.ConvTranspose2d):
            nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.GRUCell):
            for name, param in module.named_parameters():
                if 'weight' in name:
                    nn.init.kaiming_uniform_(param, nonlinearity='relu')
                elif 'bias' in name:
                    nn.init.constant_(param, 0)
    