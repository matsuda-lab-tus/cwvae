import torch  # PyTorchライブラリをインポート
import torch.nn as nn  # ニューラルネットワークの基本機能を使います
import torch.nn.functional as F  # ニューラルネットワークのための便利な関数を使います
from torch.distributions import MultivariateNormal  # 多変量正規分布を使います


# 「再帰型状態空間モデル（Recurrent State Space Model）」のセルを定義します
class RSSMCell(nn.Module):
    def __init__(
        self,
        state_size,
        detstate_size,
        embed_size,
        reset_states=False,
        min_stddev=0.0001,
        mean_only=False,
        var_scope="rssm_cell",
    ):
        super(RSSMCell, self).__init__()
        
        # パラメータの設定
        self._state_size = state_size
        self._detstate_size = detstate_size
        self._embed_size = embed_size
        self._min_stddev = min_stddev
        self._mean_only = mean_only
        self._reset_states = reset_states

        # GRUセルの定義
        self._cell = nn.GRUCell(input_size=self._embed_size, hidden_size=self._detstate_size)

        # Prior用の全結合層の定義
        self.prior_h1_dense = nn.Linear(self._state_size + self._detstate_size, self._embed_size)
        self.prior_h2_dense = nn.Linear(self._embed_size, self._embed_size)
        self.prior_mean_dense = nn.Linear(self._embed_size, self._state_size)
        self.prior_stddev_dense = nn.Linear(self._embed_size, self._state_size)

        # Posterior用の全結合層の定義
        self.posterior_h1_dense = nn.Linear(self._detstate_size + self._embed_size, self._embed_size)
        self.posterior_h2_dense = nn.Linear(self._embed_size, self._embed_size)
        self.posterior_mean_dense = nn.Linear(self._embed_size, self._state_size)
        self.posterior_stddev_dense = nn.Linear(self._embed_size, self._state_size)

    def _prior(self, prev_state, context):
        """
        事前分布を計算します。
        """
        # `prev_state["sample"]`と`context`を連結して入力とする
        inputs = torch.cat([prev_state["sample"], context], dim=-1)

        # 隠れ層1を通す
        hl = F.relu(self.prior_h1_dense(inputs))

        # GRUセルを使用してdet_outとdet_stateを計算
        det_out = self._cell(hl, prev_state["det_state"])
        det_state = det_out  # det_stateをdet_outに合わせて更新

        # det_outを次の隠れ層に入力
        hl = F.relu(self.prior_h2_dense(det_out))

        # 平均と標準偏差を計算
        mean = self.prior_mean_dense(hl)
        stddev = F.softplus(self.prior_stddev_dense(hl) + 0.54) + self._min_stddev

        # 平均のみを使用する場合
        if self._mean_only:
            sample = mean
        else:
            # PyTorchでのMultivariateNormalによるサンプリング
            sample = MultivariateNormal(mean, torch.diag_embed(stddev)).sample()

        # 結果をリターン
        return {
            "mean": mean,
            "stddev": stddev,
            "sample": sample,
            "det_out": det_out,
            "det_state": det_state,
            "output": torch.cat([sample, det_out], dim=-1),
        }


    def _posterior(self, obs_inputs, prev_state, context):
        """
        事後分布を計算します。
        """
        # 事前分布の計算
        prior = self._prior(prev_state, context)
        
        # Posteriorの入力としてprior["det_out"]と観測入力を連結
        inputs = torch.cat([prior["det_out"], obs_inputs], dim=-1)
        
        # 隠れ層を通す
        hl = F.relu(self.posterior_h1_dense(inputs))
        hl = F.relu(self.posterior_h2_dense(hl))
        
        # 平均と標準偏差の計算
        mean = self.posterior_mean_dense(hl)
        stddev = F.softplus(self.posterior_stddev_dense(hl) + 0.54) + self._min_stddev
        
        # 平均のみを使う場合
        if self._mean_only:
            sample = mean
        else:
            sample = MultivariateNormal(mean, torch.diag_embed(stddev)).sample()

        return {
            "mean": mean,
            "stddev": stddev,
            "sample": sample,
            "det_out": prior["det_out"],
            "det_state": prior["det_state"],
            "output": torch.cat([sample, prior["det_out"]], dim=-1),
        }

    def forward(self, prev_out, inputs, use_obs):
        """
        前の出力と現在の入力を使って事前分布と事後分布を計算します。
        """
        prev_state = prev_out["state"]
        obs_input, context, reset_state = inputs

        # 状態をリセットするかどうかをチェック
        if not self._reset_states:
            reset_state = torch.ones_like(reset_state)
        prev_state["sample"] *= reset_state

        # Prior計算
        prior = self._prior(prev_state, context)

        # Posterior計算
        if use_obs:
            posterior = self._posterior(obs_input, prev_state, context)
        else:
            posterior = prior

        return {"out": (prior, posterior), "state": posterior}

    def zero_state(self, batch_size, device):
        """
        初期状態をゼロで設定し、stateキーを含む辞書として返す。
        """
        return {
            "state": {
                "sample": torch.zeros(batch_size, self._state_size, device=device),
                "det_state": torch.zeros(batch_size, self._detstate_size, device=device),
            }
        }


    def zero_out_state(self, batch_size, device):
        """
        zero_stateに基づき、アウトプット状態を初期化（先行研究に準拠）。
        """
        zero_st = self.zero_state(batch_size, device)
        return {"out": (zero_st, zero_st), "state": zero_st}
