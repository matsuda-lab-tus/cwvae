import torch
import torch.nn as nn

# Encoder: e_l^t = e(x_t:t+k^l-1)
class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Encoder, self).__init__()
        self.rnn = nn.GRU(input_dim, hidden_dim)

    def forward(self, x_seq):
        # x_seqはシーケンスデータ (x_t:t+k^l-1)
        _, h_n = self.rnn(x_seq)  # GRUの最後の隠れ状態を取得
        return h_n

# Posterior transition: q_l^t(s_l^t | s_l^t-1, s_l+1^t, e_l^t)
class PosteriorTransition(nn.Module):
    def __init__(self, latent_dim, hidden_dim):
        super(PosteriorTransition, self).__init__()
        self.fc = nn.Linear(latent_dim * 2 + hidden_dim, latent_dim)

    def forward(self, s_l_prev, s_l_next, e_l):
        # s_l_prev, s_l_next: 前と後の潜在状態, e_l: エンコーダ出力
        
        # e_lは次元が異なるため、次元を合わせる
        if e_l.dim() == 3:  # 3次元ならバッチ次元とタイムステップ次元を考慮
            e_l = e_l.squeeze(0)  # バッチ次元を削除
        
        input_data = torch.cat([s_l_prev, s_l_next, e_l], dim=-1)
        return self.fc(input_data)


# Prior transition: p_l^t(s_l^t | s_l^t-1, s_l+1^t)
class PriorTransition(nn.Module):
    def __init__(self, latent_dim):
        super(PriorTransition, self).__init__()
        self.fc = nn.Linear(latent_dim * 2, latent_dim)

    def forward(self, s_l_prev, s_l_next):
        # s_l_prev: 前の潜在状態, s_l_next: 次の潜在状態
        input_data = torch.cat([s_l_prev, s_l_next], dim=-1)
        return self.fc(input_data)

# Decoder: p(x_t | s_1^t)
class Decoder(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super(Decoder, self).__init__()
        self.fc = nn.Linear(latent_dim, output_dim)

    def forward(self, s_1):
        # s_1: 最下位レベルの潜在変数
        return self.fc(s_1)

# モデルのサイズ設定
input_dim = 128  # 入力画像の特徴量次元
hidden_dim = 64  # エンコーダの隠れ層次元
latent_dim = 32  # 潜在変数の次元
output_dim = 128  # デコーダの出力次元 (再構成された画像の特徴量次元)

# モデルインスタンスの作成
encoder = Encoder(input_dim, hidden_dim)
posterior_transition = PosteriorTransition(latent_dim, hidden_dim)
prior_transition = PriorTransition(latent_dim)
decoder = Decoder(latent_dim, output_dim)

# サンプルデータ
x_seq = torch.randn(10, 1, input_dim)  # サンプルシーケンスデータ
s_l_prev = torch.randn(1, latent_dim)  # 前の潜在状態
s_l_next = torch.randn(1, latent_dim)  # 次の潜在状態

# 各コンポーネントの実行
e_l = encoder(x_seq)  # エンコーダ出力
posterior_output = posterior_transition(s_l_prev, s_l_next, e_l)
prior_output = prior_transition(s_l_prev, s_l_next)
decoded_output = decoder(s_l_prev)

print("Encoder output:", e_l)
print("Posterior transition output:", posterior_output)
print("Prior transition output:", prior_output)
print("Decoder output:", decoded_output)
