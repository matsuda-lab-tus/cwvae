# エンコーダーは動画のフレームから特徴を抽出し、潜在空間に変換する役割を果たし、デコーダーは潜在空間から元の画像を再構築する役割を果たします。
import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, levels, tmp_abs_factor, dense_layers=3, embed_size=1024, channels_mult=2):
        # channels_mult:畳み込みおよびデコンボリューションの層で使用するフィルタの数を拡張するための倍率です。この値を大きくすることで、より多くのフィルタを適用し、特徴抽出の能力を高めることができます。例えば、channels_mult=2 に設定すると、フィルタ数が2倍になります。
        # dense_layers:全結合層の数を調整します。通常、深いモデルを使用すると、モデルのキャパシティが増加し、より複雑なパターンを学習できますが、過学習のリスクも伴います。
        # エンコーダーは、動画フレームから特徴を抽出し、全結合層を通して潜在空間に変換します。
        # embed_size: 潜在空間の次元数です。エンコーダーが抽出する特徴のサイズに対応します。embed_size=800 の場合、800次元のベクトルに特徴がまとめられます。
        super(Encoder, self).__init__()
        self.levels = levels # モデルにおける階層の数です。これは、異なる時間スケールの特徴を学習するために使われます。
        self.tmp_abs_factor = tmp_abs_factor # 時間抽象化の倍率。これは各レベルでどれだけ時間を抽象化するかを示します。
        self.activation = nn.LeakyReLU(negative_slope=0.01)
        self.channels_mult = channels_mult

        # 畳み込み層の定義
        filters = 32
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=channels_mult * filters, kernel_size=4, stride=2, padding=1)
        # in_channels=3:入力が3チャンネルのRGB画像を示しています。
        # out_channels=channels_mult * filters: 出力する特徴マップの数をフィルタ数の倍で決めています（channels_mult * filters）。これにより、各層で扱う特徴の次元数を制御しています。
        self.conv2 = nn.Conv2d(in_channels=channels_mult * filters, out_channels=channels_mult * filters * 2, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(in_channels=channels_mult * filters * 2, out_channels=channels_mult * filters * 4, kernel_size=4, stride=2, padding=1)
        self.conv4 = nn.Conv2d(in_channels=channels_mult * filters * 4, out_channels=channels_mult * filters * 8, kernel_size=4, stride=2, padding=1)

        # 全結合層の定義
        self.dense_layers = nn.ModuleList()
        in_features = channels_mult * filters * 8 * 4 * 4  # 畳み込み層の出力サイズに基づく
        for _ in range(dense_layers):
            dense_layer = nn.Linear(in_features, embed_size)
            self.dense_layers.append(dense_layer)
            in_features = embed_size  # 次の層の入力は現在の出力サイズ

        # 潜在変数の平均と分散を出力する層の定義
        self.output_layers_mu = nn.ModuleList()
        self.output_layers_logvar = nn.ModuleList()
        for _ in range(levels):
            self.output_layers_mu.append(nn.Linear(embed_size, embed_size))
            self.output_layers_logvar.append(nn.Linear(embed_size, embed_size))

    def forward(self, obs):
        # obs の形状: [batch_size, timesteps, channels, height, width]
        batch_size, timesteps, channels, height, width = obs.size()

        # 時間次元をバッチ次元に結合
        obs = obs.view(batch_size * timesteps, channels, height, width)

        # 畳み込み層の処理
        hidden = self.activation(self.conv1(obs))
        hidden = self.activation(self.conv2(hidden))
        hidden = self.activation(self.conv3(hidden))
        hidden = self.activation(self.conv4(hidden))

        # 畳み込み層の出力形状を確認
        hidden = hidden.view(batch_size * timesteps, -1)

        # 全結合層の処理
        for dense_layer in self.dense_layers:
            hidden = self.activation(dense_layer(hidden))

        # レベルごとの出力を作成
        outputs_mu = []
        outputs_logvar = []
        for i in range(self.levels):
            output_mu = self.output_layers_mu[i](hidden)
            output_logvar = self.output_layers_logvar[i](hidden)
            # 元の形状に戻す
            output_mu = output_mu.view(batch_size, timesteps, -1)
            output_logvar = output_logvar.view(batch_size, timesteps, -1)
            outputs_mu.append(output_mu)
            outputs_logvar.append(output_logvar)
            print(f"Level {i}: Output_mu shape: {output_mu.shape}, Output_logvar shape: {output_logvar.shape}")
        
        return outputs_mu, outputs_logvar


class Decoder(nn.Module):
    def __init__(self, output_channels, embed_size, channels_mult=1):
        super(Decoder, self).__init__()
        self.channels_mult = channels_mult
        self._out_channels = output_channels
        self.embed_size = embed_size

        # 全結合層の定義
        self.fc = nn.Linear(embed_size, channels_mult * 256 * 4 * 4)

        # デコンボリューション層の定義
        self.deconv1 = nn.ConvTranspose2d(channels_mult * 256, channels_mult * 128, kernel_size=4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(channels_mult * 128, channels_mult * 64, kernel_size=4, stride=2, padding=1)
        self.deconv3 = nn.ConvTranspose2d(channels_mult * 64, channels_mult * 32, kernel_size=4, stride=2, padding=1)
        self.deconv4 = nn.ConvTranspose2d(channels_mult * 32, self._out_channels, kernel_size=4, stride=2, padding=1)

        self.activation = nn.LeakyReLU(negative_slope=0.01)
        self.out_activation = nn.Tanh()

    def forward(self, states):
        # states のサイズは (batch_size, timesteps, embed_size)
        batch_size, timesteps, feature_dim = states.size()
        
        # 全結合層に入力するためにフラット化する
        states = states.view(batch_size * timesteps, feature_dim)  # フラット化: [B*T, embed_size]
        
        # 全結合層の処理
        hidden = self.fc(states)  # [B*T, 256*channels_mult*4*4]

        # デコンボリューション層の処理
        hidden = hidden.view(batch_size * timesteps, 256 * self.channels_mult, 4, 4)  # [B*T, 256*channels_mult, 4, 4]
        hidden = self.activation(self.deconv1(hidden))  # [B*T, 128*channels_mult, 8, 8]
        hidden = self.activation(self.deconv2(hidden))  # [B*T, 64*channels_mult, 16, 16]
        hidden = self.activation(self.deconv3(hidden))  # [B*T, 32*channels_mult, 32, 32]
        out = self.out_activation(self.deconv4(hidden))  # [B*T, output_channels, 64, 64]

        # 出力を元の形状に戻す
        out = out.view(batch_size, timesteps, self._out_channels, 64, 64)  # [B, T, output_channels, 64, 64]
        print(f"Final output shape: {out.shape}")

        return out
