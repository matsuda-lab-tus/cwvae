import torch  # PyTorchライブラリを使います
import torch.nn as nn  # ニューラルネットワークの基本機能を使います
import torch.nn.functional as F  # ニューラルネットワークのための便利な関数を使います
import math  # 数学的な計算をするためのライブラリ

class Encoder(nn.Module):
    """
    Multi-level Video Encoder.
    """

    def __init__(
        self, levels, tmp_abs_factor, dense_layers=3, embed_size=1024, channels_mult=1
    ):
        super(Encoder, self).__init__()
        self._levels = levels
        self._tmp_abs_factor = tmp_abs_factor
        self._dense_layers = dense_layers
        self._embed_size = embed_size
        self._channels_mult = channels_mult
        self._activation = nn.LeakyReLU(negative_slope=0.2)

        # Conv layers setup
        filters = 32
        self.h1_conv = nn.Conv2d(
            in_channels=3, out_channels=channels_mult * filters, kernel_size=4, stride=2
        )
        self.h2_conv = nn.Conv2d(
            in_channels=channels_mult * filters,
            out_channels=channels_mult * filters * 2,
            kernel_size=4,
            stride=2,
        )
        self.h3_conv = nn.Conv2d(
            in_channels=channels_mult * filters * 2,
            out_channels=channels_mult * filters * 4,
            kernel_size=4,
            stride=2,
        )
        self.h4_conv = nn.Conv2d(
            in_channels=channels_mult * filters * 4,
            out_channels=channels_mult * filters * 8,
            kernel_size=4,
            stride=2,
        )
        self.conv_output_size = channels_mult * filters * 8 * 4  # Calculated output size

        # Fully connected (dense) layers for each level
        self.level_dense_layers = nn.ModuleList()
        for level in range(1, self._levels):
            level_layers = nn.ModuleList()
            in_features = self.conv_output_size
            for _ in range(self._dense_layers - 1):
                dense_layer = nn.Linear(in_features, self._embed_size)
                level_layers.append(dense_layer)
                in_features = self._embed_size
            dense_layer = nn.Linear(in_features, self.conv_output_size)
            level_layers.append(dense_layer)
            self.level_dense_layers.append(level_layers)

    def forward(self, obs):
        # Reshape to combine batch and time dimensions for convolution
        batch_size, seq_len, channels, height, width = obs.size()
        hidden = obs.reshape(batch_size * seq_len, channels, height, width)

        # Apply convolutional layers
        hidden = self._activation(self.h1_conv(hidden))
        hidden = self._activation(self.h2_conv(hidden))
        hidden = self._activation(self.h3_conv(hidden))
        hidden = self._activation(self.h4_conv(hidden))

        hidden = hidden.flatten(start_dim=1)  # Flatten the conv output
        hidden = hidden.view(batch_size, seq_len, -1)  # Reshape back to (B, T, :)

        layer = hidden  # Initialize current layer
        layers = [layer]  # Collect output for each level
        print(f"[DEBUG] Input shape at level 0: {layer.shape}")

        # Process each hierarchical level
        for level in range(1, self._levels):
            dense_layers = self.level_dense_layers[level - 1]
            for dense_layer in dense_layers[:-1]:  # Apply all but last dense layer
                layer = F.relu(dense_layer(layer))
            layer = dense_layers[-1](layer)  # Apply last dense layer

            # Temporal abstraction
            timesteps_to_merge = self._tmp_abs_factor
            timesteps_to_pad = (timesteps_to_merge - (layer.size(1) % timesteps_to_merge)) % timesteps_to_merge
            if timesteps_to_pad > 0:
                padding = torch.zeros(batch_size, timesteps_to_pad, layer.size(-1)).to(layer.device)
                layer = torch.cat([layer, padding], dim=1)
                print(f"[DEBUG] Padded {timesteps_to_pad} timesteps at level {level}")

            merged_timesteps = layer.size(1) // timesteps_to_merge
            layer = layer.view(batch_size, merged_timesteps, timesteps_to_merge, -1).sum(dim=2)
            layers.append(layer)  # Append current level's output
            print(f"[DEBUG] Input shape at level {level}: {layer.shape}")

        return layers


# デコーダーのクラスを定義します
class Decoder(nn.Module):
    """ States to Images Decoder """

    def __init__(self, out_channels, channels_mult=1):
        super(Decoder, self).__init__()
        self._out_channels = out_channels  # 出力チャンネル数
        self._channels_mult = channels_mult  # チャンネルの倍率
        self._activation = nn.LeakyReLU(negative_slope=0.2)
        self._out_activation = nn.Tanh()  # 出力層の活性化関数

        # フィルターの数を定義
        filters = 32

        # 畳み込み転置層を定義
        self.h1_dense = nn.Linear(1024, self._channels_mult * 1024)  # 全結合層
        self.h2_deconv = nn.ConvTranspose2d(
            in_channels=1024,
            out_channels=self._channels_mult * filters * 4,  # 128
            kernel_size=5,
            stride=2,
        )
        self.h3_deconv = nn.ConvTranspose2d(
            in_channels=self._channels_mult * filters * 4,  # 128
            out_channels=self._channels_mult * filters * 2,  # 64
            kernel_size=5,
            stride=2,
        )
        self.h4_deconv = nn.ConvTranspose2d(
            in_channels=self._channels_mult * filters * 2,  # 64
            out_channels=self._channels_mult * filters,  # 32
            kernel_size=6,
            stride=2,
        )
        self.out_deconv = nn.ConvTranspose2d(
            in_channels=self._channels_mult * filters,  # 32
            out_channels=self._out_channels,  # 出力チャンネル数
            kernel_size=6,
            stride=2,
        )

    def forward(self, states):
        batch_size, timesteps, feature_dim = states.size()
        print(f"[DEBUG] Decoder input shape: {states.shape}")

        # 全結合層で入力を処理
        hidden = self._activation(self.h1_dense(states))
        hidden = hidden.view(-1, 1024, 1, 1)  # (B * T, 1024, 1, 1)
        print(f"[DEBUG] After reshaping to (1x1): {hidden.shape}")

        # 畳み込み転置層を通してアップサンプリング
        hidden = self._activation(self.h2_deconv(hidden))
        print(f"[DEBUG] After h2_deconv: {hidden.shape}")
        
        hidden = self._activation(self.h3_deconv(hidden))
        print(f"[DEBUG] After h3_deconv: {hidden.shape}")
        
        hidden = self._activation(self.h4_deconv(hidden))
        print(f"[DEBUG] After h4_deconv: {hidden.shape}")

        # 出力層（最終の畳み込み転置層）
        out = self._out_activation(self.out_deconv(hidden))
        print(f"[DEBUG] After out_deconv (final layer): {out.shape}")

        # 元の次元に戻す
        out = out.view(batch_size, timesteps, self._out_channels, 64, 64)
        print(f"[DEBUG] Final output shape: {out.shape}")

        return out


# if __name__ == "__main__": # モデルのインスタンスを作成
#     encoder = Encoder(levels=3, tmp_abs_factor=6, dense_layers=3, embed_size=256, channels_mult=1)
#     decoder = Decoder(output_channels=3, embed_size=256, channels_mult=1)

#     # ダミーデータを作成
#     batch_size = 50  # バッチサイズ
#     seq_len = 100  # シーケンス長
#     obs = torch.randn(batch_size, seq_len, 3, 64, 64)  # 入力データ

#     # エンコーダーを通して処理
#     layers = encoder(obs)  # エンコーダーを通して処理
#     print(f"Number of layers: {len(layers)}")  # 出力の数を表示
#     for i, layer in enumerate(layers):  # 各出力に対して
#         print(f"Layer {i}: {layer.size()}")  # 出力の形を表示


#     # デコーダーを通して処理
#     obs = torch.randn(batch_size, seq_len, 256)  # 入力データ
    
#     outputs = decoder(obs)  # デコーダーを通して処理
    # print(f"Final output shape: {outputs.size()}")  # 最終出力の形を表示
    # for key, value in intermediate_outputs.items():  # 各中間出力に対して
    #     print(f"{key}: {value.size()}")  # 中間出力の形を表示
