import torch
import torch.nn as nn
import torch.nn.functional as F
import math  # 追加

class Encoder(nn.Module):
    def __init__(self, levels, tmp_abs_factor, dense_layers=3, embed_size=800, channels_mult=1):
        super(Encoder, self).__init__()
        self.levels = levels
        self.tmp_abs_factor = tmp_abs_factor
        self.activation = nn.LeakyReLU(negative_slope=0.01)
        self.channels_mult = channels_mult
        self.dense_layers_num = dense_layers
        self.embed_size = embed_size

        # 畳み込み層の定義
        filters = 8  # conv_output_size = channels_mult * filters * 8 * 4 * 4 = 1 * 8 * 8 * 4 * 4 = 1024
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=channels_mult * filters, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(in_channels=channels_mult * filters, out_channels=channels_mult * filters * 2, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(in_channels=channels_mult * filters * 2, out_channels=channels_mult * filters * 4, kernel_size=4, stride=2, padding=1)
        self.conv4 = nn.Conv2d(in_channels=channels_mult * filters * 4, out_channels=channels_mult * filters * 8, kernel_size=4, stride=2, padding=1)

        # 畳み込み層の出力サイズ計算（入力画像サイズが64x64の場合）
        self.conv_output_size = channels_mult * filters * 8 * 4 * 4  # 1 * 8 * 8 * 4 * 4 = 1024

        # レベルごとの全結合層の定義
        self.level_dense_layers = nn.ModuleList()
        for level in range(1, self.levels):
            level_layers = nn.ModuleList()
            in_features = self.conv_output_size
            for _ in range(self.dense_layers_num - 1):
                dense_layer = nn.Linear(in_features, self.embed_size)
                level_layers.append(dense_layer)
                in_features = self.embed_size  # 次の層の入力は現在の出力サイズ
            # 最後の層（再び conv_output_size に戻す）
            dense_layer = nn.Linear(in_features, self.conv_output_size)
            level_layers.append(dense_layer)
            self.level_dense_layers.append(level_layers)

    def forward(self, obs):
        """
        obs の形状: [batch_size, timesteps, channels, height, width]
        """
        batch_size, timesteps, channels, height, width = obs.size()

        # 時間次元をバッチ次元に結合
        obs = obs.reshape(batch_size * timesteps, channels, height, width)

        # 畳み込み層の処理
        hidden = self.activation(self.conv1(obs))  # [B*T, C, H, W]
        hidden = self.activation(self.conv2(hidden))
        hidden = self.activation(self.conv3(hidden))
        hidden = self.activation(self.conv4(hidden))

        # 畳み込み層の出力形状を確認
        hidden = hidden.reshape(batch_size, timesteps, -1)  # Shape: (batch_size, timesteps, feature_dim)
        layer = hidden

        layers = []
        layers.append(layer)  # レベル0の特徴量
        print(f"[DEBUG] Input shape at level 0: {layer.shape}")  # 例: [50, 100, 1024]

        feat_size = layer.size(-1)  # 1024

        for level in range(1, self.levels):
            # このレベルの全結合層を適用
            dense_layers = self.level_dense_layers[level - 1]
            hidden = layer
            for dense_layer in dense_layers[:-1]:
                hidden = F.relu(dense_layer(hidden))
            hidden = dense_layers[-1](hidden)  # 最後の層は活性化関数なし

            layer = hidden  # レイヤーを更新

            # 時間ステップのマージ
            timesteps_to_merge = self.tmp_abs_factor  # 各レベルでのマージ係数を固定
            current_timesteps = layer.size(1)
            timesteps_to_pad = (timesteps_to_merge - (current_timesteps % timesteps_to_merge)) % timesteps_to_merge
            if timesteps_to_pad > 0:
                padding = torch.zeros(batch_size, timesteps_to_pad, feat_size).to(layer.device)
                layer = torch.cat([layer, padding], dim=1)
                print(f"[DEBUG] Padded {timesteps_to_pad} timesteps at level {level}")

            # 時間ステップをマージ
            merged_timesteps = layer.size(1) // timesteps_to_merge
            layer = layer.view(batch_size, merged_timesteps, timesteps_to_merge, feat_size)
            # マージされた時間ステップで集約（例：合計）
            layer = layer.sum(dim=2)
            layers.append(layer)
            print(f"[DEBUG] Input shape at level {level}: {layer.shape}")  # 例: [50, 17, 1024]

        return layers

class Decoder(nn.Module):
    def __init__(self, output_channels, embed_size, channels_mult=1):
        super(Decoder, self).__init__()
        self.embed_size = embed_size
        self.output_channels = output_channels
        self.channels_mult = channels_mult
        
        # アップサンプリング層を4層に増やす（4x4 -> 8x8 -> 16x16 -> 32x32 -> 64x64）
        self.fc = nn.Linear(embed_size, 512 * 4 * 4)  # [B*T, 512*4*4]
        self.deconv1 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1)  # 4x4 -> 8x8
        self.deconv2 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)  # 8x8 -> 16x16
        self.deconv3 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)   # 16x16 -> 32x32
        self.deconv4 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)    # 32x32 -> 64x64
        self.deconv5 = nn.ConvTranspose2d(32, output_channels, kernel_size=3, stride=1, padding=1)  # 64x64 -> 64x64
        
        self.activation = nn.ReLU()
        self.final_activation = nn.Sigmoid()  # 出力を0-1にスケーリング
    
    def forward(self, x):
        batch_size, timesteps, embed_size = x.size()
        print(f"[DEBUG] Input to Decoder: {x.shape}, min: {x.min()}, max: {x.max()}")

        # fc層に入力するためのデータ形状を確認
        x = x.view(batch_size * timesteps, embed_size)  # [B*T, embed_size]
        print(f"[DEBUG] After reshaping for fc layer: {x.shape}, min: {x.min()}, max: {x.max()}")

        # fc層を通過
        x = self.activation(self.fc(x))  # [B*T, 512*4*4]
        print(f"[DEBUG] After fc: {x.shape}, min: {x.min()}, max: {x.max()}")

        # 4x4の形状に変換
        x = x.view(batch_size * timesteps, 512, 4, 4)  # [B*T, 512, 4, 4]
        print(f"[DEBUG] After reshaping to 4x4: {x.shape}, min: {x.min()}, max: {x.max()}")

        # deconv1層を通過（4x4 -> 8x8）
        x = self.activation(self.deconv1(x))  # [B*T, 256, 8, 8]
        print(f"[DEBUG] After deconv1 (8x8): {x.shape}, min: {x.min()}, max: {x.max()}")

        # deconv2層を通過（8x8 -> 16x16）
        x = self.activation(self.deconv2(x))  # [B*T, 128, 16, 16]
        print(f"[DEBUG] After deconv2 (16x16): {x.shape}, min: {x.min()}, max: {x.max()}")

        # deconv3層を通過（16x16 -> 32x32）
        x = self.activation(self.deconv3(x))  # [B*T, 64, 32, 32]
        print(f"[DEBUG] After deconv3 (32x32): {x.shape}, min: {x.min()}, max: {x.max()}")

        # deconv4層を通過（32x32 -> 64x64）
        x = self.activation(self.deconv4(x))  # [B*T, 32, 64, 64]
        print(f"[DEBUG] After deconv4 (64x64): {x.shape}, min: {x.min()}, max: {x.max()}")

        # deconv5層を通過（64x64 -> 64x64）
        x = self.deconv5(x)  # [B*T, output_channels, 64, 64]
        print(f"[DEBUG] After deconv5 (final layer): {x.shape}, min: {x.min()}, max: {x.max()}")

        # 最終活性化関数を適用（Sigmoidで出力を0〜1にスケール）
        x = self.final_activation(x)
        print(f"[DEBUG] After final activation (Sigmoid): {x.shape}, min: {x.min()}, max: {x.max()}")

        # 最終的な出力の形状を [batch_size, timesteps, output_channels, 64, 64] に変換
        x = x.view(batch_size, timesteps, self.output_channels, 64, 64)  # [B, T, output_channels, 64, 64]
        print(f"[DEBUG] Final output shape: {x.shape}, min: {x.min()}, max: {x.max()}")
        return x
