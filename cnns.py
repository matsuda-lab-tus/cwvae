import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class Encoder(nn.Module):
    def __init__(self, levels, tmp_abs_factor, dense_layers=3, embed_size=100, channels_mult=1):
        super(Encoder, self).__init__()
        self.levels = levels
        self.tmp_abs_factor = tmp_abs_factor
        self.activation = nn.LeakyReLU(negative_slope=0.01)
        self.channels_mult = channels_mult
        self.dense_layers_num = dense_layers
        self.embed_size = embed_size

        # Define convolutional layers
        filters = 32  
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=channels_mult * filters, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(in_channels=channels_mult * filters, out_channels=channels_mult * filters * 2, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(in_channels=channels_mult * filters * 2, out_channels=channels_mult * filters * 4, kernel_size=4, stride=2, padding=1)
        self.conv4 = nn.Conv2d(in_channels=channels_mult * filters * 4, out_channels=channels_mult * filters * 8, kernel_size=4, stride=2, padding=1)

        # Output size after convolutions (input size = 64x64)
        self.conv_output_size = channels_mult * filters * 8 * 4 * 4  # Adjust output feature size

        # Fully connected layers for each level
        self.level_dense_layers = nn.ModuleList()
        for level in range(1, self.levels):
            level_layers = nn.ModuleList()
            in_features = self.conv_output_size
            for _ in range(self.dense_layers_num - 1):
                dense_layer = nn.Linear(in_features, self.embed_size)
                level_layers.append(dense_layer)
                in_features = self.embed_size  # Next layer's input is current layer's output size
            dense_layer = nn.Linear(in_features, self.conv_output_size)
            level_layers.append(dense_layer)
            self.level_dense_layers.append(level_layers)

    def forward(self, obs):
        batch_size, timesteps, channels, height, width = obs.size()
        obs = obs.reshape(batch_size * timesteps, channels, height, width)

        # Convolutional layers processing
        hidden = self.activation(self.conv1(obs))
        hidden = self.activation(self.conv2(hidden))
        hidden = self.activation(self.conv3(hidden))
        hidden = self.activation(self.conv4(hidden))

        hidden = hidden.view(batch_size, timesteps, -1)  # Flatten convolutional output
        layer = hidden

        layers = []
        layers.append(layer)
        print(f"[DEBUG] Input shape at level 0: {layer.shape}")

        for level in range(1, self.levels):
            dense_layers = self.level_dense_layers[level - 1]
            hidden = layer
            for dense_layer in dense_layers[:-1]:
                hidden = F.relu(dense_layer(hidden))
            hidden = dense_layers[-1](hidden)

            layer = hidden
            feat_size = layer.size(-1)

            # Merge time steps
            timesteps_to_merge = self.tmp_abs_factor
            current_timesteps = layer.size(1)
            timesteps_to_pad = (timesteps_to_merge - (current_timesteps % timesteps_to_merge)) % timesteps_to_merge
            if timesteps_to_pad > 0:
                padding = torch.zeros(batch_size, timesteps_to_pad, feat_size).to(layer.device)
                layer = torch.cat([layer, padding], dim=1)
                print(f"[DEBUG] Padded {timesteps_to_pad} timesteps at level {level}")

            merged_timesteps = math.ceil(layer.size(1) / timesteps_to_merge)
            layer = layer.view(batch_size, merged_timesteps, timesteps_to_merge, feat_size)
            layer = layer.sum(dim=2)
            layers.append(layer)
            print(f"[DEBUG] Input shape at level {level}: {layer.shape}")

        return layers


class Decoder(nn.Module):
    def __init__(self, output_channels, embed_size, channels_mult=1, final_activation=None):
        super(Decoder, self).__init__()
        self.embed_size = embed_size
        self.output_channels = output_channels
        self.channels_mult = channels_mult

        # 全結合層で1024次元に変換
        self.fc = nn.Linear(self.embed_size, channels_mult * 1024)

        # 逆畳み込み層（TensorFlow版に合わせてフィルター数を調整）
        filters = 32
        self.deconv1 = nn.ConvTranspose2d(channels_mult * filters * 4, channels_mult * filters * 4, kernel_size=5, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(channels_mult * filters * 4, channels_mult * filters * 2, kernel_size=5, stride=2, padding=1)
        self.deconv3 = nn.ConvTranspose2d(channels_mult * filters * 2, channels_mult * filters, kernel_size=6, stride=2, padding=1)
        self.deconv4 = nn.ConvTranspose2d(channels_mult * filters, channels_mult * filters // 2, kernel_size=6, stride=2, padding=1)
        self.deconv5 = nn.ConvTranspose2d(channels_mult * filters // 2, output_channels, kernel_size=6, stride=2, padding=1)

        self.activation = nn.LeakyReLU(negative_slope=0.01)

        if final_activation is not None:
            self.final_activation = final_activation
        else:
            self.final_activation = nn.Tanh()

    def forward(self, x):
        batch_size, timesteps, embed_size = x.size()
        print(f"[DEBUG] Input to Decoder: {x.shape}, min: {x.min()}, max: {x.max()}")

        # Reshape for fc layer
        x = x.view(batch_size * timesteps, embed_size)
        print(f"[DEBUG] After reshaping for fc layer: {x.shape}, min: {x.min()}, max: {x.max()}")

        # Pass through fully connected layer
        x = self.activation(self.fc(x))
        print(f"[DEBUG] After fc: {x.shape}, min: {x.min()}, max: {x.max()}")

        # Reshape to 4x4 grid for deconvolution
        x = x.view(batch_size * timesteps, 512, 4, 4)
        print(f"[DEBUG] After reshaping to 4x4: {x.shape}, min: {x.min()}, max: {x.max()}")

        # Upsample through deconvolution layers
        x = self.activation(self.deconv1(x))
        print(f"[DEBUG] After deconv1 (8x8): {x.shape}, min: {x.min()}, max: {x.max()}")

        x = self.activation(self.deconv2(x))
        print(f"[DEBUG] After deconv2 (16x16): {x.shape}, min: {x.min()}, max: {x.max()}")

        x = self.activation(self.deconv3(x))
        print(f"[DEBUG] After deconv3 (32x32): {x.shape}, min: {x.min()}, max: {x.max()}")

        x = self.activation(self.deconv4(x))
        print(f"[DEBUG] After deconv4 (64x64): {x.shape}, min: {x.min()}, max: {x.max()}")

        x = self.deconv5(x)
        print(f"[DEBUG] After deconv5 (final layer): {x.shape}, min: {x.min()}, max: {x.max()}")

        x = self.final_activation(x)
        print(f"[DEBUG] After final activation (Tanh): {x.shape}, min: {x.min()}, max: {x.max()}")

        x = x.view(batch_size, timesteps, self.output_channels, 64, 64)
        print(f"[DEBUG] Final output shape: {x.shape}, min: {x.min()}, max: {x.max()}")
        return x
    
