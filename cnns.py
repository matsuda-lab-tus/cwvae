import torch  # PyTorchライブラリを使います
import torch.nn as nn  # ニューラルネットワークの基本機能を使います
import torch.nn.functional as F  # ニューラルネットワークのための便利な関数を使います
import math  # 数学的な計算をするためのライブラリ


# エンコーダーのクラスを定義します
class Encoder(nn.Module):
    def __init__(self, levels, tmp_abs_factor, dense_layers=3, embed_size=1024, channels_mult=1):
        super(Encoder, self).__init__()  # 親クラスを初期化
        self.levels = levels  # 階層の数を設定
        self.tmp_abs_factor = tmp_abs_factor  # 時間の絶対的な因子を設定
        self.activation = nn.LeakyReLU(negative_slope=0.01)  # 活性化関数をLeakyReLUに設定
        self.channels_mult = channels_mult  # チャンネルの倍率を設定
        self.dense_layers_num = dense_layers  # 密な層の数を設定
        self.embed_size = embed_size  # 埋め込みサイズを設定

        # 畳み込み層を定義します
        filters = 32  # フィルター数を設定
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=channels_mult * filters, kernel_size=4, stride=2, padding=1)  # 畳み込み層1
        self.conv2 = nn.Conv2d(in_channels=channels_mult * filters, out_channels=channels_mult * filters * 2, kernel_size=4, stride=2, padding=1)  # 畳み込み層2
        self.conv3 = nn.Conv2d(in_channels=channels_mult * filters * 2, out_channels=channels_mult * filters * 4, kernel_size=4, stride=2, padding=1)  # 畳み込み層3
        self.conv4 = nn.Conv2d(in_channels=channels_mult * filters * 4, out_channels=channels_mult * filters * 8, kernel_size=4, stride=2, padding=1)  # 畳み込み層4

        # 畳み込み後の出力サイズを計算
        self.conv_output_size = channels_mult * filters * 8 * 4 * 4  # 出力サイズを調整

        # 各階層の全結合層を定義
        self.level_dense_layers = nn.ModuleList()  # モジュールリストを作成
        for level in range(1, self.levels):  # 各階層に対して
            level_layers = nn.ModuleList()  # 各階層の層を格納するリスト
            in_features = self.conv_output_size  # 入力サイズを設定
            for _ in range(self.dense_layers_num - 1):  # 密な層の数だけループ
                dense_layer = nn.Linear(in_features, self.embed_size)  # 全結合層を作成
                level_layers.append(dense_layer)  # 層を追加
                in_features = self.embed_size  # 次の層の入力サイズを更新
            dense_layer = nn.Linear(in_features, self.conv_output_size)  # 最後の全結合層を作成
            level_layers.append(dense_layer)  # 最後の層を追加
            self.level_dense_layers.append(level_layers)  # 各階層の層を保存

    def forward(self, obs):
        # 入力のサイズを取得
        batch_size, seq_len, channels, height, width = obs.size()
        # 入力を適切な形に変形
        obs = obs.reshape(batch_size * seq_len, channels, height, width)

        # 畳み込み層を通して処理します
        hidden = self.activation(self.conv1(obs))  # 畳み込み層1を通す
        hidden = self.activation(self.conv2(hidden))  # 畳み込み層2を通す
        hidden = self.activation(self.conv3(hidden))  # 畳み込み層3を通す
        hidden = self.activation(self.conv4(hidden))  # 畳み込み層4を通す

        hidden = hidden.view(batch_size, seq_len, -1)  # 畳み込み出力をフラットにする
        layer = hidden  # 現在の層を設定

        layers = []  # 各階層の出力を保存するリスト
        layers.append(layer)  # 最初の層を追加
        print(f"[DEBUG] Input shape at level 0: {layer.shape}")  # 入力の形を表示

        # 各階層ごとに全結合層を通します
        for level in range(1, self.levels):
            dense_layers = self.level_dense_layers[level - 1]  # 現在の階層の密な層を取得
            hidden = layer  # 前の層の出力を設定
            for dense_layer in dense_layers[:-1]:  # 最後の層以外に対してループ
                hidden = F.relu(dense_layer(hidden))  # 活性化関数を通す
            hidden = dense_layers[-1](hidden)  # 最後の層を通す

            layer = hidden  # 現在の層を更新
            feat_size = layer.size(-1)  # 特徴サイズを取得

            # 時間ステップを統合します
            timesteps_to_merge = self.tmp_abs_factor  # 統合する時間ステップ数
            current_timesteps = layer.size(1)  # 現在の時間ステップ数
            # 不足する時間ステップを計算
            timesteps_to_pad = (timesteps_to_merge - (current_timesteps % timesteps_to_merge)) % timesteps_to_merge
            if timesteps_to_pad > 0:  # パディングが必要な場合
                padding = torch.zeros(batch_size, timesteps_to_pad, feat_size).to(layer.device)  # パディング用のゼロを作成
                layer = torch.cat([layer, padding], dim=1)  # パディングを追加
                print(f"[DEBUG] Padded {timesteps_to_pad} timesteps at level {level}")  # パディングの情報を表示

            # 時間ステップを統合
            merged_timesteps = math.ceil(layer.size(1) / timesteps_to_merge)  # 統合された時間ステップ数
            layer = layer.view(batch_size, merged_timesteps, timesteps_to_merge, feat_size)  # 形を変形
            layer = layer.sum(dim=2)  # 時間ステップを統合
            layers.append(layer)  # 現在の層を追加
            print(f"[DEBUG] Input shape at level {level}: {layer.shape}")  # 現在の層の形を表示

        return layers  # 各階層の出力を返す


# デコーダーのクラスを定義します
class Decoder(nn.Module):
    def __init__(self, output_channels, embed_size, channels_mult=1, final_activation=None):
        super(Decoder, self).__init__()  # 親クラスを初期化
        self.embed_size = embed_size  # 埋め込みサイズを設定
        self.output_channels = output_channels  # 出力チャンネル数を設定
        self.channels_mult = channels_mult  # チャンネルの倍率を設定

        # 全結合層を作成し、1024次元に変換
        self.fc = nn.Linear(self.embed_size, 1024)  # 埋め込みサイズから1024への全結合層

        # フィルター数を設定
        filters = 32

        # ConvTranspose2d の in_channels と out_channels を修正
        self.deconv1 = nn.ConvTranspose2d(  # 畳み込み転置層1
            in_channels=1024,
            out_channels=self.channels_mult * filters * 4,  # 128
            kernel_size=5,
            stride=2,
            padding=1
        )
        self.deconv2 = nn.ConvTranspose2d(  # 畳み込み転置層2
            in_channels=self.channels_mult * filters * 4,  # 128
            out_channels=self.channels_mult * filters * 2,  # 64
            kernel_size=5,
            stride=2,
            padding=1
        )
        self.deconv3 = nn.ConvTranspose2d(  # 畳み込み転置層3
            in_channels=self.channels_mult * filters * 2,  # 64
            out_channels=self.channels_mult * filters,      # 32
            kernel_size=6,
            stride=2,
            padding=1
        )
        self.deconv4 = nn.ConvTranspose2d(  # 畳み込み転置層4
            in_channels=self.channels_mult * filters,       # 32
            out_channels=self.channels_mult * filters // 2, # 16
            kernel_size=6,
            stride=2,
            padding=2
        )
        self.deconv5 = nn.ConvTranspose2d(  # 畳み込み転置層5
            in_channels=self.channels_mult * filters // 2,  # 16
            out_channels=output_channels,                   # 出力チャネル数
            kernel_size=6,
            stride=2,
            padding=2
        )

        self.activation = nn.LeakyReLU(negative_slope=0.01)  # 活性化関数をLeakyReLUに設定

        # 最終的な活性化関数を設定
        if final_activation is not None:
            self.final_activation = final_activation
        else:
            self.final_activation = nn.Tanh()  # デフォルトはTanh

    def forward(self, x):
        intermediate_outputs = {}  # 中間出力を保存する辞書

        batch_size, timesteps, embed_size = x.size()  # 入力のサイズを取得
        print(f"[DEBUG] Input to Decoder: {x.shape}, min: {x.min()}, max: {x.max()}")  # 入力の形を表示

        # 全結合層に通すために形を変更
        x = x.reshape(batch_size * timesteps, embed_size)  # 形を変更
        print(f"[DEBUG] After reshaping for fc layer: {x.shape}, min: {x.min()}, max: {x.max()}")  # 形の変化を表示

        # 全結合層を通します
        x = self.activation(self.fc(x))  # 活性化関数を通す
        intermediate_outputs['fc'] = x  # fc層の出力を保存
        print(f"[DEBUG] After fc: {x.shape}, min: {x.min()}, max: {x.max()}")  # 形の変化を表示

        # 1x1のグリッドに変形します
        x = x.view(batch_size * timesteps, 1024, 1, 1)  # 形を変更
        intermediate_outputs['reshaped'] = x  # リシェイプ後の出力を保存
        print(f"[DEBUG] After reshaping to 1x1: {x.shape}, min: {x.min()}, max: {x.max()}")  # 形の変化を表示

        # 畳み込み転置層を通してアップサンプリングします
        x = self.activation(self.deconv1(x))  # 畳み込み転置層1を通す
        intermediate_outputs['deconv1'] = x  # deconv1層の出力を保存
        print(f"[DEBUG] After deconv1: {x.shape}, min: {x.min()}, max: {x.max()}")  # 形の変化を表示

        x = self.activation(self.deconv2(x))  # 畳み込み転置層2を通す
        intermediate_outputs['deconv2'] = x  # deconv2層の出力を保存
        print(f"[DEBUG] After deconv2: {x.shape}, min: {x.min()}, max: {x.max()}")  # 形の変化を表示

        x = self.activation(self.deconv3(x))  # 畳み込み転置層3を通す
        intermediate_outputs['deconv3'] = x  # deconv3層の出力を保存
        print(f"[DEBUG] After deconv3: {x.shape}, min: {x.min()}, max: {x.max()}")  # 形の変化を表示

        x = self.activation(self.deconv4(x))  # 畳み込み転置層4を通す
        intermediate_outputs['deconv4'] = x  # deconv4層の出力を保存
        print(f"[DEBUG] After deconv4: {x.shape}, min: {x.min()}, max: {x.max()}")  # 形の変化を表示

        x = self.deconv5(x)  # 最後の畳み込み転置層を通す
        intermediate_outputs['deconv5'] = x  # deconv5層の出力を保存
        print(f"[DEBUG] After deconv5 (final layer): {x.shape}, min: {x.min()}, max: {x.max()}")  # 形の変化を表示

        x = self.final_activation(x)  # 最終的な活性化関数を通す
        intermediate_outputs['final_activation'] = x  # 最終活性化関数後の出力を保存
        print(f"[DEBUG] After final activation ({self.final_activation.__class__.__name__}): {x.shape}, min: {x.min()}, max: {x.max()}")  # 形の変化を表示

        # 元の次元に戻します
        x = x.view(batch_size, timesteps, self.output_channels, 64, 64)  # 形を戻す
        print(f"[DEBUG] Final output shape: {x.shape}, min: {x.min()}, max: {x.max()}")  # 最終出力の形を表示

        return x, intermediate_outputs  # 最終出力と中間出力を返す
