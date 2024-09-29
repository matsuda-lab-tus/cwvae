# PyTorchを使って動画ファイルをデータセットとして扱い、モデルのトレーニングや評価に使用できる形式に変換するためのもの

import os
import torch
from torch.utils.data import Dataset, DataLoader
import imageio
import numpy as np

class VideoDataset(Dataset): # torch.utils.data.Dataset を継承したクラスで、ディスク上に保存されている動画ファイルをデータセットとして扱うためのクラスです。
    def __init__(self, video_dir, transform=None):
        # video_dir: 動画ファイルが保存されているディレクトリのパスを指定します。
        # self.video_files: 指定されたディレクトリ内の .mp4 ファイルをすべてリスト化して保持します。

        """
        動画ファイルを読み込み、PyTorchのデータセットとして扱うためのクラス。

        Args:
            video_dir (str): 動画ファイルが保存されているディレクトリのパス
            transform (callable, optional): データに適用するトランスフォーム
        """
        self.video_dir = video_dir
        self.video_files = [os.path.join(video_dir, f) for f in os.listdir(video_dir) if f.endswith('.mp4')]
        self.transform = transform # transform: 動画に適用するトランスフォーム（データ変換やデータ拡張などの操作）があれば、それを保持します。

    def __len__(self):
        return len(self.video_files) # データセットのサイズ（読み込む動画ファイルの数）を返します。これにより、データセットがどれだけのデータを持っているかがわかります。

    def __getitem__(self, idx):
        # 指定されたインデックス idx に基づいて動画ファイルを取得し、動画をフレームごとに読み込んでテンソルに変換します。
        video_path = self.video_files[idx]

        # 動画ファイルをフレームごとに読み込む
        video = self._load_video(video_path)

        if self.transform: # 必要に応じて transform が適用されます。
            video = self.transform(video)

        return video

    def _load_video(self, video_path):
        """動画ファイルをフレームごとに読み込み、テンソルに変換"""
        reader = imageio.get_reader(video_path, 'ffmpeg')
        frames = [frame for frame in reader]
        video = np.stack(frames, axis=0)  # (time, height, width, channels)
        video = torch.tensor(video, dtype=torch.float32)  # PyTorchのテンソルに変換
        # その後、torch.tensor でフレームをテンソルに変換し、0〜255のピクセル値を0〜1に正規化します。

        # 正規化 (0-255 を 0-1 に変換)
        video /= 255.0

        # (time, channels, height, width) に変更
        # 最後に、(時間, 高さ, 幅, チャンネル数)→(時間, チャンネル数, 高さ, 幅) の順に変更して返します。
        video = video.permute(0, 3, 1, 2)

        return video

def load_dataset(cfg):
    """
    動画データセットをロードし、DataLoaderを返す。
    Args:
        cfg (dict): 設定情報
    """
    # 訓練データと検証データを読み込む
    # トレーニングデータと検証データをそれぞれ VideoDataset クラスで読み込みます。ここでは、トレーニングデータと検証データが同じディレクトリから読み込まれていますが、これを分けたい場合は別のディレクトリを指定します。
    train_data_batch = VideoDataset(video_dir=cfg.datadir)
    val_data_batch = VideoDataset(video_dir=cfg.datadir)  # 検証データも同様に

    # DataLoaderでデータをバッチごとにロード
    train_loader = DataLoader(train_data_batch, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_data_batch, batch_size=cfg.batch_size, shuffle=False)

    return train_loader, val_loader
