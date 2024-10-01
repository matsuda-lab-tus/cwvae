import os
import torch
from torch.utils.data import Dataset, DataLoader
import imageio
import numpy as np

class VideoDataset(Dataset):
    def __init__(self, video_dir, transform=None):
        """
        動画ファイルを読み込み、PyTorchのデータセットとして扱うためのクラス。

        Args:
            video_dir (str): 動画ファイルが保存されているディレクトリのパス
            transform (callable, optional): データに適用するトランスフォーム
        """
        self.video_dir = video_dir
        self.video_files = [os.path.join(video_dir, f) for f in os.listdir(video_dir) if f.endswith('.mp4')]
        self.transform = transform

    def __len__(self):
        return len(self.video_files)

    def __getitem__(self, idx):
        """
        指定されたインデックスの動画をフレームごとに読み込み、変換を適用して返す。

        Args:
            idx (int): データセット内のインデックス
        """
        video_path = self.video_files[idx]
        
        # 動画ファイルを読み込み
        video = self._load_video(video_path)

        # 必要ならばtransformを適用
        if self.transform:
            video = self.transform(video)

        return video

    def _load_video(self, video_path):
        """
        動画ファイルを読み込み、テンソルに変換する。

        Args:
            video_path (str): 動画ファイルのパス

        Returns:
            torch.Tensor: 正規化された動画データテンソル (time, channels, height, width)
        """
        reader = imageio.get_reader(video_path, 'ffmpeg')
        frames = [frame for frame in reader]
        
        # 動画を (time, height, width, channels) の形で取得
        video = np.stack(frames, axis=0)
        video = torch.tensor(video, dtype=torch.float32)

        # ピクセル値を 0-1 に正規化
        video /= 255.0

        # (time, channels, height, width) に次元を変更
        video = video.permute(0, 3, 1, 2)

        return video

def load_dataset(cfg, transform=None):
    """
    動画データセットをロードし、DataLoaderを返す。

    Args:
        cfg (dict): 設定情報
        transform (callable, optional): データに適用するトランスフォーム

    Returns:
        train_loader, val_loader: トレーニングデータと検証データのDataLoader
    """
    # トレーニングとテストのディレクトリを設定
    train_dir = os.path.join(cfg.datadir, 'train')  # トレーニング用ディレクトリ
    test_dir = os.path.join(cfg.datadir, 'test')    # テスト用ディレクトリ

    # データセットの作成
    train_dataset = VideoDataset(video_dir=train_dir, transform=transform)
    test_dataset = VideoDataset(video_dir=test_dir, transform=transform)

    # DataLoaderの作成
    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    return train_loader, test_loader
