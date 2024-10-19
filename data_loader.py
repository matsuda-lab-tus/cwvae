import os
import torch
from torch.utils.data import Dataset, DataLoader
import imageio
import numpy as np
from PIL import Image  # 追加
from torchvision import transforms


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
        video = self._load_video(video_path)  # [time, height, width, channels]

        # フレームごとに transform を適用
        frames = []
        for frame in video:
            # numpy.ndarray を PIL.Image に変換
            frame = Image.fromarray(frame)
            if self.transform:
                frame = self.transform(frame)
            else:
                # transform がない場合でもテンソルに変換
                frame = transforms.ToTensor()(frame)
            frames.append(frame)

        # フレームをスタックして [time, channels, height, width] のテンソルに
        video_tensor = torch.stack(frames)  # [time, channels, height, width]

        return video_tensor

    def _load_video(self, video_path):
        """
        動画ファイルを読み込み、numpy.ndarray に変換する。

        Args:
            video_path (str): 動画ファイルのパス

        Returns:
            numpy.ndarray: 動画データ (time, height, width, channels)
        """
        reader = imageio.get_reader(video_path, 'ffmpeg')
        frames = [frame for frame in reader]
        reader.close()
        
        # 動画を (time, height, width, channels) の形で取得
        video = np.stack(frames, axis=0)
        return video

# 画像データの正規化を含むトランスフォーム
transform = transforms.Compose([
    transforms.Resize((64, 64)),  # 画像を64x64にリサイズ
    transforms.ToTensor(),  # [0, 255]のピクセル値を[0, 1]にスケーリング
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 正規化を適用して[-1, 1]にスケーリング
])


def load_dataset(datadir, batch_size, transform=transform):
    """
    動画データセットをロードし、DataLoaderを返す。

    Args:
        datadir (str): データディレクトリのパス
        batch_size (int): バッチサイズ
        transform (callable, optional): データに適用するトランスフォーム

    Returns:
        train_loader, test_loader: トレーニングデータとテストデータのDataLoader
    """
    # トレーニングとテストのディレクトリを設定
    train_dir = os.path.join(datadir, 'train')  # トレーニング用ディレクトリ
    test_dir = os.path.join(datadir, 'test')    # テスト用ディレクトリ

    # データセットの作成
    train_dataset = VideoDataset(video_dir=train_dir, transform=transform)
    test_dataset = VideoDataset(video_dir=test_dir, transform=transform)

    # DataLoaderの作成
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    return train_loader, test_loader

