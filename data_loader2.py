import os
import torch
from torch.utils.data import Dataset, DataLoader
import imageio
import numpy as np
from torchvision import transforms

class VideoDataset2(Dataset):
    def __init__(self, video_dir, target_resolution=(64, 64), target_frames=100, transform=None):
        """
        動画ファイルを読み込み、PyTorchのデータセットとして扱うためのクラス。

        Args:
            video_dir (str): 動画ファイルが保存されているディレクトリのパス
            target_resolution (tuple): 動画フレームをリサイズする目標の解像度 (height, width)
            target_frames (int): 統一するフレーム数
            transform (callable, optional): データに適用するトランスフォーム
        """
        self.video_dir = video_dir
        self.video_files = [os.path.join(video_dir, f) for f in os.listdir(video_dir) if f.endswith('.mp4')]
        self.target_resolution = target_resolution
        self.target_frames = target_frames
        self.transform = transform

        # デフォルトのトランスフォーム (リサイズ)
        self.resize_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(self.target_resolution),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.video_files)

    def __getitem__(self, idx):
        """
        指定されたインデックスの動画をフレームごとに読み込み、変換を適用して返す。

        Args:
            idx (int): データセット内のインデックス
        """
        video_path = self.video_files[idx]
        
        try:
            # 動画ファイルを読み込み
            video = self._load_video(video_path)

            # 必要ならばtransformを適用
            if self.transform:
                video = self.transform(video)

            return video
        except Exception as e:
            print(f"Error loading video at {video_path}: {e}")
            return None  # エラーが発生した場合にはNoneを返す

    def _load_video(self, video_path):
        """
        動画ファイルを読み込み、フレーム数と解像度を統一したテンソルに変換する。

        Args:
            video_path (str): 動画ファイルのパス

        Returns:
            torch.Tensor: 正規化された動画データテンソル (time, channels, height, width)
        """
        try:
            reader = imageio.get_reader(video_path, 'ffmpeg')
            frames = [frame for frame in reader]

            # 動画のフレーム数を統一
            video = self._adjust_frame_count(frames)

            # フレームごとにリサイズしてテンソルに変換
            resized_video = []
            for frame in video:
                frame_tensor = self.resize_transform(frame)  # リサイズ & Tensor化
                resized_video.append(frame_tensor)

            # 動画を (time, channels, height, width) の形で取得
            video_tensor = torch.stack(resized_video)

            return video_tensor
        except Exception as e:
            print(f"Failed to load video {video_path}: {e}")
            raise e

    def _adjust_frame_count(self, frames):
        """
        フレーム数を `target_frames` に合わせるために、フレーム数が多ければサンプリングし、
        少なければパディングする。

        Args:
            frames (list): 動画フレームのリスト

        Returns:
            list: 統一されたフレーム数の動画フレームリスト
        """
        num_frames = len(frames)

        if num_frames > self.target_frames:
            # フレーム数が多い場合、サンプリングして減らす
            indices = np.linspace(0, num_frames - 1, self.target_frames).astype(int)
            frames = [frames[i] for i in indices]
        elif num_frames < self.target_frames:
            # フレーム数が少ない場合、最後のフレームを繰り返してパディング
            last_frame = frames[-1]
            while len(frames) < self.target_frames:
                frames.append(last_frame)

        return frames


def load_dataset(cfg, transform=None):
    """
    動画データセットをロードし、DataLoaderを返す。

    Args:
        cfg (dict): 設定情報
        transform (callable, optional): データに適用するトランスフォーム

    Returns:
        train_loader, test_loader: トレーニングデータと検証データのDataLoader
    """
    # ディレクトリとバッチサイズを設定（デフォルト値も設定可能）
    datadir = cfg.get('datadir', './data')  # デフォルトで './data' に設定
    batch_size = cfg.get('batch_size', 16)  # デフォルトでバッチサイズ 16

    # トレーニングと検証のディレクトリを設定
    train_dir = os.path.join(datadir, 'train', 'intended')  # トレーニング用の intended ディレクトリ
    test_dir = os.path.join(datadir, 'test', 'intended')  # 検証用の intended ディレクトリ

    # データセットの作成
    train_dataset = VideoDataset2(video_dir=train_dir, target_resolution=(128, 128), target_frames=100, transform=transform)
    test_dataset = VideoDataset2(video_dir=test_dir, target_resolution=(128, 128), target_frames=100, transform=transform)

    # DataLoaderの作成
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    return train_loader, test_loader
