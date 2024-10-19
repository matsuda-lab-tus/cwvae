import os
import torch
from torch.utils.data import Dataset, DataLoader
import imageio
import numpy as np
from PIL import Image
from torchvision import transforms


class VideoDataset(Dataset):
    def __init__(self, video_dir, seq_len=100, transform=None):
        self.video_dir = video_dir
        self.video_files = [os.path.join(video_dir, f) for f in os.listdir(video_dir) if f.endswith('.mp4')]
        self.transform = transform
        self.seq_len = seq_len  # 抽出するシーケンスの長さ

    def __len__(self):
        return len(self.video_files)

    def __getitem__(self, idx):
        video_path = self.video_files[idx]
        video = self._load_video(video_path)  # [time, height, width, channels]

        # 動画からフレームを取得し、必要なシーケンス長に合わせてトリミング
        video_tensor = self._process_video(video)  # [num_sequences, seq_len, channels, height, width]

        return video_tensor

    def _load_video(self, video_path):
        reader = imageio.get_reader(video_path, 'ffmpeg')
        frames = [frame for frame in reader]
        reader.close()

        # 動画を (time, height, width, channels) の形で取得
        video = np.stack(frames, axis=0)
        return video

    def _process_video(self, video):
        # 動画のフレーム数を取得
        total_frames = video.shape[0]
        
        # シーケンス数を計算
        num_sequences = total_frames // self.seq_len
        
        # トリミングして指定されたシーケンス長に合わせる
        trimmed_video = video[:num_sequences * self.seq_len]  # 必要なフレーム数にカット
        trimmed_video = trimmed_video.reshape(num_sequences, self.seq_len, *trimmed_video.shape[1:])  # [num_sequences, seq_len, height, width, channels]

        frames = []
        for seq in trimmed_video:
            for frame in seq:
                frame = Image.fromarray(frame)
                if self.transform:
                    frame = self.transform(frame)
                frames.append(frame)

        # フレームをスタックして [num_sequences, seq_len, channels, height, width] のテンソルに
        video_tensor = torch.stack(frames)  # [num_sequences * seq_len, channels, height, width]
        video_tensor = video_tensor.view(num_sequences, self.seq_len, *video_tensor.shape[1:])  # [num_sequences, seq_len, channels, height, width]

        # 最終的な形状: [num_sequences, seq_len, channels, height, width]
        return video_tensor


# 画像データの正規化を含むトランスフォーム
transform = transforms.Compose([
    transforms.Resize((64, 64)),  # 画像を64x64にリサイズ
    transforms.ToTensor(),  # [0, 255]のピクセル値を[0, 1]にスケーリング
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 正規化を適用して[-1, 1]にスケーリング
])


def load_dataset(datadir, batch_size, seq_len=100, transform=transform):
    train_dir = os.path.join(datadir, 'train')
    test_dir = os.path.join(datadir, 'test')

    train_dataset = VideoDataset(video_dir=train_dir, seq_len=seq_len, transform=transform)
    test_dataset = VideoDataset(video_dir=test_dir, seq_len=seq_len, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    return train_loader, test_loader


def test_data_loader(datadir, batch_size):
    train_loader, test_loader = load_dataset(datadir, batch_size)

    # トレーニングデータの確認
    print("Testing train_loader...")
    for batch_idx, data in enumerate(train_loader):
        print(f"Batch {batch_idx + 1}:")
        print(f"Data shape: {data.shape}")  # データの形状を確認

        if batch_idx == 2:  # 3バッチ分だけ確認
            break

    # テストデータの確認
    print("\nTesting test_loader...")
    for batch_idx, data in enumerate(test_loader):
        print(f"Batch {batch_idx + 1}:")
        print(f"Data shape: {data.shape}")  # データの形状を確認

        if batch_idx == 2:  # 3バッチ分だけ確認
            break


if __name__ == "__main__":
    datadir = "./minerl_navigate/"  # データが格納されているディレクトリのパス
    batch_size = 16  # バッチサイズを指定
    test_data_loader(datadir, batch_size)
