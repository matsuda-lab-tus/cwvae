import os
import torch
from torch.utils.data import Dataset, DataLoader
import imageio
import numpy as np
from PIL import Image
from torchvision import transforms


class VideoDataset(Dataset):
    def __init__(self, video_dir, transform=None):
        self.video_dir = video_dir
        self.video_files = [os.path.join(video_dir, f) for f in os.listdir(video_dir) if f.endswith('.mp4')]
        self.transform = transform

    def __len__(self):
        return len(self.video_files)

    def __getitem__(self, idx):
        video_path = self.video_files[idx]
        video = self._load_video(video_path)  # [time, height, width, channels]

        frames = []
        for frame in video:
            frame = Image.fromarray(frame)
            if self.transform:
                frame = self.transform(frame)
            frames.append(frame)

        # フレームをスタックして [time, channels, height, width] のテンソルに
        video_tensor = torch.stack(frames)  # [time, channels, height, width]

        return video_tensor

    def _load_video(self, video_path):
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
    train_dir = os.path.join(datadir, 'train')
    test_dir = os.path.join(datadir, 'test')

    train_dataset = VideoDataset(video_dir=train_dir, transform=transform)
    test_dataset = VideoDataset(video_dir=test_dir, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    return train_loader, test_loader
