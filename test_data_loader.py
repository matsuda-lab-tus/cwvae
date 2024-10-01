import os
import torch
from torch.utils.data import DataLoader
import unittest
import tempfile
import shutil
import imageio
import numpy as np
from data_loader import VideoDataset, load_dataset  # 必要に応じてインポート

class TestVideoDataset(unittest.TestCase):

    def setUp(self):
        # テスト用の一時ディレクトリと動画ファイルを作成
        self.test_dir = tempfile.mkdtemp()
        self.create_dummy_video_file(self.test_dir, 'test_video.mp4')

        # テスト用の設定
        self.cfg = {
            'datadir': self.test_dir,
            'batch_size': 2
        }

    def tearDown(self):
        # テストが終了したら一時ディレクトリを削除
        shutil.rmtree(self.test_dir)

    def create_dummy_video_file(self, dir, filename):
        """ダミーの動画ファイルを作成する"""
        filepath = os.path.join(dir, filename)
        writer = imageio.get_writer(filepath, fps=24)

        # 10フレームのダミー動画を作成 (64x64の黒いフレーム)
        for _ in range(10):
            frame = np.zeros((64, 64, 3), dtype=np.uint8)
            writer.append_data(frame)
        writer.close()

    def test_video_dataset_loading(self):
        """VideoDatasetが正しく動作するかテスト"""
        dataset = VideoDataset(video_dir=self.cfg['datadir'])

        # データセットの長さが1（動画ファイルが1つだけ）か確認
        self.assertEqual(len(dataset), 1)

        # データセットの最初の動画が正しく読み込まれるか確認
        video = dataset[0]
        self.assertEqual(video.shape, (10, 3, 64, 64))  # 10フレーム、(3, 64, 64)の形状

    def test_load_dataset(self):
        """DataLoaderが正しく動作するかテスト"""
        train_loader, val_loader = load_dataset(self.cfg)

        # DataLoaderからデータが正しくロードされるか確認
        for batch in train_loader:
            self.assertEqual(batch.shape, (1, 10, 3, 64, 64))  # バッチサイズ1, 10フレーム, 3チャンネル, 64x64

        for batch in val_loader:
            self.assertEqual(batch.shape, (1, 10, 3, 64, 64))  # 検証データも同様に確認

if __name__ == '__main__':
    unittest.main()
