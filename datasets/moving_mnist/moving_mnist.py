"""moving_mnist dataset."""  # このファイルはMoving MNISTデータセットに関するコードです。

import os  # OSライブラリをインポートして、ファイルやディレクトリを操作します
import zipfile  # ZIPファイルを扱うためのライブラリをインポートします
from pathlib import Path  # パス操作のためのライブラリをインポートします
import imageio  # 動画を読み込むためのライブラリをインポートします
import torch  # PyTorchライブラリをインポートします
from torch.utils.data import Dataset  # PyTorchのDatasetクラスをインポートします
import torchvision.transforms as transforms  # 画像の変換を行うためのライブラリをインポートします

# データセットに関する説明文
_DESCRIPTION = """
# Moving MNIST Dataset

References:
@article{saxena2021clockworkvae,
  title={Clockwork Variational Autoencoders}, 
  author={Saxena, Vaibhav and Ba, Jimmy and Hafner, Danijar},
  journal={arXiv preprint arXiv:2102.09532},
  year={2021},
}
"""

# 引用のフォーマット
_CITATION = """
@article{saxena2021clockwork,
      title={Clockwork Variational Autoencoders}, 
      author={Vaibhav Saxena and Jimmy Ba and Danijar Hafner},
      year={2021},
      eprint={2102.09532},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
"""

# データセットをダウンロードするURL
_DOWNLOAD_URL = "https://archive.org/download/moving_mnist/moving_mnist_2digit.zip"

# Moving MNISTデータセットクラスの定義
class MovingMnist2Digit(Dataset):
    """PyTorchを使ったMoving MNISTデータセットのクラスです。"""

    def __init__(self, root_dir, split="train", transform=None):
        """
        コンストラクタ（初期化）メソッド
        Args:
            root_dir (string): 抽出されたデータセットのディレクトリ。
            split (string): "train"（訓練データ）または "test"（テストデータ）を指定。
            transform (callable, optional): サンプルに適用する変換。
        """
        self.root_dir = Path(root_dir) / split  # データセットのパスを設定
        self.video_files = list(self.root_dir.glob("*.mp4"))  # ディレクトリ内の全MP4ファイルをリストに格納
        self.transform = transform  # 画像変換処理を設定

    def __len__(self):
        return len(self.video_files)  # データセット内の動画ファイルの数を返す

    def __getitem__(self, idx):
        """指定されたインデックスの動画を取得します。"""
        if torch.is_tensor(idx):
            idx = idx.tolist()  # インデックスがテンソルの場合はリストに変換

        video_path = self.video_files[idx]  # インデックスに基づいて動画のパスを取得
        video = self._load_video(video_path)  # 動画を読み込む

        sample = {"video": video}  # サンプルを辞書形式で作成

        if self.transform:  # 変換が指定されている場合
            sample = self.transform(sample)  # サンプルに変換を適用

        return sample  # サンプルを返す

    def _load_video(self, video_path):
        """imageioを使ってファイルから動画を読み込みます。"""
        reader = imageio.get_reader(video_path)  # 動画リーダーを作成
        frames = [frame for frame in reader]  # 各フレームを読み込みます
        video = torch.from_numpy(np.stack(frames, axis=0)).unsqueeze(1)  # 動画をテンソルに変換（形状: (time, 1, height, width)）
        return video  # 動画データを返します

# データセットをダウンロードして解凍する関数
def download_and_extract(url, download_path):
    """データセットをダウンロードして解凍します。"""
    # ZIPファイルをダウンロード
    zip_path = download_path / "moving_mnist_2digit.zip"  # ZIPファイルのパスを設定
    os.system(f"wget {url} -O {zip_path}")  # wgetを使ってデータをダウンロード

    # ZIPファイルを解凍
    with zipfile.ZipFile(zip_path, "r") as zip_ref:  # ZIPファイルを開く
        zip_ref.extractall(download_path)  # 解凍する

    print(f"Dataset downloaded and extracted to {download_path}")  # 完了メッセージを表示


# 使用例
if __name__ == "__main__":
    # データセットのディレクトリを定義し、必要に応じてダウンロードします
    data_dir = Path("./moving_mnist_dataset")  # データ保存先のパスを設定
    if not data_dir.exists():  # ディレクトリが存在しない場合
        download_and_extract(_DOWNLOAD_URL, data_dir)  # データをダウンロードして解凍

    # 訓練データセットを読み込む
    transform = transforms.Compose([transforms.ToTensor()])  # 変換処理を定義
    train_dataset = MovingMnist2Digit(root_dir=data_dir, split="train", transform=transform)  # データセットを作成

    # データローダーを作成
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True)  # データローダーを作成

    # データローダーを反復処理する
    for batch in train_loader:
        videos = batch["video"]  # バッチから動画を取得
        print(videos.shape)  # 動画の形状を表示（形状: (バッチサイズ, 時間, 1, 高さ, 幅)）
