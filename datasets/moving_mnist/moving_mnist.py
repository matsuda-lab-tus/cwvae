"""moving_mnist dataset."""

import os
import zipfile
from pathlib import Path
import imageio
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

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

_DOWNLOAD_URL = "https://archive.org/download/moving_mnist/moving_mnist_2digit.zip"


class MovingMnist2Digit(Dataset):
    """Dataset class for Moving MNIST using PyTorch."""

    def __init__(self, root_dir, split="train", transform=None):
        """
        Args:
            root_dir (string): Directory with the extracted dataset.
            split (string): "train" or "test" to specify the dataset split.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = Path(root_dir) / split
        self.video_files = list(self.root_dir.glob("*.mp4"))
        self.transform = transform

    def __len__(self):
        return len(self.video_files)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        video_path = self.video_files[idx]
        video = self._load_video(video_path)

        sample = {"video": video}

        if self.transform:
            sample = self.transform(sample)

        return sample

    def _load_video(self, video_path):
        """Loads video from file using imageio."""
        reader = imageio.get_reader(video_path)
        frames = [frame for frame in reader]
        video = torch.from_numpy(np.stack(frames, axis=0)).unsqueeze(1)  # Shape: (time, 1, height, width)
        return video


def download_and_extract(url, download_path):
    """Downloads and extracts dataset."""
    # Downloading the zip file
    zip_path = download_path / "moving_mnist_2digit.zip"
    os.system(f"wget {url} -O {zip_path}")

    # Extracting the zip file
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(download_path)

    print(f"Dataset downloaded and extracted to {download_path}")


# Usage example:
if __name__ == "__main__":
    # Define dataset directory and download it if necessary
    data_dir = Path("./moving_mnist_dataset")
    if not data_dir.exists():
        download_and_extract(_DOWNLOAD_URL, data_dir)

    # Load the training dataset
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = MovingMnist2Digit(root_dir=data_dir, split="train", transform=transform)

    # Create a DataLoader for the dataset
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True)

    # Iterate over the DataLoader
    for batch in train_loader:
        videos = batch["video"]
        print(videos.shape)  # Should be (batch_size, time, 1, height, width)
