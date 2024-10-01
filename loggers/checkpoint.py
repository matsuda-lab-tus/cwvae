import os
import torch

class Checkpoint:
    def __init__(self, log_dir_root):
        self._log_dir_root = log_dir_root
        self.log_dir_model = os.path.join(self._log_dir_root, "model")
        self._ckpt_name = "model.pth"

    def save(self, model, optimizer, epoch, save_dir=None):
        """
        モデルとオプティマイザの状態を保存します。
        """
        if save_dir is None:
            os.makedirs(self.log_dir_model, exist_ok=True)
            save_path = os.path.join(self.log_dir_model, self._ckpt_name)
        else:
            save_path = os.path.join(self._log_dir_root, save_dir)
            os.makedirs(save_path, exist_ok=True)
            save_path = os.path.join(save_path, self._ckpt_name)

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, save_path)
        print(f"Checkpoint saved at {save_path}")

    def restore(self, model, optimizer=None, restore_dir=None):
        """
        モデルとオプティマイザの状態を復元します。
        """
        if restore_dir is None:
            restore_path = os.path.join(self.log_dir_model, self._ckpt_name)
        else:
            restore_path = os.path.join(self._log_dir_root, restore_dir, self._ckpt_name)

        if not os.path.exists(restore_path):
            raise FileNotFoundError(f"Checkpoint not found at {restore_path}")

        checkpoint = torch.load(restore_path)
        model.load_state_dict(checkpoint['model_state_dict'])

        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        epoch = checkpoint['epoch']
        print(f"Checkpoint restored from {restore_path}, starting from epoch {epoch}")
        return epoch
    
    def get_last_step(self):
        # 復元した際に取得したステップを返す
        return getattr(self, 'start_step', 0)