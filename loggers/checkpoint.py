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

        # DataParallelの場合、model.module.state_dict()を保存
        if isinstance(model, torch.nn.DataParallel):
            state_dict = model.module.state_dict()
        else:
            state_dict = model.state_dict()

        torch.save({
            'epoch': epoch,
            'model_state_dict': state_dict,
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

        checkpoint = torch.load(restore_path, map_location=model.device if hasattr(model, 'device') else 'cpu')
        
        # DataParallelの場合、model.moduleにロード
        if isinstance(model, torch.nn.DataParallel):
            model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint['model_state_dict'])

        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        epoch = checkpoint['epoch']
        print(f"Checkpoint restored from {restore_path}, starting from epoch {epoch}")
        return epoch
    
    def exists(self):
        """
        チェックポイントが存在するかを確認します。
        """
        return os.path.exists(os.path.join(self.log_dir_model, self._ckpt_name))
    
    @property
    def latest_checkpoint(self):
        """
        最新のチェックポイントのパスを取得します。
        """
        return os.path.join(self.log_dir_model, self._ckpt_name)
    
    def get_last_step(self):
        """
        復元した際に取得したステップを返します。
        """
        return getattr(self, 'start_step', 0)
