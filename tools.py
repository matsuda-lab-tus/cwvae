import yaml
from pathlib import Path
import numpy as np
import imageio
import os
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.autograd import Variable
from skimage.metrics import structural_similarity as ssim_metric
from skimage.metrics import peak_signal_noise_ratio as psnr_metric
from torchvision.utils import save_image

class AttrDict(dict):
    """
    A dictionary class that allows attribute-style access to dictionary keys.
    """
    __setattr__ = dict.__setitem__
    __getattr__ = dict.__getitem__

class Module(nn.Module):
    """
    A custom PyTorch module to lazily instantiate layers and cache them for reuse.
    """
    def __init__(self):
        super().__init__()
        self._modules = {}

    def get(self, name, PyTorch_layer, *args, **kwargs):
        """
        Retrieves a cached layer or creates it if it does not exist.
        """
        if name not in self._modules:
            self._modules[name] = PyTorch_layer(*args, **kwargs)
        return self._modules[name]

class Step:
    """
    A simple step counter to keep track of iterations.
    """
    def __init__(self):
        self._step = nn.Parameter(torch.tensor(0), requires_grad=False)

    def increment(self):
        self._step.data += 1

    def __call__(self):
        return self._step.item()

def exp_name(cfg, model_dir_prefix=None):
    """
    Generates an experiment name based on the configuration.
    """
    exp_name = f"{cfg.dataset}_cwvae_{cfg.cell_type.lower()}"
    exp_name += f"_{cfg.levels}l_f{cfg.tmp_abs_factor}"
    exp_name += f"_decsd{cfg.dec_stddev}"
    exp_name += f"_enchl{cfg.enc_dense_layers}_ences{cfg.enc_dense_embed_size}_edchnlmult{cfg.channels_mult}"
    exp_name += f"_ss{cfg.cell_stoch_size}_ds{cfg.cell_deter_size}_es{cfg.cell_embed_size}"
    exp_name += f"_seq{cfg.seq_len}_lr{cfg.lr}_bs{cfg.batch_size}"
    return exp_name

def validate_config(cfg):
    """
    Validates the configuration to ensure compatibility and integrity.
    """
    assert cfg.channels is not None and cfg.channels > 0, f"Incompatible channels = {cfg.channels} found in config."
    assert cfg.open_loop_ctx % (cfg.tmp_abs_factor ** (cfg.levels - 1)) == 0, \
        f"Incompatible open-loop context length {cfg.open_loop_ctx} and temporal abstraction factor {cfg.tmp_abs_factor} for levels {cfg.levels}"
    assert cfg.datadir is not None, "data root directory cannot be None."
    assert cfg.logdir is not None, "log root directory cannot be None."

def read_configs(config_path, base_config_path=None, **kwargs):
    """
    Reads and parses the configuration files (YAML format) and returns a configuration dictionary.
    """
    class TorchDeviceLoader(yaml.SafeLoader):
        pass

    def torch_device_constructor(loader, node):
        sequence = loader.construct_sequence(node)
        if isinstance(sequence, list) and len(sequence) == 1:
            return torch.device(sequence[0])
        else:
            raise ValueError(f"Invalid sequence format for torch.device: {sequence}")

    # Register custom tag to read torch.device from YAML
    TorchDeviceLoader.add_constructor('tag:yaml.org,2002:python/object/apply:torch.device', torch_device_constructor)

    if base_config_path is not None:
        base_config = yaml.safe_load(Path(base_config_path).read_text())
        config = base_config.copy()
        config.update(yaml.load(Path(config_path).read_text(), Loader=TorchDeviceLoader))
        assert len(set(config).difference(base_config)) == 0, "Found new keys in config. Make sure to set them in base_config first."
    else:
        with open(config_path, "r") as f:
            config = yaml.load(f, Loader=TorchDeviceLoader)

    config = AttrDict(config)

    if kwargs.get("datadir", None) is not None:
        config.datadir = kwargs["datadir"]
    if kwargs.get("logdir", None) is not None:
        config.logdir = kwargs["logdir"]

    validate_config(config)
    return config

def scan(cell, inputs, use_obs, initial):
    """
    Scans through the input sequence and applies the RNN cell at each time step.
    """
    assert initial is not None, "initial cannot be None. Pass zero_state instead."
    
    outputs = []
    state = initial

    for t in range(inputs.size(1)):  # Iterate over the sequence length
        inp_t = inputs[:, t]  # Extract input at time step t
        out, state = cell(state, inp_t, use_obs=use_obs)
        outputs.append(out)

    outputs = torch.stack(outputs, dim=1)  # Stack the outputs back to (B, T, ...)
    return outputs, state

def _to_padded_strip(images):
    """
    Converts a sequence of images into a single padded strip.
    """
    if len(images.shape) <= 3:
        images = np.expand_dims(images, -1)  # Add channel dimension if not present
    c_dim = images.shape[-1]
    x_dim = images.shape[-3]
    y_dim = images.shape[-2]
    padding = 1
    result = np.zeros(
        (x_dim, y_dim * images.shape[0] + padding * (images.shape[0] - 1), c_dim),
        dtype=np.uint8,
    )
    for i in range(images.shape[0]):
        result[:, i * y_dim + i * padding : (i + 1) * y_dim + i * padding, :] = images[i]
    if result.shape[-1] == 1:
        result = np.reshape(result, result.shape[:2])  # Remove channel dimension if grayscale
    return result

def compute_metrics(gt, pred):
    """
    SSIMおよびPSNRを計算する関数。
    
    Args:
    gt (np.ndarray): グラウンドトゥルース画像 (batch_size, seq_len, channels, height, width)
    pred (np.ndarray): 予測画像 (batch_size, seq_len, channels, height, width)
    
    Returns:
    np.ndarray, np.ndarray: 各バッチおよび各時刻に対するSSIMとPSNRの配列
    """
    # gt と pred の形状を確認
    print(f"[DEBUG] gt shape: {gt.shape}")
    print(f"[DEBUG] pred shape: {pred.shape}")

    # GTと予測の形状が異なる場合、修正する
    if gt.shape != pred.shape:
        # バッチ次元が異なる場合、GTの形状をPredに合わせる
        if gt.shape[0] != pred.shape[0]:
            if gt.shape[0] == 64 and pred.shape[0] == 1:
                gt = np.expand_dims(gt, axis=0)
            elif gt.shape[0] == 1 and pred.shape[0] == 64:
                pred = np.squeeze(pred, axis=0)
            else:
                raise ValueError(f"Unexpected batch size mismatch. GT shape: {gt.shape}, Pred shape: {pred.shape}")

    # バッチサイズとシーケンス長を取得
    bs = pred.shape[0]  # batch_size
    T = pred.shape[1]   # seq_len

    # SSIMとPSNRを格納する配列を初期化
    ssim = np.zeros((bs, T))
    psnr = np.zeros((bs, T))

    # 各バッチおよび各時刻に対してメトリクスを計算
    for i in range(bs):
        for t in range(T):
            # SSIMとPSNRを計算（各フレームについて1回計算する）
            ssim[i, t] = ssim_metric(gt[i, t], pred[i, t], multichannel=True, data_range=gt[i, t].max() - gt[i, t].min())
            psnr[i, t] = psnr_metric(gt[i, t], pred[i, t], data_range=gt[i, t].max() - gt[i, t].min())

    return ssim, psnr

def plot_metrics(mean_metric, std_metric, logdir, metric_name):
    """
    Plots the mean and standard deviation of the given metrics over time.
    """
    x = np.arange(len(mean_metric))
    plt.figure()
    plt.plot(x, mean_metric, label=f"Mean {metric_name.upper()}")
    plt.fill_between(x, mean_metric - std_metric, mean_metric + std_metric, alpha=0.2)
    plt.xlabel("Time Step")
    plt.ylabel(metric_name.upper())
    plt.title(f"{metric_name.upper()} over Time")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(logdir, f"{metric_name}.png"))
    plt.close()

def save_as_grid(images, path, filename, nrow=8):
    """
    Saves a batch of images as a grid.
    """
    if isinstance(images, np.ndarray):
        images = torch.from_numpy(images).float() / 255.0  # Convert NumPy array to PyTorch tensor and normalize
    elif isinstance(images, torch.Tensor):
        images = images.float() / 255.0
    else:
        raise TypeError(f"Unsupported type for images: {type(images)}")

    if images.ndimension() == 3:
        images = images.unsqueeze(0)  # Add batch dimension if needed
    
    print(f"[DEBUG] save_as_grid - images shape: {images.shape}, min: {images.min().item()}, max: {images.max().item()}")

    save_path = os.path.join(path, filename)
    print(f"[DEBUG] Saving image grid to: {save_path}")

    try:
        save_image(images, save_path, nrow=nrow)
        print(f"[INFO] Image grid saved successfully to: {save_path}")
    except Exception as e:
        print(f"[ERROR] Failed to save image grid to {save_path}: {e}")
        import traceback
        traceback.print_exc()
