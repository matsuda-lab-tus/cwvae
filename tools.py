import yaml
from pathlib import Path
import torch
from torch import nn
import numpy as np
import imageio
import os
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim_metric
from skimage.metrics import peak_signal_noise_ratio as psnr_metric
from torchvision.utils import save_image
from yaml.constructor import ConstructorError

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
        # 文字列形式（スカラーノード）を処理
        if isinstance(node, yaml.ScalarNode):
            device_str = loader.construct_scalar(node)
            return torch.device(device_str)
        # リスト形式（シーケンスノード）を処理
        elif isinstance(node, yaml.SequenceNode):
            sequence = loader.construct_sequence(node)
            if isinstance(sequence, list) and len(sequence) == 2 and sequence[0] == 'cuda':
                return torch.device(f'cuda:{sequence[1]}')
        # どちらでもない場合はエラー
        raise ValueError(f"Invalid format for torch.device: {loader.construct_object(node)}")

    def attrdict_constructor(loader, node):
        return AttrDict(loader.construct_mapping(node, deep=True))

    # カスタムコンストラクターの登録
    TorchDeviceLoader.add_constructor('!torch.device', torch_device_constructor)
    TorchDeviceLoader.add_constructor('!attrdict', attrdict_constructor)

    if base_config_path is not None:
        base_config = yaml.safe_load(Path(base_config_path).read_text())
        config = base_config.copy()
        config.update(yaml.load(Path(config_path).read_text(), Loader=TorchDeviceLoader))
        # assert len(set(config).difference(base_config)) == 0, "Found new keys in config. Make sure to set them in base_config first."
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
    グラウンドトゥルース (gt) と予測 (pred) のSSIMおよびPSNRを計算する関数
    """
    seq_len = gt.shape[1]
    ssim = np.zeros((gt.shape[0], seq_len))
    psnr = np.zeros((gt.shape[0], seq_len))
    
    for i in range(gt.shape[0]):
        for t in range(seq_len):
            # SSIMの計算時にwin_sizeを5に設定
            ssim[i, t] = ssim_metric(
                gt[i, t], pred[i, t],
                multichannel=True,
                data_range=gt[i, t].max() - gt[i, t].min(),
                win_size=5,  # 画像サイズが64x64なので、5x5のウィンドウサイズを指定
                channel_axis=-1  # チャンネルが最後の軸（カラーチャンネル）
            )
            # PSNRの計算
            psnr[i, t] = 10 * np.log10(1 / np.mean((gt[i, t] - pred[i, t]) ** 2))
    
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

from torchvision.utils import save_image

def save_as_grid(images, full_save_path, nrow=8):
    try:
        save_image(images, full_save_path, nrow=nrow)
    except Exception as e:
        print(f"[ERROR] Failed to save image grid to {full_save_path}: {str(e)}")
