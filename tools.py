import yaml
from pathlib import Path
import numpy as np
import imageio
import os
from skimage.metrics import peak_signal_noise_ratio as psnr_metric
from skimage.metrics import structural_similarity as ssim_metric
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.autograd import Variable


class AttrDict(dict):
    __setattr__ = dict.__setitem__
    __getattr__ = dict.__getitem__


class Module(nn.Module):
    def __init__(self):
        super().__init__()
        self._modules = {}

    def get(self, name, PyTorch_layer, *args, **kwargs):
        if name not in self._modules:
            self._modules[name] = PyTorch_layer(*args, **kwargs)
        return self._modules[name]


class Step:
    def __init__(self):
        self._step = nn.Parameter(torch.tensor(0), requires_grad=False)

    def increment(self):
        self._step.data += 1

    def __call__(self):
        return self._step.item()


def exp_name(cfg, model_dir_prefix=None):
    exp_name = f"{cfg.dataset}_cwvae_{cfg.cell_type.lower()}"
    exp_name += f"_{cfg.levels}l_f{cfg.tmp_abs_factor}"
    exp_name += f"_decsd{cfg.dec_stddev}"
    exp_name += f"_enchl{cfg.enc_dense_layers}_ences{cfg.enc_dense_embed_size}_edchnlmult{cfg.channels_mult}"
    exp_name += f"_ss{cfg.cell_stoch_size}_ds{cfg.cell_deter_size}_es{cfg.cell_embed_size}"
    exp_name += f"_seq{cfg.seq_len}_lr{cfg.lr}_bs{cfg.batch_size}"
    return exp_name


def validate_config(cfg):
    assert (
        cfg.channels is not None and cfg.channels > 0
    ), f"Incompatible channels = {cfg.channels} found in config."
    assert (
        cfg.open_loop_ctx % (cfg.tmp_abs_factor ** (cfg.levels - 1)) == 0
    ), f"Incompatible open-loop context length {cfg.open_loop_ctx} and temporal abstraction factor {cfg.tmp_abs_factor} for levels {cfg.levels}"
    assert cfg.datadir is not None, "data root directory cannot be None."
    assert cfg.logdir is not None, "log root directory cannot be None."


def read_configs(config_path, base_config_path=None, **kwargs):
    if base_config_path is not None:
        base_config = yaml.safe_load(Path(base_config_path).read_text())
        config = base_config.copy()
        config.update(yaml.safe_load(Path(config_path).read_text()))
        assert len(set(config).difference(base_config)) == 0, "Found new keys in config. Make sure to set them in base_config first."
    else:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
    config = AttrDict(config)

    if kwargs.get("datadir", None) is not None:
        config.datadir = kwargs["datadir"]
    if kwargs.get("logdir", None) is not None:
        config.logdir = kwargs["logdir"]

    validate_config(config)
    return config


def scan(cell, inputs, use_obs, initial):
    assert initial is not None, "initial cannot be None. Pass zero_state instead."
    
    outputs = []
    state = initial

    for t in range(inputs.size(1)):  # (B, T, ...)
        inp_t = inputs[:, t]  # Extract input at time step t
        out, state = cell(state, inp_t, use_obs=use_obs)
        outputs.append(out)

    outputs = torch.stack(outputs, dim=1)  # Stack back to (B, T, ...)
    return outputs, state


def _to_padded_strip(images):
    if len(images.shape) <= 3:
        images = np.expand_dims(images, -1)
    c_dim = images.shape[-1]
    x_dim = images.shape[-3]
    y_dim = images.shape[-2]
    padding = 1
    result = np.zeros(
        (x_dim, y_dim * images.shape[0] + padding * (images.shape[0] - 1), c_dim),
        dtype=np.uint8,
    )
    for i in range(images.shape[0]):
        result[:, i * y_dim + i * padding : (i + 1) * y_dim + i * padding, :] = images[
            i
        ]
    if result.shape[-1] == 1:
        result = np.reshape(result, result.shape[:2])
    return result


def save_as_grid(images, save_dir, filename, strip_width=50):
    results = []
    if images.shape[0] < strip_width:
        results.append(_to_padded_strip(images))
    else:
        for i in range(0, images.shape[0], strip_width):
            results.append(_to_padded_strip(images[i : i + strip_width]))
    grid = np.concatenate(results, axis=0)
    imageio.imwrite(os.path.join(save_dir, filename), grid)
    print(f"Written grid file {os.path.join(save_dir, filename)}")


def compute_metrics(gt, pred):
    gt = np.transpose(gt, [0, 1, 4, 2, 3])
    pred = np.transpose(pred, [0, 1, 4, 2, 3])
    bs = gt.shape[0]
    T = gt[0].shape[0]
    ssim = np.zeros((bs, T))
    psnr = np.zeros((bs, T))
    for i in range(bs):
        for t in range(T):
            for c in range(gt[i][t].shape[0]):
                ssim[i, t] += ssim_metric(gt[i][t][c], pred[i][t][c])
                psnr[i, t] += psnr_metric(gt[i][t][c], pred[i][t][c])
            ssim[i, t] /= gt[i][t].shape[0]
            psnr[i, t] /= gt[i][t].shape[0]

    return ssim, psnr


def plot_metrics(metrics, logdir, name):
    mean_metric = np.squeeze(np.mean(metrics, axis=0))
    stddev_metric = np.squeeze(np.std(metrics, axis=0))
    np.savez(os.path.join(logdir, f"{name}_mean.npz"), mean_metric)
    np.savez(os.path.join(logdir, f"{name}_stddev.npz"), stddev_metric)

    plt.figure()
    axes = plt.gca()
    axes.yaxis.grid(True)
    plt.plot(mean_metric, color="blue")
    axes.fill_between(
        np.arange(0, mean_metric.shape[0]),
        mean_metric - stddev_metric,
        mean_metric + stddev_metric,
        color="blue",
        alpha=0.4,
    )
    plt.savefig(os.path.join(logdir, f"{name}_range.png"))
