from argparse import ArgumentParser
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional, List
import yaml
import torch
import numpy as np

# Config class definition with default values and properties
@dataclass
class Config:
    # General paths
    config: str = "./configs/minerl.yml"  # Path to config yaml file
    datadir: str = "./minerl_navigate"  # Path to root data directory
    logdir: str = "./logs"  # Path to root log directory

    # Model configuration
    levels: int = 3  # Number of levels in the hierarchy
    tmp_abs_factor: int = 6  # Temporal abstraction factor used at each level
    dec_stddev: float = 1.0  # Standard deviation of the decoder distribution
    enc_dense_layers: int = 3  # Number of dense hidden layers at each level
    enc_dense_embed_size: int = 1000  # Size of dense hidden embeddings
    cell_stoch_size: int = 20
    cell_deter_size: int = 200
    cell_embed_size: int = 200
    gru_hidden_prior: int = 200
    gru_hidden_posterior: int = 200
    cell_min_stddev: float = 0.0001  # Minimum standard deviation of prior and posterior distributions
    use_obs: Optional[str] = None  # String of T/Fs per level, e.g. TTF to skip obs at the top level
    channels_mult: int = 1  # Multiplier for the number of channels in the conv encoder
    filters: int = 32  # Base number of channels in the convolutions

    # Dataset settings
    dataset: str = "minerl"  # Dataset type
    seq_len: int = 100  # Length of training sequences
    eval_seq_len: int = 1000  # Total length of evaluation sequences
    channels: int = 3  # Number of channels in the output video

    # Training configuration
    lr: float = 0.0003
    batch_size: int = 50
    num_epochs: int = 300
    clip_grad_norm_by: float = 10000
    seed: int = np.random.randint(np.iinfo(np.int32).max)

    # Summary and evaluation settings
    open_loop_ctx: int = 36  # Number of context frames for open loop prediction
    save_gifs: bool = True
    save_scalars_every: int = 1000
    save_gifs_every: int = 1000
    save_model_every: int = 1000
    save_named_model_every: int = 5000
    num_val_batches: int = 1
    num_examples: int = 100  # Number of examples to evaluate on
    num_samples: int = 1  # Samples to generate per example
    no_save_grid: bool = False  # Prevent saving grids of images

    # KL-related terms
    beta: Optional[float] = None
    free_nats: Optional[float] = None
    kl_grad_post_perc: Optional[float] = None

    # Default constraints (for specific fields)
    cell_type: str = "RSSMCell"
    cell_mean_only: str = "false"
    cell_reset_state: str = "false"

    def config_file(self, eval=False):
        return Path(self.logdir).parent / "config.yml" if eval else Path(self.config)

    def save(self):
        self.exp_rootdir.mkdir(parents=True, exist_ok=True)
        with (self.exp_rootdir / "config.yml").open("w") as f:
            yaml.dump(asdict(self), f, default_flow_style=False)

    @property
    def exp_rootdir(self):
        return Path(self.logdir) / self.dataset / self._run_name

    @property
    def _run_name(self):
        return f"{self.dataset}_cwvae_{self.cell_type.lower()}_" \
               f"{self.levels}l_f{self.tmp_abs_factor}_decsd{self.dec_stddev}_" \
               f"enchl{self.enc_dense_layers}_ences{self.enc_dense_embed_size}_" \
               f"edchnlmult{self.channels_mult}_ss{self.cell_stoch_size}_" \
               f"ds{self.cell_deter_size}_es{self.cell_embed_size}_seq{self.seq_len}_" \
               f"lr{self.lr}_bs{self.batch_size}"

    @property
    def total_filters(self):
        return self.filters * self.channels_mult

    @property
    def use_observations(self) -> List[bool]:
        if self.use_obs is None:
            return [True] * self.levels
        assert len(self.use_obs) == self.levels
        return [c == 'T' for c in self.use_obs]

    def load_dataset(self, eval=False):
        from data_loader import load_dataset  # Assuming a custom data loader exists
        train_loader, val_loader = load_dataset(self.datadir, self.batch_size)
        return train_loader if not eval else val_loader

# Function to parse configuration from command line arguments and YAML
def parse_config(eval=False):
    parser = ArgumentParser()
    for f in fields(Config):
        kwargs = dict(default=f.default, type=type(f.default)) if not f.type == bool else dict(action="store_true")
        parser.add_argument(f'--{f.name}', **kwargs)
    args = parser.parse_args()

    # Load config from YAML and update defaults
    config_file = Path(args.config) if args.config else Config().config_file(eval)
    with open(config_file, 'r') as f:
        yaml_config = yaml.safe_load(f)
    
    # Merge command line arguments and YAML configuration
    config = Config(**{**yaml_config, **vars(args)})
    return config
