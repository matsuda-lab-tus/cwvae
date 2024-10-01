import os
import torch
import torch.distributions as dist
from torch.utils.tensorboard import SummaryWriter
import loggers.gif_summary as gif_summary

class Summary:
    def __init__(self, log_dir_root, save_gifs=True, var_scope="cwvae"):
        self._log_dir_root = log_dir_root
        self._log_dir_train = os.path.join(self._log_dir_root, "train")
        self._log_dir_val = os.path.join(self._log_dir_root, "val")
        os.makedirs(self._log_dir_train, exist_ok=True)
        os.makedirs(self._log_dir_val, exist_ok=True)
        self._writer_train = SummaryWriter(self._log_dir_train)
        self._writer_val = SummaryWriter(self._log_dir_val)

        self._save_gifs = save_gifs
        self._var_scope = var_scope

        self.scalar_summary = []
        self.gif_summary = None

    def build_summary(self, cfg, model_components, **kwargs):
        assert self.scalar_summary == [], "Can only call self.scalar_summary once."

        # Scalar summaries (loss, learning rate, etc.)
        self.scalar_summary.append(('total_loss', model_components["training"]["loss"]))
        self.scalar_summary.append(('nll_term', model_components["training"]["nll_term"]))
        self.scalar_summary.append(('kl_term', model_components["training"]["kl_term"]))
        self.scalar_summary.append(('learning_rate', cfg.lr))
        self.scalar_summary.append(('grad_norm', kwargs.get("grad_norm", 0.0)))  # デフォルト値を追加

        # Adding per-level summaries.
        for lvl in range(cfg.levels):
            # KL(posterior || prior) at each level (avg across batch, sum across time).
            kld_level = model_components["training"]["kld_all_levels"][lvl]
            kl_mean = kld_level.sum(dim=1).mean()
            self.scalar_summary.append((f"avg_kl_prior_posterior__level_{lvl}", kl_mean))

            # Prior entropy.
            prior = model_components["training"]["priors"][lvl]
            prior_dist = dist.MultivariateNormal(prior["mean"], torch.diag_embed(prior["stddev"]))
            prior_entropy_mean = prior_dist.entropy().sum(dim=1).mean()
            self.scalar_summary.append((f"avg_entropy_prior__level_{lvl}", prior_entropy_mean))

            # Posterior entropy.
            posterior = model_components["training"]["posteriors"][lvl]
            posterior_dist = dist.MultivariateNormal(posterior["mean"], torch.diag_embed(posterior["stddev"]))
            posterior_entropy_mean = posterior_dist.entropy().sum(dim=1).mean()
            self.scalar_summary.append((f"avg_entropy_posterior__level_{lvl}", posterior_entropy_mean))

        # GIF summaries (optional).
        if self._save_gifs:
            # model_components に "open_loop_obs_decoded" が含まれているか確認
            if "open_loop_obs_decoded" in model_components and model_components["open_loop_obs_decoded"] is not None:
                print("Building GIF summaries...")
                prior_multistep = model_components["open_loop_obs_decoded"]["prior_multistep"]  # [batch, steps, channels, height, width]
                gt_multistep = model_components["open_loop_obs_decoded"]["gt_multistep"]        # [batch, steps, channels, height, width]

                # stddev_tensor を prior_multistep と同じデバイスに配置
                stddev_tensor = torch.tensor(cfg.dec_stddev, device=prior_multistep.device)

                # Normal distribution を作成
                prior_dist = dist.Normal(
                    prior_multistep,
                    stddev_tensor,
                )

                # image_summaries 関数にデコードされた priors と gt を渡す
                self.gif_summary = gif_summary.image_summaries(
                    prior_decoded=prior_multistep,
                    gt_multistep=gt_multistep,
                    clip_by=(0.0, 1.0),
                    name=self._var_scope,
                    max_batch=8,
                )
            else:
                print("Warning: 'open_loop_obs_decoded' not found in model_components. GIF summaries will be skipped.")

    def add_scalar(self, tag, scalar_value, global_step=None, train=True):
        """
        Adds a scalar value to TensorBoard.

        Args:
            tag (str): データの識別子
            scalar_value (float or int): 記録する値
            global_step (int, optional): 記録するステップ
            train (bool): Trueの場合はトレーニングのログに、Falseの場合は検証のログに記録
        """
        writer = self._writer_train if train else self._writer_val
        writer.add_scalar(tag, scalar_value, global_step)

    def save(self, step, train=True, model=None, optimizer=None):
        """
        Save scalar summaries to TensorBoard.

        Args:
            step: Current training step or epoch.
            train: Boolean indicating whether it's training or validation.
            model: Optional; model to save (if saving weights is needed).
            optimizer: Optional; optimizer to save (if saving optimizer state is needed).
        """
        writer = self._writer_train if train else self._writer_val

        # Log scalar summaries.
        for name, value in self.scalar_summary:
            writer.add_scalar(name, value, step)

        # Log GIF summaries if present.
        if self._save_gifs and self.gif_summary is not None:
            gifs = self.gif_summary
            for idx, gif in enumerate(gifs):
                writer.add_image(f"{self._var_scope}/gif_{idx}", gif, step, dataformats="HWC")

        # Optionally save model and optimizer states.
        if model is not None and optimizer is not None:
            checkpoint_path = os.path.join(self._log_dir_root, f"checkpoint_{step}.pth")
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'step': step,
            }, checkpoint_path)

        writer.flush()
