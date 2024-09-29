import torch
import torchvision.transforms as transforms
import numpy as np

def image_summaries(prior_decoded, gt_multistep, clip_by=(0.0, 1.0), name="cwvae", max_batch=8):
    """
    prior_decoded: Decoded prior images, shape [batch, steps, channels, height, width]
    gt_multistep: Ground truth images, shape [batch, steps, channels, height, width]
    Returns a list of GIFs
    """
    gifs = []
    # Clip the images
    prior_decoded = torch.clamp(prior_decoded, min=clip_by[0], max=clip_by[1])
    gt_multistep = torch.clamp(gt_multistep, min=clip_by[0], max=clip_by[1])

    # Convert to CPU and numpy
    prior_decoded = prior_decoded.cpu().detach().numpy()
    gt_multistep = gt_multistep.cpu().detach().numpy()

    for i in range(max_batch):
        prior = prior_decoded[i]  # [steps, channels, height, width]
        gt = gt_multistep[i]      # [steps, channels, height, width]

        # デバッグ用: 形状を出力
        print(f"Prior shape: {prior.shape}, GT shape: {gt.shape}")

        # Concatenate prior and gt along time axis
        frames = np.concatenate((prior, gt), axis=0)  # [2*steps, channels, height, width]

        # Transpose to [steps, height, width, channels]
        frames = frames.transpose(0, 2, 3, 1)

        # Normalize to [0, 255]
        frames = (frames * 255).astype(np.uint8)

        # Create a single image by stacking frames vertically
        gif = np.vstack(frames)

        gifs.append(gif)

    return gifs
