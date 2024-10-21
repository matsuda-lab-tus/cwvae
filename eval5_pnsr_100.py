import os
import torch
import matplotlib.pyplot as plt
from torchmetrics.image import StructuralSimilarityIndexMeasure as SSIM
from torchmetrics.image import PeakSignalNoiseRatio as PSNR
import torchvision.utils as vutils
import argparse
from datetime import datetime
from cwvae import build_model
from data_loader import load_dataset, transform
from loggers.checkpoint import Checkpoint
from tools import read_configs

# SSIMとPSNRを計算するための関数
def calculate_metrics_per_frame(predicted, ground_truth, device):
    ssim = SSIM(data_range=1.0).to(device)
    psnr = PSNR(data_range=1.0).to(device)

    ssim_values = []
    psnr_values = []

    # 各フレームごとにSSIMとPSNRを計算
    for frame_idx in range(predicted.size(1)):
        pred_frame = predicted[:, frame_idx, ...]  # shape: [batch, channels, height, width]
        gt_frame = ground_truth[:, frame_idx, ...]  # shape: [batch, channels, height, width]

        ssim_value = ssim(pred_frame, gt_frame)
        psnr_value = psnr(pred_frame, gt_frame)

        ssim_values.append(ssim_value.item())
        psnr_values.append(psnr_value.item())

    return ssim_values, psnr_values

# グラフを保存する関数
def plot_and_save_metrics(frames, ssim_scores, psnr_scores, output_dir):
    plt.figure(figsize=(10, 4))

    # SSIMのグラフ
    plt.subplot(1, 2, 1)
    plt.plot(frames, ssim_scores, label='SSIM', color='red')
    plt.xlabel('Distance in Frames')
    plt.ylabel('SSIM')
    plt.title('SSIM over Time')
    plt.grid(True)

    # PSNRのグラフ
    plt.subplot(1, 2, 2)
    plt.plot(frames, psnr_scores, label='PSNR', color='blue')
    plt.xlabel('Distance in Frames')
    plt.ylabel('PSNR')
    plt.title('PSNR over Time')
    plt.grid(True)

    # グラフの保存
    plot_path = os.path.join(output_dir, "ssim_psnr_plot.png")
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()
    print(f"SSIMとPSNRのグラフを保存しました: {plot_path}")

# メイン関数の開始
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", default="./logs", type=str, help="ログディレクトリのパス")
    parser.add_argument("--datadir", default="./minerl_navigate/", type=str, help="データディレクトリのパス")
    parser.add_argument("--config", default="./configs/minerl.yml", type=str, help="設定ファイル（YAML）のパス")
    parser.add_argument("--base-config", default="./configs/base_config.yml", type=str, help="ベース設定ファイルのパス")
    parser.add_argument("--checkpoint", default="/home/yamada_24/cwvae/logs/minerl_cwvae_20241020_212029/model/model.pth", type=str, help="読み込むチェックポイントファイルのパス")

    args = parser.parse_args()
    cfg = read_configs(args.config, args.base_config, datadir=args.datadir, logdir=args.logdir)
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    cfg['device'] = device

    # 保存ディレクトリの設定
    dataset_name = cfg['dataset']
    model_name = "cwvae"
    current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_rootdir = os.path.join(cfg['logdir'], f"{dataset_name}_{model_name}_{current_time}")
    os.makedirs(exp_rootdir, exist_ok=True)

    # データセットをロード
    _, val_loader = load_dataset(cfg['datadir'], cfg['batch_size'], seq_len=cfg['seq_len'], transform=transform)

    # モデルの構築
    model_components = build_model(cfg)
    model = model_components["meta"]["model"]
    start_epoch = 0
    if os.path.exists(args.checkpoint):
        print(f"指定されたチェックポイント {args.checkpoint} からモデルを復元します")
        checkpoint_data = torch.load(args.checkpoint)
        model.load_state_dict(checkpoint_data['model_state_dict'])
        start_epoch = checkpoint_data.get('epoch', 0)
        print(f"トレーニングをエポック {start_epoch} から再開します")

    # 評価モードに切り替え
    model.eval()
    with torch.no_grad():
        frames = list(range(cfg['seq_len']))
        ssim_scores = [0] * cfg['seq_len']
        psnr_scores = [0] * cfg['seq_len']
        num_batches = 0

        # 検証データで評価
        for batch_idx, val_batch in enumerate(val_loader):
            val_batch = val_batch.to(device)
            adjusted_val_batch = val_batch.view(-1, 100, 3, 64, 64)
            val_obs_encoded = model.encoder(adjusted_val_batch)
            outputs_bot, _, val_priors, val_posteriors = model.hierarchical_unroll(val_obs_encoded)
            val_obs_decoded = model.decoder(outputs_bot)[0]

            # SSIMとPSNRの計算（フレームごと）
            ssim_values, psnr_values = calculate_metrics_per_frame(val_obs_decoded, val_batch.view(-1, 100, 3, 64, 64), device)
            
            # 各フレームのスコアを合計（後で平均を取る）
            ssim_scores = [s + v for s, v in zip(ssim_scores, ssim_values)]
            psnr_scores = [p + v for p, v in zip(psnr_scores, psnr_values)]
            num_batches += 1

            print(f"Batch {batch_idx}: SSIM={ssim_values}, PSNR={psnr_values}")

        # 各フレームごとの平均スコアを計算
        ssim_scores = [s / num_batches for s in ssim_scores]
        psnr_scores = [p / num_batches for p in psnr_scores]

        # SSIMとPSNRのグラフをプロットして保存
        plot_and_save_metrics(frames, ssim_scores, psnr_scores, exp_rootdir)
