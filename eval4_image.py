# 必要なライブラリをインポート
import os
import torch
from torchmetrics.image import StructuralSimilarityIndexMeasure as SSIM
from torchmetrics.image import PeakSignalNoiseRatio as PSNR
import torchvision.utils as vutils
import torchvision.transforms as transforms
import argparse
import yaml
from datetime import datetime
from torch.utils.data import DataLoader
from cwvae import build_model
from data_loader import load_dataset, transform
from loggers.checkpoint import Checkpoint
from tools import read_configs
import matplotlib.pyplot as plt
import numpy as np

# SSIMとPSNRを計算するための関数
def calculate_metrics(predicted, ground_truth, device):
    # 時間次元をバッチとして扱うためにリシェイプ
    predicted = predicted.view(-1, *predicted.shape[2:])  # [フレーム数 x バッチサイズ, 3, 64, 64]
    ground_truth = ground_truth.view(-1, *ground_truth.shape[2:])  # 同様にリシェイプ

    ssim = SSIM(data_range=1.0).to(device)  # SSIMをデバイスに移動
    psnr = PSNR(data_range=1.0).to(device)  # PSNRもデバイスに移動

    # SSIMとPSNRを計算
    ssim_value = ssim(predicted, ground_truth)
    psnr_value = psnr(predicted, ground_truth)

    return ssim_value.item(), psnr_value.item()

# 予測画像と真の画像を横に並べて保存するための関数
def save_side_by_side_comparison(predicted, ground_truth, img_path):
    # 画像を正規化して[-1,1] -> [0,1]に変換
    predicted = (predicted + 1) / 2
    ground_truth = (ground_truth + 1) / 2

    # 予測画像と真の画像を横に結合
    comparison = torch.cat((ground_truth, predicted), dim=2)  # 横に並べるためにdim=2で結合

    # 画像を保存
    vutils.save_image(comparison, img_path)

# 予測画像と真の画像をグリッド表示してラベル付きで保存する関数
def save_comparison_images_with_labels(predicted, ground_truth, output_path, filename):
    # 各画像をリシェイプして、[バッチサイズ * フレーム数, 3, 64, 64]の形に変換
    predicted = predicted.view(-1, *predicted.shape[2:])
    ground_truth = ground_truth.view(-1, *ground_truth.shape[2:])
    
    # グリッド画像の作成 (横に並べるためにnrowを2に設定)
    combined = torch.cat((ground_truth, predicted), dim=2)  # 横に結合するためにdim=2を指定
    grid = vutils.make_grid(combined, nrow=10, normalize=True, value_range=(-1, 1))

    # ラベル付きで画像を表示・保存
    plt.figure(figsize=(20, 5))
    plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
    plt.axis('off')
    plt.title('Left: Ground Truth | Right: Prediction', fontsize=16)
    plt.savefig(os.path.join(output_path, filename))
    plt.close()

# メイン関数の開始
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", default="./logs", type=str, help="ログディレクトリのパス")
    parser.add_argument("--datadir", default="./minerl_navigate/", type=str, help="データディレクトリのパス")
    parser.add_argument("--config", default="./configs/minerl.yml", type=str, help="設定ファイル（YAML）のパス")
    parser.add_argument("--base-config", default="./configs/base_config.yml", type=str, help="ベース設定ファイルのパス")
    parser.add_argument("--checkpoint", default="/home/yamada_24/cwvae/logs/minerl_cwvae_20241020_212029/model/model.pth", type=str, help="読み込むチェックポイントファイルのパス")

    args = parser.parse_args()

    # 設定ファイルの読み込み
    cfg = read_configs(args.config, args.base_config, datadir=args.datadir, logdir=args.logdir)

    # デバイスの設定
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    cfg['device'] = device

    # 保存ディレクトリの設定
    dataset_name = cfg['dataset']
    model_name = "cwvae"
    current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_rootdir = os.path.join(cfg['logdir'], f"{dataset_name}_{model_name}_{current_time}")
    os.makedirs(exp_rootdir, exist_ok=True)

    # データセットをロード
    train_loader, val_loader = load_dataset(cfg['datadir'], cfg['batch_size'], seq_len=cfg['seq_len'], transform=transform)

    # モデルの構築
    model_components = build_model(cfg)
    model = model_components["meta"]["model"]
    encoder = model_components["training"]["encoder"]
    decoder = model_components["training"]["decoder"]

    # Checkpointの初期化
    start_epoch = 0
    if args.checkpoint is not None:
        # 指定されたチェックポイントファイルからモデルを読み込む
        if os.path.exists(args.checkpoint):
            print(f"指定されたチェックポイント {args.checkpoint} からモデルを復元します")
            checkpoint_data = torch.load(args.checkpoint)
            model.load_state_dict(checkpoint_data['model_state_dict'])
            start_epoch = checkpoint_data.get('epoch', 0)
            print(f"トレーニングをエポック {start_epoch} から再開します")
        else:
            print(f"指定されたチェックポイント {args.checkpoint} が見つかりません。新しいモデルで開始します。")
    else:
        model.apply(model.init_weights)

    # 評価モードに切り替え
    model.eval()
    with torch.no_grad():
        ssim_scores = []
        psnr_scores = []

        # 検証データで評価
        for batch_idx, val_batch in enumerate(val_loader):
            val_batch = val_batch.to(device)
            adjusted_val_batch = val_batch.view(-1, 100, 3, 64, 64)
            val_obs_encoded = model.encoder(adjusted_val_batch)
            outputs_bot, _, val_priors, val_posteriors = model.hierarchical_unroll(val_obs_encoded)
            val_obs_decoded = model.decoder(outputs_bot)[0]

            # SSIMとPSNRの計算
            ssim_value, psnr_value = calculate_metrics(val_obs_decoded, val_batch.view(-1, 100, 3, 64, 64), device)
            ssim_scores.append(ssim_value)
            psnr_scores.append(psnr_value)

            print(f"Batch {batch_idx}: SSIM={ssim_value}, PSNR={psnr_value}")

            # 画像の保存
            output_dir = os.path.join(exp_rootdir, f"val_outputs_epoch_{start_epoch + 1}")
            os.makedirs(output_dir, exist_ok=True)
            for i in range(val_obs_decoded.size(0)):
                img_path = os.path.join(output_dir, f"val_pred_comparison_{batch_idx * val_loader.batch_size + i}.png")
                try:
                    # 予測画像と真の画像を横に並べて保存
                    save_side_by_side_comparison(val_obs_decoded[i], val_batch.view(-1, 100, 3, 64, 64)[i], img_path)

                    # 予測画像と真の画像をラベル付きでグリッド表示して保存
                    save_comparison_images_with_labels(val_obs_decoded[i:i+1], val_batch.view(-1, 100, 3, 64, 64)[i:i+1], output_dir, f"comparison_grid_{batch_idx * val_loader.batch_size + i}.png")
                except Exception as e:
                    print(f"画像の保存中にエラーが発生しました: {e}")

        # SSIMとPSNRの平均をログ
        average_ssim = sum(ssim_scores) / len(ssim_scores)
        average_psnr = sum(psnr_scores) / len(psnr_scores)
        print(f"Average SSIM: {average_ssim}, Average PSNR: {average_psnr}")
