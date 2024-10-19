import os
import torch
import yaml
import argparse
import numpy as np
import imageio
from torch.utils.data import DataLoader
from cwvae import build_model
from data_loader import load_dataset, transform
import tools

# 予測用の設定
def predict(model, cfg, dataloader, output_dir):
    device = cfg['device']
    model.eval()

    # 出力ディレクトリの作成
    os.makedirs(output_dir, exist_ok=True)

    with torch.no_grad():
        for batch_idx, data_batch in enumerate(dataloader):
            data_batch = data_batch.to(device)

            # データの長さをcfg['seq_len']に合わせる
            if data_batch.shape[1] > cfg['seq_len']:
                data_batch = data_batch[:, :cfg['seq_len']]

            # エンコーダーを通して特徴を抽出
            obs_encoded = model.encoder(data_batch)

            # オープンループによる予測
            ctx_len = cfg['open_loop_ctx']  # 観測部分の長さ
            pre_posteriors, pre_priors, post_priors, outputs_bot_level = model.open_loop_unroll(
                obs_encoded, ctx_len=ctx_len, use_observations=cfg.get('use_obs', True)
            )

            # デコーダーを使って予測画像を生成
            prior_multistep_decoded, _ = model.decode_prior_multistep(post_priors[0]["mean"])  # 64フレームの生成

            # 予測結果の保存
            for b in range(prior_multistep_decoded.shape[0]):  # バッチ内の各サンプルについて
                sample_output_dir = os.path.join(output_dir, f"sample_{batch_idx}_{b}")
                os.makedirs(sample_output_dir, exist_ok=True)

                for t in range(prior_multistep_decoded.shape[1]):  # 各フレームについて
                    frame = prior_multistep_decoded[b, t].cpu().numpy()
                    frame = np.clip((frame + 1) * 127.5, 0, 255).astype(np.uint8)  # [-1, 1]から[0, 255]にスケール
                    frame_path = os.path.join(sample_output_dir, f"frame_{t:03d}.png")
                    imageio.imwrite(frame_path, frame.transpose(1, 2, 0))  # (C, H, W) -> (H, W, C)

                # GIFとして保存
                gif_path = os.path.join(output_dir, f"sample_{batch_idx}_{b}.gif")
                imageio.mimsave(gif_path, [imageio.imread(os.path.join(sample_output_dir, f"frame_{t:03d}.png")) for t in range(prior_multistep_decoded.shape[1])], fps=10)

            print(f"Batch {batch_idx} predictions saved to {output_dir}")

    print("Prediction complete.")

# メイン関数
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", default="./logs", type=str, help="ログディレクトリのパス")
    parser.add_argument("--datadir", default="./minerl_navigate/", type=str, help="データディレクトリのパス")
    parser.add_argument("--config", default="./configs/minerl.yml", type=str, help="設定ファイル（YAML）のパス")
    parser.add_argument("--base-config", default="./configs/base_config.yml", type=str, help="ベース設定ファイルのパス")
    parser.add_argument("--output-dir", default="./predictions", type=str, help="予測結果の出力ディレクトリ")
    args = parser.parse_args()

    # 設定ファイルの読み込み
    cfg = tools.read_configs(args.config, args.base_config, datadir=args.datadir, logdir=args.logdir)

    # デバイスの設定
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    cfg['device'] = device

    # データセットをロード（validation dataを使用）
    _, val_loader = load_dataset(cfg['datadir'], cfg['batch_size'], transform=transform)

    # モデルの構築
    model_components = build_model(cfg)
    model = model_components["meta"]["model"]

    # チェックポイントのロード
    checkpoint_path = "/home/yamada_24/cwvae/logs/minerl_cwvae_20241019_190635/model/model.pth"
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        print("Checkpoint loaded.")
    else:
        print("No checkpoint found. Please ensure the model is trained before running predictions.")
        exit()

    # 予測を行い、結果を保存
    predict(model, cfg, val_loader, args.output_dir)

