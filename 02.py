import os
import torch
import argparse
import yaml
from torch.utils.data import DataLoader
from cwvae import build_model
from data_loader import load_dataset
import tools
from datetime import datetime
from loggers.checkpoint import Checkpoint

# 予測関数: 100フレームの入力から残り100フレームを生成
# 予測関数: 100フレームの入力から残り100フレームを生成
def generate_predictions(cfg, model, encoder, decoder, val_loader, exp_rootdir):
    print("予測を開始します。")
    device = cfg['device']
    model.eval()
    with torch.no_grad():
        for i, val_batch in enumerate(val_loader):
            if i >= 1:  # 最初のバッチのみを使用
                break
            val_batch = val_batch.to(device)
            
            # 入力シーケンスを100フレームに制限し、残りを予測
            context_frames = val_batch[:, :100]  # [バッチサイズ, 100, チャネル数, 高さ, 幅]
            future_frames_gt = val_batch[:, 100:200]  # [バッチサイズ, 100, チャネル数, 高さ, 幅]

            # バリデーションデータをエンコーダが受け入れる形式に変換
            adjusted_val_batch = context_frames.view(-1, 100, 3, 64, 64)

            # エンコーダーを通して特徴量を抽出（コンテキスト部分）
            obs_encoded_context = encoder(adjusted_val_batch)

            # 各レベルのコンテキストを拡張して、未来の予測を行うためのシーケンスに調整
            obs_encoded_full = []
            future_length = 200 - 100  # 予測するフレーム数

            for level_idx, obs_context_level in enumerate(obs_encoded_context):
                obs_encoded_dim = obs_context_level.shape[2]
                batch_size = obs_context_level.shape[0]
                downsample_factor = 2 ** level_idx
                future_length_level = future_length // downsample_factor

                if future_length_level == 0:
                    future_length_level = 1

                # 未来部分をゼロテンソルで埋めて、モデルに予測させる準備をする
                obs_future_level = torch.zeros(
                    batch_size,
                    future_length_level,
                    obs_encoded_dim,
                    device=device
                )

                # コンテキストと未来の部分を結合
                obs_full_level = torch.cat([obs_context_level, obs_future_level], dim=1)
                obs_encoded_full.append(obs_full_level)

            # ヒエラルキカルアンロールで予測を実行
            outputs_bot, _, priors, posteriors = model.hierarchical_unroll(obs_encoded_full)
            outputs_bot_future = outputs_bot[:, 70:140]  # 100フレーム以降の予測部分を取得

            # デコーダーを使って未来のフレームを画像に変換
            preds = decoder(outputs_bot_future)[0]  # [バッチサイズ, 64, 3, 64,64]

            # 保存ディレクトリの設定
            output_dir = os.path.join(exp_rootdir, f"predictions_sample_{i}")
            os.makedirs(output_dir, exist_ok=True)

            # バッチ内の各サンプルについて保存
            for sample_idx in range(preds.shape[0]):
                sample_preds = preds[sample_idx]  # [100, 3, 64, 64]
                sample_gt = future_frames_gt[sample_idx]  # [100, 3, 64, 64]

                # 予測結果のファイルパス
                pred_save_path = os.path.join(output_dir, f"predictions_sample_{sample_idx}.png")
                try:
                    tools.save_as_grid(sample_preds, pred_save_path)
                    print(f"予測結果を {pred_save_path} に保存しました。")
                except Exception as e:
                    print(f"[ERROR] Failed to save image grid to {pred_save_path}: {str(e)}")

                # Ground Truth のファイルパス
                gt_save_path = os.path.join(output_dir, f"ground_truth_sample_{sample_idx}.png")
                try:
                    tools.save_as_grid(sample_gt, gt_save_path)
                    print(f"実際の未来のフレームを {gt_save_path} に保存しました。")
                except Exception as e:
                    print(f"[ERROR] Failed to save image grid to {gt_save_path}: {str(e)}")

# メイン関数
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", default="./logs", type=str, help="ログディレクトリのパス")
    parser.add_argument("--datadir", default="./minerl_navigate/", type=str, help="データディレクトリのパス")
    parser.add_argument("--checkpoint", default="/home/yamada_24/cwvae/logs/minerl_cwvae_20241020_214406/model/model.pth", type=str, help="学習済みモデルのチェックポイントのパス")
    parser.add_argument("--config", default="./configs/minerl.yml", type=str, help="設定ファイル（YAML）のパス")
    parser.add_argument("--base-config", default="./configs/base_config.yml", type=str, help="ベース設定ファイルのパス")
    args = parser.parse_args()

    # 設定ファイルの読み込み
    cfg = tools.read_configs(args.config, args.base_config, datadir=args.datadir, logdir=args.logdir)
    
    # デバイス設定
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    cfg['device'] = device

    # データセットの読み込み
    _, val_loader = load_dataset(cfg['datadir'], cfg['batch_size'])

    # モデルの構築
    model_components = build_model(cfg)
    model = model_components["meta"]["model"]
    encoder = model_components["training"]["encoder"]
    decoder = model_components["training"]["decoder"]

    # 学習済みモデルのロード
    print(f"[DEBUG] Loading checkpoint from: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print("[DEBUG] Checkpoint loaded successfully.")

    model.to(device)
    print("[DEBUG] Model moved to device.")

    # 予測の実行
    generate_predictions(cfg, model, encoder, decoder, val_loader, args.logdir)
