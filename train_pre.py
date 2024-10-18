import os
import torch
import argparse
import yaml
from torch.utils.data import DataLoader
from cwvae import build_model
from data_loader import load_dataset
import tools
import wandb
from datetime import datetime
from loggers.checkpoint import Checkpoint
from torchvision.utils import save_image

# 予測画像を保存する関数
def save_generated_images(epoch, sample_idx, predictions, gt_frames, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # 予測結果の保存
    predictions_save_path = os.path.join(output_dir, f"epoch_{epoch+1}_sample_{sample_idx}_predictions.png")
    save_image(predictions, predictions_save_path, nrow=8)
    print(f"予測画像を保存しました: {predictions_save_path}")

    # 実際の未来のフレームの保存
    gt_save_path = os.path.join(output_dir, f"epoch_{epoch+1}_sample_{sample_idx}_ground_truth.png")
    save_image(gt_frames, gt_save_path, nrow=8)
    print(f"実際の未来のフレームを保存しました: {gt_save_path}")

# メイン関数の定義
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", default="./logs", type=str, help="ログディレクトリのパス")
    parser.add_argument("--datadir", default="./minerl_navigate/", type=str, help="データディレクトリのパス")
    parser.add_argument("--config", default="./configs/minerl.yml", type=str, help="設定ファイル（YAML）のパス")
    parser.add_argument("--base-config", default="./configs/base_config.yml", type=str, help="ベース設定ファイルのパス")
    
    args = parser.parse_args()

    # 設定ファイルの読み込み
    cfg = tools.read_configs(args.config, args.base_config, datadir=args.datadir, logdir=args.logdir)

    # wandbの初期化
    wandb.init(project="CW-VAE", config=cfg)

    # デバイスの設定
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    cfg['device'] = device

    # 保存ディレクトリの設定
    dataset_name = cfg['dataset']
    model_name = "cwvae"
    current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_rootdir = os.path.join(cfg['logdir'], f"{dataset_name}_{model_name}_{current_time}")
    os.makedirs(exp_rootdir, exist_ok=True)

    # 設定を保存
    print(cfg)
    with open(os.path.join(exp_rootdir, "config.yml"), "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)

    # データセットをロード
    train_loader, val_loader = load_dataset(cfg['datadir'], cfg['batch_size'])

    # モデルの構築
    model_components = build_model(cfg)
    model = model_components["meta"]["model"]
    encoder = model_components["training"]["encoder"]
    decoder = model_components["training"]["decoder"]

    # トレーニングのセットアップ
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['lr'], eps=1e-04)
    model.to(device)

    # Checkpointクラスの初期化
    checkpoint = Checkpoint(exp_rootdir)

    # モデルの復元（存在する場合）
    start_epoch = 0
    if checkpoint.exists():
        print(f"モデルを {checkpoint.latest_checkpoint} から復元します")
        start_epoch = checkpoint.restore(model, optimizer)
        print(f"トレーニングをエポック {start_epoch} から再開します")
    else:
        model.apply(model.init_weights)

    # トレーニングループ
    print("トレーニングを開始します。")
    start_time = datetime.now()
    step = 0
    num_epochs = cfg['num_epochs']

    for epoch in range(start_epoch, num_epochs):
        model.train()
        for batch_idx, train_batch in enumerate(train_loader):
            print(f"Epoch [{epoch + 1}/{num_epochs}] - Batch [{batch_idx + 1}/{len(train_loader)}]")
            train_batch = train_batch.to(device)

            # シーケンス長の調整
            if train_batch.shape[1] > cfg['seq_len']:
                train_batch = train_batch[:, :cfg['seq_len']]
            
            optimizer.zero_grad()
            obs_encoded = encoder(train_batch)
            outputs_bot, _, priors, posteriors = model.hierarchical_unroll(obs_encoded)
            obs_decoded = model.decoder(outputs_bot)[0]
            
            # 損失の計算
            losses = model.compute_losses(
                obs=train_batch,
                obs_decoded=obs_decoded,
                priors=priors,
                posteriors=posteriors,
                dec_stddev=cfg['dec_stddev'],
                free_nats=cfg['free_nats'],
                beta=cfg['beta']
            )
            loss = losses["loss"]
            wandb.log({"train_loss": loss.item(), "step": step, "epoch": epoch})
            loss.backward()
            optimizer.step()
            step += 1

        # 検証と生成画像の保存
        model.eval()
        with torch.no_grad():
            val_losses = []
            for sample_idx, val_batch in enumerate(val_loader):
                if sample_idx >= 1:
                    break
                val_batch = val_batch.to(device)
                context_frames = val_batch[:, :36]
                future_frames_gt = val_batch[:, 36:100]

                # エンコーダーとデコーダーを使用した予測
                val_obs_encoded = encoder(context_frames)
                obs_encoded_full = [torch.cat([level, torch.zeros_like(level)], dim=1) for level in val_obs_encoded]
                outputs_bot, _, priors, posteriors = model.hierarchical_unroll(obs_encoded_full)
                predictions = model.decoder(outputs_bot[:, 36:])[0]

                # 予測画像を保存
                save_generated_images(epoch, sample_idx, predictions, future_frames_gt, exp_rootdir)

                # 損失計算
                val_losses_dict = model.compute_losses(
                    obs=val_batch,
                    obs_decoded=predictions,
                    priors=priors,
                    posteriors=posteriors,
                    dec_stddev=cfg['dec_stddev'],
                    free_nats=cfg['free_nats'],
                    beta=cfg['beta']
                )
                val_loss = val_losses_dict["loss"].item()
                val_losses.append(val_loss)

            average_val_loss = sum(val_losses) / len(val_losses)
            wandb.log({"val_loss": average_val_loss, "epoch": epoch})
            model.train()

        # モデルの保存
        checkpoint_path = os.path.join(exp_rootdir, f"checkpoint_epoch_{epoch + 1}.pth")
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'cfg': cfg,
        }, checkpoint_path)
        print(f"モデルをエポック {epoch + 1} で {checkpoint_path} として保存しました。")

        # 最新のチェックポイントを保存
        checkpoint.save(model, optimizer, epoch)

    end_time = datetime.now()
    duration = end_time - start_time
    print(f"トレーニングが完了しました。終了時間: {end_time} | トレーニングにかかった時間: {duration}")
    wandb.log({"training_duration": str(duration), "end_time": str(end_time)})
    wandb.finish()
