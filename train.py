import os
import torch
import argparse
import yaml
from torch.utils.data import DataLoader
from cwvae import build_model
from data_loader import load_dataset
import tools
import wandb
from datetime import datetime  # 終了時間の取得に使用

# Checkpointクラスのインポート
from loggers.checkpoint import Checkpoint

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

    # わかりやすい名前を使用した保存ディレクトリの設定
    dataset_name = cfg['dataset']
    model_name = "cwvae"  # モデル名
    current_time = datetime.now().strftime('%Y%m%d_%H%M%S')  # 日時情報を短縮
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
        # モデルのパラメータを初期化
        model.apply(model.init_weights)

    # トレーニングループ
    print("トレーニングを開始します。")
    start_time = datetime.now()  # トレーニング開始時間
    step = 0
    num_epochs = cfg['num_epochs']

    for epoch in range(start_epoch, num_epochs):
        model.train()
        for batch_idx, train_batch in enumerate(train_loader):  # DataLoaderからバッチを取得
            print(f"Epoch [{epoch + 1}/{num_epochs}] - Batch [{batch_idx + 1}/{len(train_loader)}]")
            train_batch = train_batch.to(device)  # デバイスに送る（GPU or CPU）

            # train_batch のシーケンス長を cfg['seq_len'] に揃える
            if train_batch.shape[1] > cfg['seq_len']:
                train_batch = train_batch[:, :cfg['seq_len']]

            # train_batch の次元チェック
            print(f"train_batch shape after slicing: {train_batch.shape}")
            
            optimizer.zero_grad()
            
            # エンコーダーを通して特徴量を抽出
            obs_encoded = encoder(train_batch)

            # モデルの出力を取得
            outputs_bot, _, priors, posteriors = model.hierarchical_unroll(obs_encoded)
            
            # デコーダーを通して生成画像を得る
            obs_decoded = decoder(outputs_bot)
            print(f"obs_decoded shape: {obs_decoded.shape}")
            
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
            print(f"Loss calculated: {loss.item()}")

            # wandbに損失をログ
            wandb.log({"train_loss": loss.item(), "step": step, "epoch": epoch})

            loss.backward()
            
            # 勾配クリッピングの実施（必要な場合）
            if cfg['clip_grad_norm_by'] is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['clip_grad_norm_by'])
            
            optimizer.step()

            step += 1

        # エポック終了時に検証損失の計算
        model.eval()
        with torch.no_grad():
            val_losses = []
            for val_batch in val_loader:
                val_batch = val_batch.to(device)
                if val_batch.shape[1] > cfg['seq_len']:
                    val_batch = val_batch[:, :cfg['seq_len']]
                val_obs_encoded = encoder(val_batch)
                val_outputs_bot, _, val_priors, val_posteriors = model.hierarchical_unroll(val_obs_encoded)
                val_obs_decoded = decoder(val_outputs_bot)
                val_losses_dict = model.compute_losses(
                    obs=val_batch,
                    obs_decoded=val_obs_decoded,
                    priors=val_priors,
                    posteriors=val_posteriors,
                    dec_stddev=cfg['dec_stddev'],
                    free_nats=cfg['free_nats'],
                    beta=cfg['beta']
                )
                val_loss = val_losses_dict["loss"].item()
                val_losses.append(val_loss)
                print(f"Validation Loss for current batch: {val_loss}")
            average_val_loss = sum(val_losses) / len(val_losses)
            wandb.log({"val_loss": average_val_loss, "epoch": epoch})
            model.train()

        # エポックごとにモデルを保存（エポック番号を含む）
        checkpoint_path = os.path.join(exp_rootdir, f"checkpoint_epoch_{epoch + 1}.pth")
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'cfg': cfg,
        }, checkpoint_path)
        print(f"モデルをエポック {epoch + 1} で {checkpoint_path} として保存しました。")

        # Checkpointクラスによる最新チェックポイントの保存（必要な場合）
        checkpoint.save(model, optimizer, epoch)

    # 終了時間をログ
    end_time = datetime.now()
    duration = end_time - start_time
    print(f"トレーニングが完了しました。終了時間: {end_time} | トレーニングにかかった時間: {duration}")
    wandb.log({"training_duration": str(duration), "end_time": str(end_time)})
    wandb.finish()
