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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg['device'] = device

    # 現在の日付と時間を使ってフォルダ名を生成
    current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_rootdir = os.path.join(cfg['logdir'], f"{cfg['dataset']}_session_{current_time}")
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

    # モデルの復元（存在する場合）
    start_epoch = 0
    checkpoint_dir = os.path.join(exp_rootdir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    checkpoint_path = os.path.join(checkpoint_dir, "latest_checkpoint.pth")
    
    if os.path.exists(checkpoint_path):
        print(f"モデルを {checkpoint_path} から復元します")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
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

        # モデルの保存
        # --- 修正開始 --- 
        # モデルを保存する際に、エポック番号付きの名前をつけて保存
        model_filename = os.path.join(checkpoint_dir, f"model_epoch_{epoch + 1}.pt")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss.item(),
        }, model_filename)
        
        # 最新のチェックポイントとして保存
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss.item(),
        }, checkpoint_path)
        # --- 修正終了 ---

    # 終了時間をログ
    end_time = datetime.now()
    duration = end_time - start_time
    print(f"トレーニングが完了しました。終了時間: {end_time} | トレーニングにかかった時間: {duration}")
    wandb.log({"training_duration": str(duration), "end_time": str(end_time)})
    wandb.finish()
