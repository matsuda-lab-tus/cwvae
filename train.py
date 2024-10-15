import os
import torch
import argparse
import yaml
from torch.utils.data import DataLoader
from cwvae import build_model
from loggers.checkpoint import Checkpoint
from data_loader import load_dataset
import tools
import wandb
from datetime import datetime  # 終了時間の取得に使用

# wandbの初期化
wandb.init(project="CW-VAE", config={"learning_rate": 0.0001, "epochs": 300})

def train_setup(cfg, model):
    """
    トレーニングのセットアップ。
    モデルをGPU/CPUに送り、オプティマイザを設定する。
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Optimizerを設定
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['lr'], eps=1e-04)

    # deviceをcfgに追加
    cfg['device'] = device

    return optimizer, device

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", default="./logs", type=str, help="ログディレクトリのパス")
    parser.add_argument("--datadir", default="./minerl_navigate/", type=str, help="データディレクトリのパス")
    parser.add_argument("--config", default="./configs/minerl.yml", type=str, help="設定ファイル（YAML）のパス")
    parser.add_argument("--base-config", default="./configs/base_config.yml", type=str, help="ベース設定ファイルのパス")
    
    args = parser.parse_args()
    cfg = tools.read_configs(args.config, args.base_config, datadir=args.datadir, logdir=args.logdir)

    # デバイスの設定を追加
    cfg['device'] = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 実験のルートディレクトリを作成
    exp_rootdir = os.path.join(cfg['logdir'], cfg['dataset'], tools.exp_name(cfg))
    os.makedirs(exp_rootdir, exist_ok=True)

    # 設定を保存
    print(cfg)
    with open(os.path.join(exp_rootdir, "config.yml"), "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)

    # データセットをロード
    train_loader, val_loader = load_dataset(cfg['datadir'], cfg['batch_size'])

    # モデルを構築
    model_components = build_model(cfg)
    model = model_components["meta"]["model"]
    encoder = model_components["training"]["encoder"]
    decoder = model_components["training"]["decoder"]

    # トレーニングのセットアップ
    optimizer, device = train_setup(cfg, model)

    # チェックポイントを定義
    checkpoint = Checkpoint(exp_rootdir)

    # モデルを復元（存在する場合）
    start_epoch = 0
    if os.path.exists(checkpoint.log_dir_model):
        print(f"モデルを {checkpoint.log_dir_model} から復元します")
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
            obs_encoded_mu, obs_encoded_logvar, obs_encoded = encoder(train_batch)

            # モデルの出力を取得
            outputs_bot, _, priors, posteriors = model.hierarchical_unroll(obs_encoded)
            
            # obs_decoded を生成
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
                val_obs_encoded_mu, val_obs_encoded_logvar = encoder(val_batch)
                val_outputs_bot, _, val_priors, val_posteriors = model.hierarchical_unroll(val_obs_encoded_mu)
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
        if (epoch + 1) % cfg['save_model_every'] == 0:
            checkpoint.save(model, optimizer, epoch)

        if cfg['save_named_model_every'] and (epoch + 1) % cfg['save_named_model_every'] == 0:
            checkpoint.save(model, optimizer, epoch, f"model_epoch_{epoch + 1}")

    # 終了時間をログ
    end_time = datetime.now()
    duration = end_time - start_time
    print(f"トレーニングが完了しました。終了時間: {end_time} | トレーニングにかかった時間: {duration}")
    wandb.log({"training_duration": str(duration), "end_time": str(end_time)})
    wandb.finish()
