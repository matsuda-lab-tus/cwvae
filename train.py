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
wandb.init(project="CW-VAE", config={"learning_rate": 0.001, "epochs": 600})

def train_setup(cfg, model):
    """
    トレーニングのセットアップ。
    モデルをGPU/CPUに送り、オプティマイザを設定する。
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Optimizerを設定
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, eps=1e-04)

    # deviceをcfgに追加
    cfg.device = device

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
    cfg.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 実験のルートディレクトリを作成
    exp_rootdir = os.path.join(cfg.logdir, cfg.dataset, tools.exp_name(cfg))
    os.makedirs(exp_rootdir, exist_ok=True)

    # 設定を保存
    print(cfg)
    with open(os.path.join(exp_rootdir, "config.yml"), "w") as f:
        yaml.dump(dict(cfg), f, default_flow_style=False)

    # データセットをロード
    train_loader, val_loader = load_dataset(cfg)

    # モデルを構築
    model_components = build_model(cfg)
    model = model_components["meta"]["model"]
    encoder = model_components["training"]["encoder"]
    decoder = model_components["training"]["decoder"]

    # トレーニングのセットアップ
    optimizer, device = train_setup(cfg, model)

    # チェックポイントを定義 (一から学習するように修正)
    checkpoint = Checkpoint(exp_rootdir)  # ここで定義

    # モデルのパラメータを初期化
    model.apply(model.init_weights)

    # トレーニングループ
    print("トレーニングを開始します。")
    start_time = datetime.now()  # トレーニング開始時間
    step = 0

    while step < cfg.max_steps:
        try:
            model.train()
            for batch_idx, train_batch in enumerate(train_loader):  # DataLoaderからバッチを取得
                print(f"Processing training batch {batch_idx + 1}/{len(train_loader)}")
                train_batch = train_batch.to(device)  # デバイスに送る（GPU or CPU）

                # train_batch のシーケンス長を cfg.seq_len に揃える
                if train_batch.shape[1] > cfg.seq_len:
                    train_batch = train_batch[:, :cfg.seq_len]

                # train_batch の次元チェック
                print(f"train_batch shape after slicing: {train_batch.shape}")
                        
                optimizer.zero_grad()
                        
                # エンコーダーを通して特徴量を抽出
                obs_encoded = encoder(train_batch)
                print(f"[DEBUG] Encoder output type: {type(obs_encoded)}")
                if isinstance(obs_encoded, (list, tuple)):
                    for i, enc in enumerate(obs_encoded):
                        print(f"[DEBUG] Encoder output at level {i}: {enc.shape}")
                else:
                    print(f"[DEBUG] Encoder output: {obs_encoded.shape}")

                # モデルの出力を取得
                outputs_bot, _, priors, posteriors = model.hierarchical_unroll(obs_encoded)
                print(f"[DEBUG] outputs_bot shape: {outputs_bot.shape}")
                print(f"[DEBUG] priors length: {len(priors)}")
                print(f"[DEBUG] posteriors length: {len(posteriors)}")

                # obs_decoded を生成
                obs_decoded = decoder(outputs_bot)
                print(f"obs_decoded shape: {obs_decoded.shape}")
                        
                # 損失の計算
                losses = model.compute_losses(
                    obs=train_batch,
                    obs_decoded=obs_decoded,
                    priors=priors,
                    posteriors=posteriors,
                    dec_stddev=cfg.dec_stddev,
                    free_nats=cfg.free_nats,
                    beta=cfg.beta
                )
                loss = losses["loss"]
                print(f"Loss calculated: {loss.item()}")

                # wandbに損失をログ
                wandb.log({"train_loss": loss.item(), "step": step})

                loss.backward()
                        
                # 勾配クリッピングの実施（必要な場合）
                if cfg.clip_grad_norm_by is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_grad_norm_by)
                        
                optimizer.step()

                # 検証損失の計算
                if step % cfg.save_scalars_every == 0:
                    model.eval()
                    with torch.no_grad():
                        val_losses = []
                        for val_batch in val_loader:
                            val_batch = val_batch.to(device)
                            if val_batch.shape[1] > cfg.seq_len:
                                val_batch = val_batch[:, :cfg.seq_len]
                            val_obs_encoded = encoder(val_batch)
                            print(f"[DEBUG] Validation encoder output type: {type(val_obs_encoded)}")
                            if isinstance(val_obs_encoded, (list, tuple)):
                                for i, enc in enumerate(val_obs_encoded):
                                    print(f"[DEBUG] Validation encoder output at level {i}: {enc.shape}")
                            else:
                                print(f"[DEBUG] Validation encoder output: {val_obs_encoded.shape}")

                            val_outputs_bot, _, val_priors, val_posteriors = model.hierarchical_unroll(val_obs_encoded)
                            val_obs_decoded = decoder(val_outputs_bot)
                            print(f"Validation obs_decoded shape: {val_obs_decoded.shape}")
                            val_losses_dict = model.compute_losses(
                                obs=val_batch,
                                obs_decoded=val_obs_decoded,
                                priors=val_priors,
                                posteriors=val_posteriors,
                                dec_stddev=cfg.dec_stddev,
                                free_nats=cfg.free_nats,
                                beta=cfg.beta
                            )
                            val_loss = val_losses_dict["loss"].item()
                            val_losses.append(val_loss)
                            print(f"Validation Loss for current batch: {val_loss}")
                        average_val_loss = sum(val_losses) / len(val_losses)
                        wandb.log({"val_loss": average_val_loss, "step": step})
                        model.train()

                # モデルの保存
                if step % cfg.save_model_every == 0:
                    checkpoint.save(model, optimizer, step)

                if cfg.save_named_model_every and step % cfg.save_named_model_every == 0:
                    checkpoint.save(model, optimizer, step, f"model_{step}")

                step += 1

                if step >= cfg.max_steps:
                    print("Reached max_steps. Stopping training.")
                    break

        except Exception as e:
            print(f"トレーニングが {str(e)} により中断されました")
            wandb.log({"error": str(e)})
            break

    # 終了時間をログ
    end_time = datetime.now()
    duration = end_time - start_time
    print(f"トレーニングが完了しました。終了時間: {end_time} | トレーニングにかかった時間: {duration}")
    wandb.log({"training_duration": str(duration), "end_time": str(end_time)})
    wandb.finish()
