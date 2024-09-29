import os
import torch
import argparse
import yaml
from torch.utils.data import DataLoader
from cwvae import build_model
from loggers.summary import Summary
from loggers.checkpoint import Checkpoint
from data_loader import load_dataset
import tools

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
    cfg.device = device  # 修正: キーアクセスではなく属性アクセスを使用

    return optimizer, device

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", default="./logs", type=str, help="ログディレクトリのパス")
    parser.add_argument("--datadir", default="./minerl_navigate/train", type=str, help="データディレクトリのパス")
    parser.add_argument("--config", default="./configs/minerl.yml", type=str, help="設定ファイル（YAML）のパス")
    parser.add_argument("--base-config", default="./configs/base_config.yml", type=str, help="ベース設定ファイルのパス")
    
    args = parser.parse_args()
    cfg = tools.read_configs(args.config, args.base_config, datadir=args.datadir, logdir=args.logdir)

    # デバイスの設定を追加 (ここが新しい部分)
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

    # トレーニングのセットアップ
    optimizer, device = train_setup(cfg, model)

    # サマリーを定義
    summary = Summary(exp_rootdir, save_gifs=cfg.save_gifs)
    summary.build_summary(cfg, model_components)

    # チェックポイントを定義
    checkpoint = Checkpoint(exp_rootdir)

    # モデルを復元（存在する場合）
    start_step = 0
    if os.path.exists(checkpoint.log_dir_model):
        print(f"モデルを {checkpoint.log_dir_model} から復元します")
        checkpoint.restore(model, optimizer)
        start_step = checkpoint.get_last_step()
        print(f"トレーニングをステップ {start_step} から再開します")
    else:
        # モデルのパラメータを初期化
        model.apply(model.init_weights)

    # トレーニングループ
    print("トレーニングを開始します。")

    step = start_step
    while step < cfg.max_steps:
        try:
            model.train()
            for batch_idx, train_batch in enumerate(train_loader):  # DataLoaderからバッチを取得
                print(f"Processing training batch {batch_idx + 1}/{len(train_loader)}")
                train_batch = train_batch.to(device)  # デバイスに送る（GPU or CPU）
                
                optimizer.zero_grad()
                
                # モデルの出力を取得
                outputs = model.hierarchical_unroll(train_batch)
                priors = outputs[2]  # priors を取得
                posteriors = outputs[3]  # posteriors を取得
                
                # `priors` や `posteriors` が Tensor オブジェクトであることを確認
                if isinstance(priors, list):
                    priors = priors[0]["mean"]  # 最初のレベルの mean 値を取得
                if isinstance(posteriors, list):
                    posteriors = posteriors[0]["mean"]  # 最初のレベルの mean 値を取得
                
                # obs_decoded を生成
                obs_decoded = model.decoder(priors)  # デコード
                print(f"obs_decoded shape: {obs_decoded.shape}")
                
                # 損失の計算
                loss = model.compute_losses(
                    obs=train_batch,
                    obs_decoded=obs_decoded,
                    priors=outputs[2],
                    posteriors=outputs[3],
                    dec_stddev=cfg.dec_stddev,
                    free_nats=cfg.free_nats,
                    beta=cfg.beta
                )["loss"]
                print(f"Loss calculated: {loss.item()}")
                
                loss.backward()
                
                # 勾配クリッピングの実施（必要な場合）
                if cfg.clip_grad_norm_by is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_grad_norm_by)
                
                optimizer.step()

                # サマリーやモデルの保存
                if step % cfg.save_scalars_every == 0:
                    train_loss = loss.item()
                    summary.save_scalar("loss", train_loss, step)

                    # 検証損失の計算
                    model.eval()
                    with torch.no_grad():
                        val_losses = []
                        for val_batch in val_loader:
                            val_batch = val_batch.to(device)
                            val_outputs = model.hierarchical_unroll(val_batch)
                            val_priors = val_outputs[2]
                            val_posteriors = val_outputs[3]
                            val_obs_decoded = model.decoder(val_priors[0]["mean"])  # デコード
                            val_loss = model.compute_losses(
                                obs=val_batch,
                                obs_decoded=val_obs_decoded,
                                priors=val_priors,
                                posteriors=val_posteriors,
                                dec_stddev=cfg.dec_stddev,
                                free_nats=cfg.free_nats,
                                beta=cfg.beta
                            )["loss"].item()
                            val_losses.append(val_loss)
                            print(f"Validation Loss for current batch: {val_loss}")
                        average_val_loss = sum(val_losses) / len(val_losses)
                        summary.save_scalar("val_loss", average_val_loss, step)
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
            break

    print("トレーニングが完了しました。")
