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

# モデルをトレーニングする関数
def train(cfg, train_loader, val_loader, model, encoder, decoder, optimizer, exp_rootdir, num_epochs):
    print("トレーニングを開始します。")
    device = cfg['device']
    start_epoch = 0

    # チェックポイント管理を初期化（学習済みモデルがないため最初から学習開始）
    checkpoint = Checkpoint(exp_rootdir)
    
    # トレーニングループ
    for epoch in range(start_epoch, num_epochs):
        model.train()
        for batch_idx, train_batch in enumerate(train_loader):
            train_batch = train_batch.to(device)
            # シーケンス長を設定された長さに調整
            if train_batch.shape[1] > cfg['seq_len']:
                train_batch = train_batch[:, :cfg['seq_len']]

            optimizer.zero_grad()

            # エンコーダーを通して特徴量を抽出
            obs_encoded = encoder(train_batch)

            # モデルの出力を取得
            outputs_bot, _, priors, posteriors = model.hierarchical_unroll(obs_encoded)
            
            # デコーダーを通して再構成された画像を得る
            obs_decoded = decoder(outputs_bot)[0]

            # 損失を計算
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
            print(f"エポック {epoch + 1}/{num_epochs}, バッチ {batch_idx + 1}/{len(train_loader)}, 損失: {loss.item()}")

            # 損失を逆伝播
            loss.backward()
            optimizer.step()

        # エポック終了時に検証
        validate(model, encoder, decoder, val_loader, device, epoch, exp_rootdir)

        # チェックポイントを保存
        checkpoint.save(model, optimizer, epoch)
        print(f"エポック {epoch + 1} でモデルを保存しました。")

    print("トレーニングが完了しました。")

# 検証関数
def validate(model, encoder, decoder, val_loader, device, epoch, exp_rootdir):
    model.eval()
    val_losses = []
    with torch.no_grad():
        for val_batch in val_loader:
            val_batch = val_batch.to(device)
            if val_batch.shape[1] > cfg['seq_len']:
                val_batch = val_batch[:, :cfg['seq_len']]
            obs_encoded = encoder(val_batch)
            outputs_bot, _, priors, posteriors = model.hierarchical_unroll(obs_encoded)
            obs_decoded = decoder(outputs_bot)[0]
            val_losses_dict = model.compute_losses(
                obs=val_batch,
                obs_decoded=obs_decoded,
                priors=priors,
                posteriors=posteriors,
                dec_stddev=cfg['dec_stddev'],
                free_nats=cfg['free_nats'],
                beta=cfg['beta']
            )
            val_losses.append(val_losses_dict["loss"].item())
        print(f"エポック {epoch + 1} の検証損失: {sum(val_losses) / len(val_losses)}")

# メイン関数
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", default="./logs", type=str, help="ログディレクトリのパス")
    parser.add_argument("--datadir", default="./minerl_navigate/", type=str, help="データディレクトリのパス")
    parser.add_argument("--config", default="./configs/minerl.yml", type=str, help="設定ファイル（YAML）のパス")
    parser.add_argument("--base-config", default="./configs/base_config.yml", type=str, help="ベース設定ファイルのパス")
    args = parser.parse_args()

    # 設定ファイルの読み込み
    cfg = tools.read_configs(args.config, args.base_config, datadir=args.datadir, logdir=args.logdir)

    # デバイス設定
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    cfg['device'] = device

    # データセットの読み込み
    train_loader, val_loader = load_dataset(cfg['datadir'], cfg['batch_size'])

    # モデルの構築
    model_components = build_model(cfg)
    model = model_components["meta"]["model"]
    encoder = model_components["training"]["encoder"]
    decoder = model_components["training"]["decoder"]

    # オプティマイザの設定
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['lr'])

    # 実験用ディレクトリの設定
    exp_rootdir = os.path.join(cfg['logdir'], f"cwvae_experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(exp_rootdir, exist_ok=True)

    # モデルのトレーニング
    train(cfg, train_loader, val_loader, model, encoder, decoder, optimizer, exp_rootdir, cfg['num_epochs'])

    # 予測の出力
    print("予測を開始します。")
    model.eval()
    with torch.no_grad():
        for i, val_batch in enumerate(val_loader):
            if i >= 1:  # 最初のバッチのみを使用
                break
            val_batch = val_batch.to(device)
            obs_encoded = encoder(val_batch)
            outputs_bot, _, priors, posteriors = model.hierarchical_unroll(obs_encoded)
            obs_decoded = decoder(outputs_bot)[0]
            
            # 予測結果を保存
            output_dir = os.path.join(exp_rootdir, f"predictions_batch_{i}")
            os.makedirs(output_dir, exist_ok=True)
            tools.save_as_grid(obs_decoded, output_dir, "predictions.png")
            print(f"予測結果を {output_dir} に保存しました。")
