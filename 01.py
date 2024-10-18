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

# 予測関数: 36フレームの入力から残り64フレームを生成
def generate_predictions(cfg, model, encoder, decoder, val_loader, exp_rootdir):
    print("予測を開始します。")
    device = cfg['device']
    model.eval()
    with torch.no_grad():
        for i, val_batch in enumerate(val_loader):
            if i >= 1:  # 最初のバッチのみを使用
                break
            val_batch = val_batch.to(device)
            
            # 入力シーケンスを36フレームに制限し、残りを予測
            context_frames = val_batch[:, :36]  # [バッチサイズ, 36, チャネル数, 高さ, 幅]
            future_frames_gt = val_batch[:, 36:100]  # 残りの実際のフレーム（比較用）

            # エンコーダーを通して特徴量を抽出（コンテキスト部分）
            obs_encoded_context = encoder(context_frames)

            # 各レベルのコンテキストを拡張して、未来の予測を行うためのシーケンスに調整
            obs_encoded_full = []
            future_length = 100 - 36  # 予測するフレーム数

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
            outputs_bot_future = outputs_bot[:, 36:]  # 36フレーム以降の予測部分を取得

            # デコーダーを使って未来のフレームを画像に変換
            preds, intermediate_outputs = decoder(outputs_bot_future)
            
            # 保存ディレクトリの設定
            output_dir = os.path.join(exp_rootdir, f"predictions_sample_{i}")
            os.makedirs(output_dir, exist_ok=True)

            # 予測結果を保存
            tools.save_as_grid(preds, output_dir, "predictions.png")
            print(f"予測結果を {output_dir} に保存しました。")

            # ground truth も保存して比較用にする
            tools.save_as_grid(future_frames_gt, output_dir, "ground_truth.png")
            print(f"実際の未来のフレームを {output_dir} に保存しました。")

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

    # モデルのトレーニング（エポック数は50）
    train(cfg, train_loader, val_loader, model, encoder, decoder, optimizer, exp_rootdir, 50)

    # 36フレームの入力から残りの64フレームを予測し、結果を保存
    generate_predictions(cfg, model, encoder, decoder, val_loader, exp_rootdir)
