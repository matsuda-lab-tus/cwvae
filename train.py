import os  # オペレーティングシステムに関する機能を提供するライブラリ
import torch  # PyTorchライブラリをインポート
import argparse  # コマンドライン引数を扱うためのライブラリ
import yaml  # YAML形式のファイルを読み込むためのライブラリ
from torch.utils.data import DataLoader  # データローダーを使うためのモジュール
from cwvae import build_model  # CW-VAEモデルを構築する関数をインポート
from data_loader import load_dataset  # データセットを読み込む関数をインポート
import tools  # ツール関数をインポート
import wandb  # 実験を管理するためのライブラリ（Weights and Biases）
from datetime import datetime  # 日付と時刻を扱うためのライブラリ
from data_loader import transform  # データセットの変換関数をインポート
import torchvision.utils as vutils  # 画像を保存するためのユーティリティをインポート

# Checkpointクラスのインポート
from loggers.checkpoint import Checkpoint  # モデルのチェックポイントを扱うクラスをインポート

# メイン関数の定義
if __name__ == "__main__":  # スクリプトが直接実行された場合
    parser = argparse.ArgumentParser()  # コマンドライン引数を解析するためのオブジェクトを作成
    # 引数を追加して、各引数の説明を設定
    parser.add_argument("--logdir", default="./logs", type=str, help="ログディレクトリのパス")
    parser.add_argument("--datadir", default="./minerl_navigate/", type=str, help="データディレクトリのパス")
    parser.add_argument("--config", default="./configs/minerl.yml", type=str, help="設定ファイル（YAML）のパス")
    parser.add_argument("--base-config", default="./configs/base_config.yml", type=str, help="ベース設定ファイルのパス")
    
    args = parser.parse_args()  # 引数を解析

    # 設定ファイルの読み込み
    cfg = tools.read_configs(args.config, args.base_config, datadir=args.datadir, logdir=args.logdir)

    # wandbの初期化（実験の記録を開始）
    wandb.init(project="CW-VAE", config=cfg)

    # デバイスの設定（GPUが使えるか確認）
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cfg['device'] = device  # 設定にデバイス情報を追加

    # 保存ディレクトリの設定（実験名をわかりやすく）
    dataset_name = cfg['dataset']  # データセット名を取得
    model_name = "cwvae"  # モデル名を設定
    current_time = datetime.now().strftime('%Y%m%d_%H%M%S')  # 現在の日時を取得
    exp_rootdir = os.path.join(cfg['logdir'], f"{dataset_name}_{model_name}_{current_time}")  # 保存先のパスを設定
    os.makedirs(exp_rootdir, exist_ok=True)  # 保存先のディレクトリを作成（存在しない場合）

    # 設定を保存
    print(cfg)  # 設定内容を表示
    with open(os.path.join(exp_rootdir, "config.yml"), "w") as f:  # 設定ファイルを保存
        yaml.dump(cfg, f, default_flow_style=False)  # YAML形式で書き込む

    # データセットをロード
    train_loader, val_loader = load_dataset(cfg['datadir'], cfg['batch_size'], seq_len=cfg['seq_len'], transform=transform)

    # モデルの構築
    model_components = build_model(cfg)  # モデルを構築
    model = model_components["meta"]["model"]  # モデル本体を取得
    encoder = model_components["training"]["encoder"]  # エンコーダを取得
    decoder = model_components["training"]["decoder"]  # デコーダを取得

    # トレーニングのセットアップ
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['lr'], eps=1e-04)  # Adamオプティマイザーを設定
    model.to(device)  # モデルをデバイスに移動

    # Checkpointクラスの初期化（モデルの保存管理）
    checkpoint = Checkpoint(exp_rootdir)

    # モデルの復元（存在する場合）
    start_epoch = 0  # 開始エポックを初期化
    if checkpoint.exists():  # チェックポイントが存在するか確認
        print(f"モデルを {checkpoint.latest_checkpoint} から復元します")  # 復元メッセージ
        start_epoch = checkpoint.restore(model, optimizer)  # モデルを復元し、開始エポックを取得
        print(f"トレーニングをエポック {start_epoch} から再開します")  # 再開エポックを表示
    else:
        # モデルのパラメータを初期化
        model.apply(model.init_weights)  # モデルの重みを初期化

    # トレーニングループの開始
    print("トレーニングを開始します。")
    start_time = datetime.now()  # トレーニング開始時間を取得
    step = 0  # ステップを初期化
    num_epochs = cfg['num_epochs']  # エポック数を取得

    # トレーニングループ
    for epoch in range(start_epoch, num_epochs):  # 各エポックに対して
        model.train()  # モデルをトレーニングモードに設定
        for batch_idx, train_batch in enumerate(train_loader):  # トレーニングデータローダーからバッチを取得
            train_batch = train_batch.to(device)  # バッチをデバイスに移動

            # obsをtrain_batchに置き換え
            obs = train_batch  # 入力データをバッチに設定

            # 元の形状: [16, 5, 100, 3, 64, 64]
            # 必要な形状: [16 * 5, 100, 3, 64, 64]
            obs = obs.view(-1, 100, 3, 64, 64)  # バッチサイズを合わせるために形状を変換

            optimizer.zero_grad()  # 勾配をゼロにリセット
            obs_encoded = encoder(obs)  # 入力データをエンコーダでエンコード

            outputs_bot, _, priors, posteriors = model.hierarchical_unroll(obs_encoded)  # モデルを通して予測を行う
            print(model.decoder(outputs_bot))  # デコーダの出力を確認するために表示
            obs_decoded = model.decoder(outputs_bot)[0]  # デコーダで予測を生成

            # 形状を確認
            print(f"obs shape: {obs.shape}, obs_decoded shape: {obs_decoded.shape}")  # 入力と出力の形状を表示

            # 損失の計算
            losses = model.compute_losses(  # 損失を計算
                obs=train_batch.view(-1, 100, 3, 64, 64),
                obs_decoded=obs_decoded,
                priors=priors,
                posteriors=posteriors,
                dec_stddev=cfg['dec_stddev'],
                kl_grad_post_perc=cfg['kl_grad_post_perc'],
                free_nats=cfg['free_nats'],
                beta=cfg['beta']
            )
            loss = losses["loss"]  # 損失値を取得
            print(f"Loss calculated: {loss.item()}")  # 損失値を表示

            # wandbに損失をログ
            wandb.log({"train_loss": loss.item(), "step": step, "epoch": epoch})  # 損失をwandbに記録

            loss.backward()  # 損失の勾配を計算
            
            if cfg['clip_grad_norm_by'] is not None:  # 勾配のクリッピングが設定されている場合
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['clip_grad_norm_by'])  # 勾配をクリッピング
            
            optimizer.step()  # オプティマイザーでパラメータを更新

            step += 1  # ステップを増やす

        # エポック終了時に検証損失の計算
        model.eval()  # モデルを評価モードに設定
        with torch.no_grad():  # 勾配計算を無効にする
            val_losses = []  # 検証損失を格納するリスト
            for batch_idx, val_batch in enumerate(val_loader):  # 検証データローダーからバッチを取得
                val_batch = val_batch.to(device)  # バッチをデバイスに移動

                # バリデーションデータをエンコーダが受け入れる形式に変換
                adjusted_val_batch = val_batch.view(-1, 100, 3, 64, 64)

                # エンコーダを通して特徴を抽出
                val_obs_encoded = model.encoder(adjusted_val_batch)  # 検証データをエンコード

                # 予測を行う
                outputs_bot, _, val_priors, val_posteriors = model.hierarchical_unroll(val_obs_encoded)

                # デコーダで予測画像を生成
                val_obs_decoded = model.decoder(outputs_bot)[0]  # 最初の要素を取得

                # 予測画像を保存
                output_dir = os.path.join(exp_rootdir, f"val_outputs_epoch_{epoch + 1}")  # 保存先のディレクトリを設定
                os.makedirs(output_dir, exist_ok=True)  # ディレクトリを作成

                # 画像を保存するためのループ
                for i in range(val_obs_decoded.size(0)):  # 各画像に対して
                    # 画像を保存
                    img_path = os.path.join(output_dir, f"val_pred_{batch_idx * val_loader.batch_size + i}.png")  # 保存先のパスを設定
                    vutils.save_image(val_obs_decoded[i], img_path, normalize=True)  # 画像を保存

                # 検証損失の計算
                val_losses_dict = model.compute_losses(  # 検証損失を計算
                    obs=val_batch.view(-1, 100, 3, 64, 64),
                    obs_decoded=val_obs_decoded,
                    priors=val_priors,
                    posteriors=val_posteriors,
                    dec_stddev=cfg['dec_stddev'],
                    kl_grad_post_perc=cfg['kl_grad_post_perc'],
                    free_nats=cfg['free_nats'],
                    beta=cfg['beta']
                )
                val_loss = val_losses_dict["loss"].item()  # 検証損失を取得
                val_losses.append(val_loss)  # 検証損失をリストに追加
                print(f"Validation Loss for current batch: {val_loss}")  # 検証損失を表示

            average_val_loss = sum(val_losses) / len(val_losses)  # 平均検証損失を計算
            wandb.log({"val_loss": average_val_loss, "epoch": epoch})  # wandbに検証損失を記録
            model.train()  # モデルをトレーニングモードに戻す

        # エポックごとにモデルを保存
        checkpoint_path = os.path.join(exp_rootdir, f"checkpoint_epoch_{epoch + 1}.pth")  # チェックポイントのパスを設定
        torch.save({  # モデルの状態を保存
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),  # モデルの状態
            'optimizer_state_dict': optimizer.state_dict(),  # オプティマイザーの状態
            'cfg': cfg,  # 設定も保存
        }, checkpoint_path)  # ファイルに保存
        print(f"モデルをエポック {epoch + 1} で {checkpoint_path} として保存しました。")  # 保存完了メッセージ

        # Checkpointクラスによる最新チェックポイントの保存
        checkpoint.save(model, optimizer, epoch)  # 最新のモデルを保存

    # 終了時間をログ
    end_time = datetime.now()  # 終了時間を取得
    duration = end_time - start_time  # トレーニングにかかった時間を計算
    print(f"トレーニングが完了しました。終了時間: {end_time} | トレーニングにかかった時間: {duration}")  # 完了メッセージを表示
    wandb.log({"training_duration": str(duration), "end_time": str(end_time)})  # トレーニング時間を記録
    wandb.finish()  # wandbのログを終了
