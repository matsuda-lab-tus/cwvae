import os  # オペレーティングシステムに関する機能を提供するライブラリ
import torch  # PyTorchライブラリをインポート
import argparse  # コマンドライン引数を扱うためのライブラリ
import yaml  # YAML形式のファイルを読み込むためのライブラリ
from cwvae import build_model  # CW-VAEモデルを構築する関数をインポート
from data_loader import load_dataset  # データセットを読み込む関数をインポート
import tools  # ツール関数をインポート
import wandb  # 実験を管理するためのライブラリ（Weights and Biases）
from datetime import datetime  # 日付と時刻を扱うためのライブラリ
from data_loader import transform  # データセットの変換関数をインポート
import torchvision.utils as vutils  # 画像を保存するためのユーティリティをインポート
from tqdm import tqdm  # プログレスバーを表示するためのライブラリをインポート
import logging  # ログを出力するためのライブラリをインポート

# Checkpointクラスのインポート
from loggers.checkpoint import (
    Checkpoint,
)  # モデルのチェックポイントを扱うクラスをインポート

# メイン関数の定義
if __name__ == "__main__":  # スクリプトが直接実行された場合
    parser = (
        argparse.ArgumentParser()
    )  # コマンドライン引数を解析するためのオブジェクトを作成
    # 引数を追加して、各引数の説明を設定
    parser.add_argument(
        "--logdir", default="./logs", type=str, help="ログディレクトリのパス"
    )
    parser.add_argument(
        "--datadir",
        default="./minerl_navigate/",
        type=str,
        help="データディレクトリのパス",
    )
    parser.add_argument(
        "--config",
        default="./configs/minerl.yml",
        type=str,
        help="設定ファイル（YAML）のパス",
    )
    parser.add_argument(
        "--base-config",
        default="./configs/base_config.yml",
        type=str,
        help="ベース設定ファイルのパス",
    )

    args = parser.parse_args()  # 引数を解析

    # 設定ファイルの読み込み
    cfg = tools.read_configs(
        args.config, args.base_config, datadir=args.datadir, logdir=args.logdir
    )

    # wandbの初期化（実験の記録を開始）
    wandb.init(project="CW-VAE", config=cfg)

    # デバイスの設定（GPUが使えるか確認）
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cfg["device"] = device  # 設定にデバイス情報を追加

    # 保存ディレクトリの設定（実験名をわかりやすく）
    dataset_name = cfg["dataset"]  # データセット名を取得
    model_name = "cwvae"  # モデル名を設定
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")  # 現在の日時を取得
    exp_rootdir = os.path.join(
        cfg["logdir"], f"{dataset_name}_{model_name}_{current_time}"
    )  # 保存先のパスを設定
    os.makedirs(
        exp_rootdir, exist_ok=True
    )  # 保存先のディレクトリを作成（存在しない場合）

    # 設定を保存
    print(cfg)  # 設定内容を表示
    with open(os.path.join(exp_rootdir, "config.yml"), "w") as f:  # 設定ファイルを保存
        yaml.dump(cfg, f, default_flow_style=False)  # YAML形式で書き込む
    
    logging.basicConfig(
    filename=exp_rootdir+"/log.txt",          # ログを保存するファイル名
    level=logging.INFO,               # 記録するログのレベル
    format='%(asctime)s - %(message)s',  # ログのフォーマット
    datefmt='%Y-%m-%d %H:%M:%S'       # 日付のフォーマット
    )
    
    # データセットをロード
    train_loader, val_loader = load_dataset(
        cfg["datadir"], cfg["batch_size"], seq_len=cfg["seq_len"], transform=transform
    )

    # モデルの構築
    model_components = build_model(cfg)  # モデルを構築
    model = model_components["meta"]["model"]  # モデル本体を取得
    # encoder = model_components["training"]["encoder"]  # エンコーダを取得
    # decoder = model_components["training"]["decoder"]  # デコーダを取得

    # トレーニングのセットアップ
    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg["lr"], eps=1e-04
    )  # Adamオプティマイザーを設定
    model.to(device)  # モデルをデバイスに移動

    # Checkpointクラスの初期化（モデルの保存管理）
    checkpoint = Checkpoint(exp_rootdir)

    # モデルの復元（存在する場合）
    start_epoch = 0  # 開始エポックを初期化
    if checkpoint.exists():  # チェックポイントが存在するか確認
        print(
            f"モデルを {checkpoint.latest_checkpoint} から復元します"
        )  # 復元メッセージ
        start_epoch = checkpoint.restore(
            model, optimizer
        )  # モデルを復元し、開始エポックを取得
        print(
            f"トレーニングをエポック {start_epoch} から再開します"
        )  # 再開エポックを表示
    else:
        # モデルのパラメータを初期化
        model.apply(model.init_weights)  # モデルの重みを初期化

    # トレーニングループの開始
    print("トレーニングを開始します。")
    start_time = datetime.now()  # トレーニング開始時間を取得
    step = 0  # ステップを初期化
    num_epochs = cfg["num_epochs"]  # エポック数を取得

    # トレーニングループ
    for epoch in tqdm(range(start_epoch, num_epochs)):  # 各エポックに対して
        model.train()  # モデルをトレーニングモードに設定
        for batch_idx, train_batch in enumerate(
            train_loader
        ):  # トレーニングデータローダーからバッチを取得
            train_batch = train_batch.to(device)  # バッチをデバイスに移動

            # obsをtrain_batchに置き換え
            obs = train_batch  # 入力データをバッチに設定

            # 元の形状: [16, 5, 100, 3, 64, 64]
            # 必要な形状: [16 * 5, 100, 3, 64, 64]
            obs = obs.view(-1, 100, 3, 64, 64)  # バッチサイズを合わせるために形状を変換

            optimizer.zero_grad()  # 勾配をゼロにリセット
            obs_encoded = model.encoder(obs)  # 入力データをエンコーダでエンコード

            outputs_bot, _, priors, posteriors = model.hierarchical_unroll(
                obs_encoded
            )  # モデルを通して予測を行う

            # posteriors の構造を確認
            print(f"Type of posteriors: {type(posteriors)}")
            print(f"Length of posteriors: {len(posteriors)}")
            if isinstance(posteriors, list) and isinstance(posteriors[0], dict):
                det_state = posteriors[0]["det_state"]  # shape: [50, 100, 800]
                print(f"det_state type: {type(det_state)}")
                print(f"det_state shape: {det_state.size()}")  # torch.Size([50, 100, 800])

                # det_state をフラット化（batch_size * seq_len, feature_dim）
                det_state_flat = det_state.view(-1, 800)  # shape: [5000, 800]
                print(f"[DEBUG] Input to Decoder: {det_state_flat.shape}, min: {det_state_flat.min().item()}, max: {det_state_flat.max().item()}")

                # デコーダーにdet_stateを渡す
                obs_decoded = model.decoder(det_state_flat)  # タプルを返さないように修正
                print(f"[DEBUG] After fc: {obs_decoded.shape}, min: {obs_decoded.min().item()}, max: {obs_decoded.max().item()}")

                print(f"obs_decoded shape: {obs_decoded.shape}")  # 確認用プリント

            else:
                raise ValueError("posteriors の構造が想定と異なります。")


            # 損失の計算
            losses = model.compute_losses(  # 損失を計算
                obs=train_batch.view(-1, 100, 3, 64, 64),
                obs_decoded=obs_decoded,
                priors=priors,
                posteriors=posteriors,
                dec_stddev=cfg["dec_stddev"],
                kl_grad_post_perc=cfg["kl_grad_post_perc"],
                free_nats=cfg["free_nats"],
                beta=cfg["beta"],
            )
            loss = losses["loss"]  # 損失値を取得

            logging.info(f"Loss calculated: {loss.item()}")  # 損失値を表示

            # wandbに損失をログ
            wandb.log(
                {"train_loss": loss.item(), "step": step, "epoch": epoch, "nll_term": losses["nll_term"].item(), "kl_term": losses["kl_term"].item()}
            )  # 損失をwandbに記録

            loss.backward()  # 損失の勾配を計算

            if (
                cfg["clip_grad_norm_by"] is not None
            ):  # 勾配のクリッピングが設定されている場合
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), cfg["clip_grad_norm_by"]
                )  # 勾配をクリッピング

            optimizer.step()  # オプティマイザーでパラメータを更新

            step += 1  # ステップを増やす

        # エポック終了時に検証損失の計算
        model.eval()  # モデルを評価モードに設定
        with torch.no_grad():  # 勾配計算を無効にする
            val_losses = []  # 検証損失を格納するリスト
            for batch_idx, val_batch in enumerate(
                val_loader
            ):  # 検証データローダーからバッチを取得
                val_batch = val_batch.to(device)  # バッチをデバイスに移動

                # バリデーションデータをエンコーダが受け入れる形式に変換
                adjusted_val_batch = val_batch.view(-1, 100, 3, 64, 64)

                # エンコーダを通して特徴を抽出
                val_obs_encoded = model.encoder(
                    adjusted_val_batch
                )  # 検証データをエンコード

                # 予測を行う
                outputs_bot, _, val_priors, val_posteriors = model.hierarchical_unroll(
                    val_obs_encoded
                )

                # posteriorからdet_stateを取得
                if isinstance(val_posteriors, list) and isinstance(val_posteriors[0], dict):
                    val_det_state = val_posteriors[0]["det_state"]  # shape: [batch_size, seq_len, 800]
                    # det_state をフラット化（batch_size * seq_len, feature_dim）
                    val_det_state_flat = val_det_state.view(-1, val_det_state.size(-1))  # shape: [batch_size * seq_len, 800]

                    # デコーダーにval_det_state_flatを渡す
                    val_obs_decoded = model.decoder(val_det_state_flat)  # shape: [batch_size * seq_len, 3, 64, 64]
                else:
                    raise ValueError("val_posteriors の構造が想定と異なります。")

                # 予測画像を保存
                if batch_idx < 1:  # 最初の5バッチに対して
                    output_dir = os.path.join(
                        exp_rootdir, f"val_outputs_epoch_{epoch + 1}"
                    )  # 保存先のディレクトリを設定
                    os.makedirs(output_dir, exist_ok=True)  # ディレクトリを作成

                    # `val_obs_decoded` を [batch_size, seq_len, 3, 64, 64] にリシェイプ
                    seq_len = cfg["seq_len"]
                    batch_size = val_det_state.size(0)
                    val_obs_decoded_reshaped = val_obs_decoded.view(batch_size, seq_len, 3, 64, 64)

                    for seq_idx in range(batch_size):  # 各シーケンスに対して
                        # シーケンス内の画像をグリッドにまとめる
                        grid = vutils.make_grid(
                            val_obs_decoded_reshaped[seq_idx], nrow=10, normalize=True
                        )  # 例えば、10列のグリッドにまとめる

                        # 画像を保存
                        img_path = os.path.join(
                            output_dir,
                            f"val_pred_epoch{epoch + 1}_seq{batch_idx * batch_size + seq_idx}.png",
                        )  # 保存先のパスを設定
                        try:
                            vutils.save_image(grid, img_path)  # グリッド画像を保存
                        except Exception as e:
                            logging.error(f"画像の保存中にエラーが発生しました: {e}")  # エラーメッセージを表示し、次の画像の保存に進む

                # 検証損失の計算
                val_losses_dict = model.compute_losses(  # 検証損失を計算
                    obs=val_batch.view(-1, 100, 3, 64, 64),
                    obs_decoded=val_obs_decoded,
                    priors=val_priors,
                    posteriors=val_posteriors,
                    dec_stddev=cfg["dec_stddev"],
                    kl_grad_post_perc=cfg["kl_grad_post_perc"],
                    free_nats=cfg["free_nats"],
                    beta=cfg["beta"],
                )
                val_loss = val_losses_dict["loss"].item()  # 検証損失を取得
                val_losses.append(val_loss)  # 検証損失をリストに追加
                logging.info(
                    f"Validation Loss for current batch: {val_loss}"
                )  # 検証損失を表示

            average_val_loss = sum(val_losses) / len(val_losses)  # 平均検証損失を計算
            wandb.log(
                {"val_loss": average_val_loss, "epoch": epoch, }
            )  # wandbに検証損失を記録
            model.train()  # モデルをトレーニングモードに戻す

        # エポックごとにモデルを保存
        checkpoint_path = os.path.join(
            exp_rootdir, f"checkpoint_epoch_{epoch + 1}.pth"
        )  # チェックポイントのパスを設定
        torch.save(
            {  # モデルの状態を保存
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),  # モデルの状態
                "optimizer_state_dict": optimizer.state_dict(),  # オプティマイザーの状態
                "cfg": cfg,  # 設定も保存
            },
            checkpoint_path,
        )  # ファイルに保存
        logging.info(
            f"モデルをエポック {epoch + 1} で {checkpoint_path} として保存しました。"
        )  # 保存完了メッセージ

        # Checkpointクラスによる最新チェックポイントの保存
        checkpoint.save(model, optimizer, epoch)  # 最新のモデルを保存

    # 終了時間をログ
    end_time = datetime.now()  # 終了時間を取得
    duration = end_time - start_time  # トレーニングにかかった時間を計算
    logging.info(
        f"トレーニングが完了しました。終了時間: {end_time} | トレーニングにかかった時間: {duration}"
    )  # 完了メッセージを表示
    wandb.log(
        {"training_duration": str(duration), "end_time": str(end_time)}
    )  # トレーニング時間を記録
    wandb.finish()  # wandbのログを終了

