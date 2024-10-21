import os  # オペレーティングシステムに関する機能を提供するライブラリ
import torch  # PyTorchライブラリをインポート
import argparse  # コマンドライン引数を扱うためのライブラリ
import numpy as np  # 数値計算を行うためのライブラリ
from datetime import datetime  # 日付と時刻を扱うためのライブラリ
from cwvae import build_model  # CW-VAEモデルを構築する関数をインポート
from data_loader import load_dataset, transform  # データセットを読み込む関数をインポート
import tools  # ツール関数をインポート
from loggers.checkpoint import Checkpoint  # モデルのチェックポイントを扱うクラスをインポート
import torchvision.utils as vutils  # 画像保存のためのユーティリティをインポート

if __name__ == "__main__":
    parser = argparse.ArgumentParser()  # コマンドライン引数を解析するためのオブジェクトを作成
    parser.add_argument(
        "--logdir",
        default="/home/yamada_24/cwvae/logs/minerl_cwvae_20241020_212029/model",
        type=str,
        help="Path to directory containing model checkpoints (with config in the parent dir)"
    )
    parser.add_argument(
        "--datadir",
        default="./minerl_navigate/",
        type=str,
        help="Path to directory containing evaluation data"
    )
    parser.add_argument(
        "--num-examples",
        default=1,
        type=int,
        help="Number of examples to evaluate"
    )
    parser.add_argument(
        "--eval-seq-len",
        default=100,
        type=int,
        help="Total length of evaluation sequences"
    )
    parser.add_argument(
        "--open-loop-ctx",
        default=36,
        type=int,
        help="Number of context frames"
    )
    parser.add_argument(
        "--no-save-grid",
        action="store_true",
        default=False,
        help="To prevent saving grids of images"
    )

    args = parser.parse_args()

    assert os.path.exists(args.logdir), f"Log directory does not exist: {args.logdir}"
    assert os.path.exists(args.datadir), f"Data directory does not exist: {args.datadir}"

    # 評価ログディレクトリの設定
    exp_rootdir = str(os.path.abspath(os.path.join(args.logdir, os.pardir)))
    eval_logdir = os.path.join(
        exp_rootdir, f"eval_{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}"
    )
    os.makedirs(eval_logdir, exist_ok=True)

    # 設定ファイルの読み込み
    cfg = tools.read_configs(os.path.join(exp_rootdir, "config.yml"))
    cfg.batch_size = 1
    cfg.open_loop_ctx = args.open_loop_ctx

    # データセットを読み込む
    _, test_loader = load_dataset(
        args.datadir,
        cfg['batch_size'],
        seq_len=cfg['eval_seq_len'],  # 評価シーケンス長を設定
        transform=transform
    )

    # モデルを構築
    model_components = build_model(cfg)
    model = model_components["meta"]["model"]
    model.to("cuda" if torch.cuda.is_available() else "cpu")

    # チェックポイントからモデルを復元
    checkpoint = Checkpoint(exp_rootdir)
    print(f"Restoring model from {args.logdir}")
    checkpoint.restore(model)

    # 評価メトリクスの初期化
    ssim_all = []
    psnr_all = []

    # 評価ループ
    for i_ex in range(args.num_examples):
        try:
            # データローダーからバッチを取得
            data = next(iter(test_loader))  # test_loaderからデータを取得
            gts = data[0].to("cuda" if torch.cuda.is_available() else "cpu")  # グラウンドトゥルースをデバイスに移動

            # エンコーダを通して特徴を抽出
            val_obs_encoded = model_components["training"]["encoder"](gts)  # 検証データをエンコード

            # 予測を行う
            outputs_bot, _, val_priors, val_posteriors = model.hierarchical_unroll(val_obs_encoded)

            # デコーダで予測画像を生成
            val_obs_decoded = model_components["training"]["decoder"](outputs_bot)[0]  # 最初の要素を取得
            
            # メトリクスの計算
            ssim, psnr = tools.compute_metrics(gts[:, args.open_loop_ctx:], val_obs_decoded)

            # 結果の保存
            path = os.path.join(eval_logdir, f"sample{i_ex}/")
            os.makedirs(path, exist_ok=True)
            np.savez(os.path.join(path, "ground_truth.npz"), gts.cpu().numpy())
            np.savez(os.path.join(path, "predictions.npz"), val_obs_decoded.detach().cpu().numpy())  # detach()を使用して計算グラフから切り離す

            # 実際の画像を保存
            input_frames = gts[:, :args.open_loop_ctx]  # 入力フレームを取得
            np.savez(os.path.join(path, "input_frames.npz"), input_frames.cpu().numpy())  # 入力フレームを保存

            if not args.no_save_grid:
                tools.save_as_grid(gts.detach().cpu().numpy(), path, "ground_truth.png")  # detach()を追加
                tools.save_as_grid(val_obs_decoded.detach().cpu().numpy(), path, "predictions.png")  # detach()を追加
                tools.save_as_grid(input_frames.detach().cpu().numpy(), path, "input_frames.png")  # detach()を追加

            # メトリクスを記録
            ssim_all.append(ssim)
            psnr_all.append(psnr)

        except Exception as e:
            print(f"Error during evaluation: {str(e)}")

    # メトリクスのプロット
    tools.plot_metrics(ssim_all, eval_logdir, "ssim")
    tools.plot_metrics(psnr_all, eval_logdir, "psnr")

    print("Evaluation completed.")
