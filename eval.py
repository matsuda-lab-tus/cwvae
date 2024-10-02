import argparse
import pathlib
import os
from datetime import datetime
import numpy as np

from cwvae import build_model
from data_loader import *
import tools
from loggers.checkpoint import Checkpoint
import torch


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--logdir",
        default="logs/minerl/minerl_cwvae_rssmcell_3l_f6_decsd0.4_enchl3_ences800_edchnlmult1_ss100_ds800_es800_seq100_lr0.0001_bs50/model_10000/",
        type=str,
        help="モデルのチェックポイントが保存されているディレクトリのパス（configは親ディレクトリに存在）",
    )
    parser.add_argument(
        "--num-examples", default=100, type=int, help="評価するサンプル数"
    )
    parser.add_argument(
        "--eval-seq-len",
        default=100,
        type=int,
        help="評価シーケンスの総長",
    )
    parser.add_argument("--datadir", default="./minerl_navigate/", type=str)
    parser.add_argument(
        "--num-samples", default=1, type=int, help="各サンプルに対して生成する予測サンプルの数"
    )
    parser.add_argument(
        "--open-loop-ctx", default=36, type=int, help="コンテキストフレームの数"
    )
    parser.add_argument(
        "--use-obs",
        default=None,
        type=str,
        help="各レベルの観測使用フラグ（例: TTF で最上位レベルの観測をスキップ）",
    )
    parser.add_argument(
        "--no-save-grid",
        action="store_true",
        default=False,
        help="画像グリッドの保存を防止するためのフラグ",
    )

    args = parser.parse_args()

    assert os.path.exists(args.logdir)

    # ディレクトリの設定
    exp_rootdir = str(pathlib.Path(args.logdir).resolve().parent)
    eval_logdir = os.path.join(
        exp_rootdir, "eval_{}".format(datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))
    )
    os.makedirs(eval_logdir, exist_ok=True)

    # 設定ファイルの読み込み
    cfg = tools.read_configs(os.path.join(exp_rootdir, "config.yml"))
    cfg.batch_size = 1
    cfg.open_loop_ctx = args.open_loop_ctx
    if args.eval_seq_len is not None:
        cfg.eval_seq_len = args.eval_seq_len
    if args.datadir:
        cfg.datadir = args.datadir
    if args.use_obs is not None:
        assert len(args.use_obs) == cfg.levels
        args.use_obs = args.use_obs.upper()
        cfg.use_obs = [dict(T=True, F=False)[c] for c in args.use_obs]
    else:
        cfg.use_obs = True
    tools.validate_config(cfg)

    # データセットのロード
    _, val_loader = load_dataset(cfg)

    # デバイスの設定（GPUが使えるか確認）
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # モデルの構築
    model_components = build_model(cfg)
    model = model_components["meta"]["model"]
    model.to(device)

    # チェックポイントの定義と復元
    checkpoint = Checkpoint(exp_rootdir)
    print(f"モデルを {args.logdir} から復元します")

    # オプティマイザを設定
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, eps=1e-04)

    # モデルとオプティマイザを復元
    checkpoint.restore(model, optimizer)

    # 評価の実施
    ssim_all = []
    psnr_all = []

    model.eval()  # モデルを評価モードに設定
    with torch.no_grad():  # 評価時は勾配計算が不要
        for i_ex, gts_tensor in enumerate(val_loader):
            if i_ex >= args.num_examples:
                break
            try:
                # 取得したデータをデバイス（GPU/CPU）に送る
                gts_tensor = gts_tensor.to(device)

                # モデルの予測を実行
                preds = model(gts_tensor)  # forwardメソッドを用いた予測

                # 予測結果をCPUに戻し、NumPy配列に変換
                preds_np = preds.cpu().numpy()

                # SSIMとPSNRのメトリクスを計算
                ssim, psnr = tools.compute_metrics(
                    gts_tensor.cpu().numpy()[:, args.open_loop_ctx:], preds_np
                )

                # 視覚化および保存のために正規化
                gts = np.uint8(np.clip(gts_tensor.cpu().numpy(), 0, 1) * 255)
                preds_np = np.uint8(np.clip(preds_np, 0, 1) * 255)

                # SSIMおよびPSNRの平均に基づいて、予測結果をソート
                order_ssim = np.argsort(np.mean(ssim, -1))
                order_psnr = np.argsort(np.mean(psnr, -1))

                # 最も良いサンプルのメトリクスを保存
                ssim_all.append(np.expand_dims(ssim[order_ssim[-1]], 0))
                psnr_all.append(np.expand_dims(psnr[order_psnr[-1]], 0))

                # グラウンドトゥルース（GT）と予測結果を保存
                path = os.path.join(eval_logdir, "sample" + str(i_ex) + "_gt/")
                os.makedirs(path, exist_ok=True)
                np.savez(path + "gt_ctx.npz", gts[0, : args.open_loop_ctx])
                np.savez(path + "gt_pred.npz", gts[0, args.open_loop_ctx:])
                if not args.no_save_grid:
                    tools.save_as_grid(gts[0, : args.open_loop_ctx], path, "gt_ctx.png")
                    tools.save_as_grid(gts[0, args.open_loop_ctx:], path, "gt_pred.png")

                # 予測結果を保存
                path = os.path.join(eval_logdir, "sample" + str(i_ex) + "/")
                os.makedirs(path, exist_ok=True)
                np.savez(path + "random_sample_1.npz", preds_np[0])
                if args.num_samples > 1:
                    np.savez(path + "best_ssim_sample.npz", preds_np[order_ssim[-1]])
                    np.savez(path + "best_psnr_sample.npz", preds_np[order_psnr[-1]])
                    np.savez(path + "random_sample_2.npz", preds_np[1])
                if not args.no_save_grid:
                    tools.save_as_grid(preds_np[0], path, "random_sample_1.png")
                    if args.num_samples > 1:
                        tools.save_as_grid(
                            preds_np[order_ssim[-1]], path, "best_ssim_sample.png"
                        )
                        tools.save_as_grid(
                            preds_np[order_psnr[-1]], path, "best_psnr_sample.png"
                        )
                        tools.save_as_grid(preds_np[1], path, "random_sample_2.png")

            except Exception as e:
                print(f"評価中のエラー: {str(e)}")
                continue  # エラーが発生しても次のサンプルに進む

    # メトリクスのプロット
    tools.plot_metrics(ssim_all, eval_logdir, "ssim")
    tools.plot_metrics(psnr_all, eval_logdir, "psnr")
