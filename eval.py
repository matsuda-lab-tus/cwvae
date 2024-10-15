import argparse
import pathlib
import os
from datetime import datetime
import numpy as np
from cwvae import build_model
from data_loader2 import VideoDataset2  # 新しいデータローダーを使用
import tools
from loggers.checkpoint import Checkpoint
import torch
import matplotlib.pyplot as plt

def load_new_dataset(datadir, batch_size, eval_seq_len):
    # 新しいデータセットをロードする関数
    dataset = VideoDataset2(datadir)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return data_loader

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--logdir",
        default="logs/minerl/minerl_cwvae_rssmcell_3l_f6_decsd0.4_enchl3_ences800_edchnlmult1_ss100_ds800_es800_seq100_lr0.0001_bs50/model_0_20241015_151917",
        type=str,
        help="モデルのチェックポイントが保存されているディレクトリのパス（configは親ディレクトリに存在）",
    )
    parser.add_argument(
        "--num-examples", default=10, type=int, help="評価するサンプル数"
    )
    parser.add_argument(
        "--eval-seq-len", default=100, type=int, help="評価シーケンスの総長"
    )
    parser.add_argument("--datadir", default="./output2/test/intended", type=str, help="新しいデータセットのパス")
    parser.add_argument(
        "--num-samples", default=1, type=int, help="各サンプルに対して生成する予測サンプルの数"
    )
    parser.add_argument(
        "--open-loop-ctx", default=36, type=int, help="コンテキストフレームの数"
    )
    parser.add_argument(
        "--no-save-grid",
        action="store_true",
        default=False,
        help="画像グリッドの保存を防止するためのフラグ",
    )

    args = parser.parse_args()

    assert os.path.exists(args.logdir), f"ログディレクトリが存在しません: {args.logdir}"

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

    # 新しいデータセットをロード
    val_loader = load_new_dataset(args.datadir, cfg.batch_size, cfg.eval_seq_len)

    # デバイスの設定（GPUが使えるか確認）
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # モデルの構築
    model_components = build_model(cfg)
    model = model_components["meta"]["model"]
    encoder = model_components["training"]["encoder"]
    decoder = model_components["training"]["decoder"]
    model.to(device)

    # チェックポイントの定義と復元
    checkpoint = Checkpoint(exp_rootdir)
    print(f"モデルを {args.logdir} から復元します")

    # オプティマイザを設定（チェックポイントの復元に必要）
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, eps=1e-04)

    # モデルとオプティマイザを復元
    checkpoint.restore(model, optimizer)

    # 評価の実施
    ssim_all = []
    psnr_all = []

    model.eval()  # モデルを評価モードに設定
    with torch.no_grad():
        for i_ex, gts_tensor in enumerate(val_loader):
            if i_ex >= args.num_examples:
                break
            try:
                # データをデバイスに送る
                gts_tensor = gts_tensor.to(device)

                # シーケンス長を評価用に調整（パディングとカット）
                if gts_tensor.shape[1] < cfg.eval_seq_len:
                    padding = cfg.eval_seq_len - gts_tensor.shape[1]
                    gts_tensor = torch.nn.functional.pad(gts_tensor, (0, 0, 0, 0, 0, padding))
                elif gts_tensor.shape[1] > cfg.eval_seq_len:
                    gts_tensor = gts_tensor[:, :cfg.eval_seq_len]

                # コンテキストフレームと未来のフレームに分割
                context_frames = gts_tensor[:, :args.open_loop_ctx]
                future_frames_gt = gts_tensor[:, args.open_loop_ctx:]

                # コンテキストフレームをエンコード
                obs_encoded_context = encoder(context_frames)

                # 各レベルごとに処理
                obs_encoded_full = []
                future_length = future_frames_gt.shape[1]
                for level_idx, obs_context_level in enumerate(obs_encoded_context):
                    obs_encoded_dim = obs_context_level.shape[2]
                    batch_size = obs_context_level.shape[0]

                    downsample_factor = 2 ** level_idx  # レベルごとのダウンサンプリングファクター
                    future_length_level = future_length // downsample_factor

                    if future_length_level == 0:
                        future_length_level = 1  # 長さが0にならないように調整

                    obs_future_level = torch.zeros(
                        batch_size,
                        future_length_level,
                        obs_encoded_dim,
                        device=device
                    )

                    # コンテキストフレームと将来のフレームを連結
                    obs_full_level = torch.cat([obs_context_level, obs_future_level], dim=1)
                    obs_encoded_full.append(obs_full_level)

                # モデルの予測を生成
                outputs_bot, _, priors, posteriors = model.hierarchical_unroll(obs_encoded_full)

                outputs_bot_future = outputs_bot[:, args.open_loop_ctx:]

                preds = decoder(outputs_bot_future)

                # GTとPredの形状を統一
                future_frames_gt_np = np.squeeze(future_frames_gt.cpu().numpy(), axis=0)
                preds_np = np.squeeze(preds.cpu().numpy(), axis=0)

                # メトリクスの計算
                ssim, psnr = tools.compute_metrics(future_frames_gt_np, preds_np)

                ssim_all.append(ssim)
                psnr_all.append(psnr)

                gts_np = np.uint8(np.clip(gts_tensor.cpu().numpy(), 0, 1) * 255)
                preds_np_vis = np.uint8(np.clip(preds_np, 0, 1) * 255)

                # グラウンドトゥルース（GT）と予測結果を保存
                path = os.path.join(eval_logdir, f"sample{i_ex}_gt/")
                os.makedirs(path, exist_ok=True)
                np.savez(os.path.join(path, "gt_ctx.npz"), gts_np[0, : args.open_loop_ctx])
                np.savez(os.path.join(path, "gt_pred.npz"), gts_np[0, args.open_loop_ctx:])
                if not args.no_save_grid:
                    tools.save_as_grid(gts_np[0, : args.open_loop_ctx], path, "gt_ctx.png")
                    tools.save_as_grid(gts_np[0, args.open_loop_ctx:], path, "gt_pred.png")

                # 予測結果を保存
                path = os.path.join(eval_logdir, f"sample{i_ex}/")
                os.makedirs(path, exist_ok=True)
                np.savez(os.path.join(path, "predictions.npz"), preds_np)
                if not args.no_save_grid:
                    tools.save_as_grid(preds_np_vis[0], path, "predictions.png")

            except Exception as e:
                print(f"評価中のエラー: {str(e)}")
                import traceback
                traceback.print_exc()  # エラーメッセージとスタックトレースを出力してデバッグしやすく

    # メトリクスのプロット
    if ssim_all and psnr_all:
        ssim_all = np.array(ssim_all)  # 例: ssim_allの形状が (num_examples, seq_len)
        psnr_all = np.array(psnr_all)

        # ssim_allが2次元以上の場合、適切な軸で平均と標準偏差を計算
        mean_ssim = np.mean(ssim_all, axis=0)  # (seq_len,)
        std_ssim = np.std(ssim_all, axis=0)

        mean_psnr = np.mean(psnr_all, axis=0)
        std_psnr = np.std(psnr_all, axis=0)

        # x軸をシーケンス長に対応させる
        x = np.arange(len(mean_ssim))

        # SSIMのプロット
        plt.fill_between(x, mean_ssim - std_ssim, mean_ssim + std_ssim, alpha=0.2)
        plt.plot(x, mean_ssim, label="SSIM", color="blue")
        plt.xlabel("Frame")
        plt.ylabel("SSIM")
        plt.title("SSIM Over Time")
        plt.legend()
        plt.savefig(os.path.join(eval_logdir, "ssim_plot.png"))
        plt.close()

        # PSNRのプロット
        plt.fill_between(x, mean_psnr - std_psnr, mean_psnr + std_psnr, alpha=0.2)
        plt.plot(x, mean_psnr, label="PSNR", color="red")
        plt.xlabel("Frame")
        plt.ylabel("PSNR")
        plt.title("PSNR Over Time")
        plt.legend()
        plt.savefig(os.path.join(eval_logdir, "psnr_plot.png"))
        plt.close()
    else:
        print("メトリクスを計算できませんでした。評価中にエラーが発生した可能性があります。")
