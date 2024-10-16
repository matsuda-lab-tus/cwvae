import argparse
import pathlib
import os
from datetime import datetime
import numpy as np
from cwvae import build_model
from data_loader import VideoDataset  # 新しいデータローダーを使用
import tools
import torch
import matplotlib.pyplot as plt

def load_new_dataset(datadir, batch_size, eval_seq_len):
    # 新しいデータセットをロードする関数
    print(f"[DEBUG] Loading dataset from {datadir}")
    dataset = VideoDataset(datadir)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return data_loader

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--logdir",
        default="logs/minerl/minerl_cwvae_rssmcell_3l_f6_decsd0.1_enchl3_ences1000_edchnlmult1_ss100_ds800_es800_seq100_lr0.0001_bs50/checkpoints",
        type=str,
        help="モデルのチェックポイントが保存されているディレクトリのパス（configは親ディレクトリに存在）",
    )
    parser.add_argument(
        "--num-examples", default=10, type=int, help="評価するサンプル数"
    )
    parser.add_argument(
        "--eval-seq-len", default=100, type=int, help="評価シーケンスの総長"
    )
    parser.add_argument("--datadir", default="./minerl_navigate/", type=str, help="新しいデータセットのパス")
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
    print(f"[DEBUG] Log directory exists: {args.logdir}")

    # ディレクトリの設定
    exp_rootdir = str(pathlib.Path(args.logdir).resolve().parent)
    eval_logdir = os.path.join(
        exp_rootdir, "eval_{}".format(datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))
    )
    os.makedirs(eval_logdir, exist_ok=True)
    print(f"[DEBUG] Evaluation log directory created at: {eval_logdir}")

    # 設定ファイルの読み込み
    cfg = tools.read_configs(os.path.join(exp_rootdir, "config.yml"))
    cfg.batch_size = 1
    cfg.open_loop_ctx = args.open_loop_ctx
    if args.eval_seq_len is not None:
        cfg.eval_seq_len = args.eval_seq_len
    if args.datadir:
        cfg.datadir = args.datadir
    print(f"[DEBUG] Configuration loaded: {cfg}")

    # 新しいデータセットをロード
    val_loader = load_new_dataset(args.datadir, cfg.batch_size, cfg.eval_seq_len)

    # デバイスの設定（GPUが使えるか確認）
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[DEBUG] Device selected: {device}")

    # モデルの構築
    model_components = build_model(cfg)
    model = model_components["meta"]["model"]
    encoder = model_components["training"]["encoder"]
    decoder = model_components["training"]["decoder"]
    model.to(device)
    print(f"[DEBUG] Model built and moved to device: {device}")

    # チェックポイントの定義と復元
    checkpoint_path = os.path.join(args.logdir, "latest_checkpoint.pth")
    print(f"モデルを {checkpoint_path} から復元します")

    # オプティマイザを設定（チェックポイントの復元に必要）
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, eps=1e-04)

    # 最新のチェックポイント（latest_checkpoint.pth）を使用してモデルとオプティマイザを復元
    if os.path.exists(checkpoint_path):
        print(f"最新のチェックポイント: {checkpoint_path} を読み込みます")
        checkpoint = torch.load(checkpoint_path)  # torch.loadを使って直接読み込み
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print("[DEBUG] Model and optimizer state restored from checkpoint")
    else:
        raise FileNotFoundError(f"チェックポイントが見つかりません: {checkpoint_path}")

    # 評価の実施
    ssim_all = []
    psnr_all = []

    model.eval()  # モデルを評価モードに設定
    with torch.no_grad():
        for i_ex, gts_tensor in enumerate(val_loader):
            if i_ex >= args.num_examples:
                break
            print(f"[DEBUG] Evaluating sample {i_ex + 1}/{args.num_examples}")
            try:
                # データをデバイスに送る
                print(f"[DEBUG] Ground truth tensor shape: {gts_tensor.shape}")
                gts_tensor = gts_tensor.to(device)

                # シーケンス長を評価用に調整（パディングとカット）
                if gts_tensor.shape[1] < cfg.eval_seq_len:
                    padding = cfg.eval_seq_len - gts_tensor.shape[1]
                    print(f"[DEBUG] Padding ground truth tensor by {padding} frames")
                    gts_tensor = torch.nn.functional.pad(gts_tensor, (0, 0, 0, 0, 0, padding))
                elif gts_tensor.shape[1] > cfg.eval_seq_len:
                    print(f"[DEBUG] Cutting ground truth tensor to {cfg.eval_seq_len} frames")
                    gts_tensor = gts_tensor[:, :cfg.eval_seq_len]

                # コンテキストフレームと未来のフレームに分割
                context_frames = gts_tensor[:, :args.open_loop_ctx]
                future_frames_gt = gts_tensor[:, args.open_loop_ctx:]
                print(f"[DEBUG] Context frames shape: {context_frames.shape}, Future frames shape: {future_frames_gt.shape}")

                # コンテキストフレームをエンコード
                obs_encoded_context = encoder(context_frames)
                print(f"[DEBUG] Encoded context frames")

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
                    print(f"[DEBUG] Level {level_idx}: future length = {future_length_level}")

                    # コンテキストフレームと将来のフレームを連結
                    obs_full_level = torch.cat([obs_context_level, obs_future_level], dim=1)
                    obs_encoded_full.append(obs_full_level)

                # モデルの予測を生成
                outputs_bot, _, priors, posteriors = model.hierarchical_unroll(obs_encoded_full)
                print(f"[DEBUG] Model unroll completed")

                outputs_bot_future = outputs_bot[:, args.open_loop_ctx:]
                preds = decoder(outputs_bot_future)
                print(f"[DEBUG] Predictions generated")

                # GTとPredの形状を統一
                future_frames_gt_np = np.squeeze(future_frames_gt.cpu().numpy(), axis=0)
                preds_np = np.squeeze(preds.cpu().numpy(), axis=0)
                print(f"[DEBUG] Ground truth shape: {future_frames_gt_np.shape}, Predictions shape: {preds_np.shape}")

                # メトリクスの計算
                ssim, psnr = tools.compute_metrics(future_frames_gt_np, preds_np)
                ssim_all.append(ssim)
                psnr_all.append(psnr)
                print(f"[DEBUG] SSIM: {ssim}, PSNR: {psnr}")

                gts_np = np.uint8(np.clip(gts_tensor.cpu().numpy(), 0, 1) * 255)
                preds_np_vis = np.uint8(np.clip(preds_np, 0, 1) * 255)

                # グラウンドトゥルース（GT）と予測結果を保存
                path = os.path.join(eval_logdir, f"sample{i_ex}_gt/")
                os.makedirs(path, exist_ok=True)
                print(f"[DEBUG] Saving ground truth and predictions for sample {i_ex}")

                # GT (Ground Truth) の保存
                print(f"[DEBUG] gts_np shape: {gts_np.shape}, expected shape: (1, seq_len, C, H, W)")
                np.savez(os.path.join(path, "gt_ctx.npz"), gts_np[0, : args.open_loop_ctx])
                np.savez(os.path.join(path, "gt_pred.npz"), gts_np[0, args.open_loop_ctx:])
                print(f"[DEBUG] Ground truth saved: {path}")

                # 予測画像のグリッド保存
                if not args.no_save_grid:
                    try:
                        print(f"[DEBUG] Saving grid of context frames to: {os.path.join(path, 'gt_ctx.png')}")
                        tools.save_as_grid(gts_np[0, : args.open_loop_ctx], path, "gt_ctx.png")
                        print(f"[DEBUG] Saving grid of predicted frames to: {os.path.join(path, 'gt_pred.png')}")
                        tools.save_as_grid(gts_np[0, args.open_loop_ctx:], path, "gt_pred.png")
                        print("[DEBUG] GT images saved successfully")
                    except Exception as e:
                        print(f"[ERROR] Failed to save GT images: {str(e)}")
                        import traceback
                        traceback.print_exc()

                # 予測結果を保存
                path = os.path.join(eval_logdir, f"sample{i_ex}/")
                os.makedirs(path, exist_ok=True)
                print(f"[DEBUG] preds_np_vis shape: {preds_np_vis.shape}, expected shape: (1, seq_len, C, H, W)")
                np.savez(os.path.join(path, "predictions.npz"), preds_np)
                print(f"[DEBUG] Predictions saved as NPZ at: {os.path.join(path, 'predictions.npz')}")

                # 予測結果のグリッド保存
                if not args.no_save_grid:
                    try:
                        print(f"[DEBUG] Saving grid of predicted frames to: {os.path.join(path, 'predictions.png')}")
                        tools.save_as_grid(preds_np_vis[0], path, "predictions.png")
                        print("[DEBUG] Prediction images saved successfully")
                    except Exception as e:
                        print(f"[ERROR] Failed to save prediction images: {str(e)}")
                        import traceback
                        traceback.print_exc()


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
