import argparse
import pathlib
import os
from datetime import datetime
import numpy as np
from cwvae import build_model
from data_loader import VideoDataset
import tools
from loggers.checkpoint import Checkpoint
import torch
import matplotlib.pyplot as plt
import torchvision.utils as vutils
from PIL import Image
from data_loader import load_dataset

# チャンネルごとの正規化関数
def normalize(tensor):
    if tensor.dim() == 4:
        min_vals = tensor.amin(dim=(2, 3), keepdim=True)
        max_vals = tensor.amax(dim=(2, 3), keepdim=True)
    elif tensor.dim() == 3:
        min_vals = tensor.amin(dim=(1, 2), keepdim=True)
        max_vals = tensor.amax(dim=(1, 2), keepdim=True)
    else:
        print(f"[WARN] Unsupported tensor shape {tensor.shape} for normalization. Skipping normalization.")
        return tensor
    tensor = (tensor - min_vals) / (max_vals - min_vals + 1e-8)
    return tensor

# 中間出力の保存処理
def save_intermediate_outputs(intermediate_outputs, save_dir, sample_id):
    os.makedirs(save_dir, exist_ok=True)
    
    for name, tensor in intermediate_outputs.items():
        if tensor.dim() == 5:
            print(f"[WARN] {name} has 5 dimensions {tensor.shape}. Skipping image save.")
            continue
        elif tensor.dim() == 4 and tensor.size(1) in [1, 3]:
            normalized_tensor = normalize(tensor)
            grid = vutils.make_grid(normalized_tensor, nrow=8, normalize=False)
        elif tensor.dim() == 3 and tensor.size(0) in [1, 3]:
            normalized_tensor = normalize(tensor)
            grid = vutils.make_grid(normalized_tensor.unsqueeze(0), nrow=8, normalize=False)
        else:
            print(f"[WARN] {name} has unsupported shape {tensor.shape}. Saving histogram instead.")
            plt.figure()
            tensor_cpu = tensor.cpu().numpy().flatten()
            plt.hist(tensor_cpu, bins=50, color='blue', alpha=0.7)
            plt.title(f"{name} Histogram")
            plt.xlabel("Value")
            plt.ylabel("Frequency")
            save_path = os.path.join(save_dir, f"sample{sample_id}_{name}_histogram.png")
            plt.savefig(save_path)
            plt.close()
            print(f"Saved histogram of {name} to {save_path}")
            continue

        # 画像として保存
        ndarr = grid.mul(255).clamp(0, 255).byte().cpu().numpy()
        if ndarr.shape[0] in [1, 3]:
            ndarr = np.transpose(ndarr, (1, 2, 0))
        else:
            print(f"[WARN] Unexpected channel size for {name}: {ndarr.shape[0]}. Skipping image save.")
            continue

        try:
            im = Image.fromarray(ndarr)
            save_path = os.path.join(save_dir, f"sample{sample_id}_{name}.png")
            im.save(save_path)
            print(f"Saved {name} to {save_path}")
        except Exception as e:
            print(f"[ERROR] Failed to save {name} as image: {e}")
            continue

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", default="/home/yamada_24/cwvae/logs/minerl_cwvae_20241017_145429/model", type=str, help="モデルのチェックポイントが保存されているディレクトリのパス")
    parser.add_argument("--num-examples", default=10, type=int, help="評価するサンプル数")
    parser.add_argument("--eval-seq-len", default=100, type=int, help="評価シーケンスの総長")
    parser.add_argument("--datadir", default="./minerl_navigate/", type=str, help="新しいデータセットのパス")
    parser.add_argument("--num-samples", default=64, type=int, help="各サンプルに対して生成する予測サンプルの数")
    parser.add_argument("--open-loop-ctx", default=36, type=int, help="コンテキストフレームの数")
    parser.add_argument("--no-save-grid", action="store_true", default=False, help="画像グリッドの保存を防止するためのフラグ")

    args = parser.parse_args()

    assert os.path.exists(args.logdir), f"ログディレクトリが存在しません: {args.logdir}"

    # 設定ファイルの読み込み
    exp_rootdir = str(pathlib.Path(args.logdir).resolve().parent)
    eval_logdir = os.path.join(exp_rootdir, "eval_{}".format(datetime.now().strftime("%Y_%m_%d_%H_%M_%S")))
    os.makedirs(eval_logdir, exist_ok=True)
    cfg = tools.read_configs(os.path.join(exp_rootdir, "config.yml"))
    cfg.batch_size = 1
    cfg.open_loop_ctx = args.open_loop_ctx

    if args.eval_seq_len is not None:
        cfg.eval_seq_len = args.eval_seq_len
    if args.datadir:
        cfg.datadir = args.datadir

    train_loader, test_loader = load_dataset(args.datadir, cfg.batch_size)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model_components = build_model(cfg)
    model = model_components["meta"]["model"]
    encoder = model_components["training"]["encoder"]
    decoder = model_components["training"]["decoder"]
    model.to(device)

    checkpoint = Checkpoint(exp_rootdir)
    print(f"モデルを {args.logdir} から復元します")
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, eps=1e-04)
    checkpoint.restore(model, optimizer)

    ssim_all = []
    psnr_all = []

    model.eval() 
    with torch.no_grad():
        for i_ex, data in enumerate(test_loader):  
            if i_ex >= args.num_examples:
                break
            try:
                if isinstance(data, (tuple, list)):
                    gts_tensor = data[0].to(device) 
                elif isinstance(data, torch.Tensor):
                    gts_tensor = data.to(device)
                else:
                    print(f"Unexpected data type: {type(data)}")
                    continue  

                if gts_tensor.shape[1] < cfg.eval_seq_len:
                    padding = cfg.eval_seq_len - gts_tensor.shape[1]
                    gts_tensor = torch.nn.functional.pad(gts_tensor, (0, 0, 0, 0, 0, padding))
                elif gts_tensor.shape[1] > cfg.eval_seq_len:
                    gts_tensor = gts_tensor[:, :cfg.eval_seq_len]

                context_frames = gts_tensor[:, :args.open_loop_ctx]
                future_frames_gt = gts_tensor[:, args.open_loop_ctx:]

                obs_encoded_context = encoder(context_frames)
                obs_encoded_full = []
                future_length = future_frames_gt.shape[1]
                for level_idx, obs_context_level in enumerate(obs_encoded_context):
                    obs_encoded_dim = obs_context_level.shape[2]
                    batch_size = obs_context_level.shape[0]

                    downsample_factor = 2 ** level_idx 
                    future_length_level = future_length // downsample_factor

                    if future_length_level == 0:
                        future_length_level = 1  

                    obs_future_level = torch.zeros(
                        batch_size,
                        future_length_level,
                        obs_encoded_dim,
                        device=device
                    )

                    obs_full_level = torch.cat([obs_context_level, obs_future_level], dim=1)
                    obs_encoded_full.append(obs_full_level)

                outputs_bot, _, priors, posteriors = model.hierarchical_unroll(obs_encoded_full)
                outputs_bot_future = outputs_bot[:, args.open_loop_ctx:]

                preds, intermediate_outputs = decoder(outputs_bot_future)
                save_dir = os.path.join(eval_logdir, f"sample{i_ex}_intermediate_outputs")
                save_intermediate_outputs(intermediate_outputs, save_dir, sample_id=i_ex)

                preds_np = np.squeeze(preds.cpu().numpy(), axis=0)
                future_frames_gt_np = np.squeeze(future_frames_gt.cpu().numpy(), axis=0)

                print(f"Before scaling: min={preds_np.min()}, max={preds_np.max()}")
                preds_np = (preds_np - preds_np.min()) / (preds_np.max() - preds_np.min() + 1e-8)
                print(f"After scaling: min={preds_np.min()}, max={preds_np.max()}")

                future_frames_gt_np = future_frames_gt_np / 255.0

                try:
                    ssim, psnr = tools.compute_metrics(future_frames_gt_np, preds_np)
                except Exception as e:
                    print(f"メトリクス計算中のエラー: {str(e)}")
                    import traceback
                    traceback.print_exc()

                ssim_all.append(ssim)
                psnr_all.append(psnr)

                gts_np = np.uint8(np.clip(gts_tensor.cpu().numpy(), 0, 1) * 255)
                preds_np_vis = np.uint8(np.clip(preds_np, 0, 1) * 255)

                path_gt = os.path.join(eval_logdir, f"sample{i_ex}_gt/")
                os.makedirs(path_gt, exist_ok=True)
                np.savez(os.path.join(path_gt, "gt_ctx.npz"), gts_np[0, : args.open_loop_ctx])
                np.savez(os.path.join(path_gt, "gt_pred.npz"), gts_np[0, args.open_loop_ctx:])
                if not args.no_save_grid:
                    tools.save_as_grid(gts_np[0, : args.open_loop_ctx], path_gt, "gt_ctx.png")
                    tools.save_as_grid(gts_np[0, args.open_loop_ctx:], path_gt, "gt_pred.png")

                path_pred = os.path.join(eval_logdir, f"sample{i_ex}/")
                os.makedirs(path_pred, exist_ok=True)
                np.savez(os.path.join(path_pred, "predictions.npz"), preds_np)
                if not args.no_save_grid:
                    tools.save_as_grid(preds_np_vis[0], path_pred, "predictions.png")

            except Exception as e:
                print(f"評価中のエラー: {str(e)}")
                import traceback
                traceback.print_exc()  

    if ssim_all and psnr_all:
        ssim_all = np.array(ssim_all)
        psnr_all = np.array(psnr_all)
        mean_ssim = np.mean(ssim_all, axis=0)  
        std_ssim = np.std(ssim_all, axis=0)

        mean_psnr = np.mean(psnr_all, axis=0)
        std_psnr = np.std(psnr_all, axis=0)

        x = np.arange(len(mean_ssim))

        mean_ssim = np.squeeze(mean_ssim)
        std_ssim = np.squeeze(std_ssim)
        mean_psnr = np.squeeze(mean_psnr)
        std_psnr = np.squeeze(std_psnr)

        plt.fill_between(x, mean_ssim - std_ssim, mean_ssim + std_ssim, alpha=0.2)
        plt.plot(x, mean_ssim, label="SSIM", color="blue")
        plt.xlabel("Frame")
        plt.ylabel("SSIM")
        plt.title("SSIM Over Time")
        plt.legend()
        plt.savefig(os.path.join(eval_logdir, "ssim_plot.png"))
        plt.close()

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
