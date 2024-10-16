import argparse
import pathlib
import os
from datetime import datetime
import numpy as np
from cwvae import build_model
from data_loader import VideoDataset  # 新しいデータローダーを使用
import tools
from loggers.checkpoint import Checkpoint
import torch
from data_loader import load_dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--logdir",
        default="logs/minerl/minerl_cwvae_rssmcell_3l_f6_decsd0.4_enchl3_ences800_edchnlmult1_ss100_ds800_es800_seq100_lr0.0001_bs50/model_230000",
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

    # ディレクトリの設定
    exp_rootdir = str(pathlib.Path(args.logdir).resolve().parent)
    predictions_dir = os.path.join(exp_rootdir, "predictions")
    os.makedirs(predictions_dir, exist_ok=True)

    # 設定ファイルの読み込み
    cfg = tools.read_configs(os.path.join(exp_rootdir, "config.yml"))
    cfg.batch_size = 1
    cfg.open_loop_ctx = args.open_loop_ctx
    if args.eval_seq_len is not None:
        cfg.eval_seq_len = args.eval_seq_len
    if args.datadir:
        cfg.datadir = args.datadir

    # データセットのロード（train_loader, test_loaderを返す）
    train_loader, test_loader = load_dataset(args.datadir, cfg.batch_size)

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
    model.eval()  # モデルを評価モードに設定
    with torch.no_grad():
        for i_ex, data in enumerate(test_loader):  # test_loaderを使用
            if i_ex >= args.num_examples:
                break
            try:
                # データをデバイスに送る
                if isinstance(data, (tuple, list)):
                    gts_tensor = data[0].to(device)  # データがタプルまたはリストの場合、最初の要素を使用
                elif isinstance(data, torch.Tensor):
                    gts_tensor = data.to(device)
                else:
                    print(f"Unexpected data type: {type(data)}")
                    continue  # このイテレーションをスキップ

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

                # 予測結果とグラウンドトゥルースのスケーリング調整
                preds_np = np.squeeze(preds.cpu().numpy(), axis=0)

                # スケール調整（Tanhの出力を0-1にスケーリング）
                preds_np = (preds_np + 1) / 2

                preds_np_vis = np.uint8(np.clip(preds_np, 0, 1) * 255)

                # 予測結果を保存
                sample_dir = os.path.join(predictions_dir, f"sample_{i_ex}/")
                os.makedirs(sample_dir, exist_ok=True)
                np.savez(os.path.join(sample_dir, "predictions.npz"), preds_np)
                if not args.no_save_grid:
                    tools.save_as_grid(preds_np_vis[0], sample_dir, "predictions.png")

            except Exception as e:
                print(f"評価中のエラー: {str(e)}")
                import traceback
                traceback.print_exc()  # エラーメッセージとスタックトレースを出力してデバッグ
