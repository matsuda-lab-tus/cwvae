import argparse  # コマンドライン引数を扱うためのライブラリ
import pathlib  # パス操作を便利にするライブラリ
import os  # オペレーティングシステムに関する機能を提供するライブラリ
from datetime import datetime  # 日付と時刻を扱うためのライブラリ
import numpy as np  # 数値計算を行うためのライブラリ
from cwvae import build_model  # CWVAEモデルを構築するための関数をインポート
from data_loader import VideoDataset  # 動画データセットを読み込むためのクラスをインポート
import tools  # ツール関数をインポート
from loggers.checkpoint import Checkpoint  # チェックポイントを扱うためのクラスをインポート
import torch  # PyTorchライブラリをインポート
import matplotlib.pyplot as plt  # グラフを描くためのライブラリをインポート
import torchvision.utils as vutils  # 画像処理に便利な関数をインポート
from PIL import Image  # 画像を扱うためのライブラリをインポート
from data_loader import load_dataset  # データセットを読み込む関数をインポート

# チャンネルごとの正規化関数
def normalize(tensor):
    if tensor.dim() == 4:  # テンソルが4次元の場合
        # 各チャンネルの最小値と最大値を計算
        min_vals = tensor.amin(dim=(2, 3), keepdim=True)
        max_vals = tensor.amax(dim=(2, 3), keepdim=True)
    elif tensor.dim() == 3:  # テンソルが3次元の場合
        min_vals = tensor.amin(dim=(1, 2), keepdim=True)  # 各チャンネルの最小値を計算
        max_vals = tensor.amax(dim=(1, 2), keepdim=True)  # 各チャンネルの最大値を計算
    else:  # それ以外の形状の場合
        print(f"[WARN] Unsupported tensor shape {tensor.shape} for normalization. Skipping normalization.")
        return tensor  # 正規化を行わずに元のテンソルを返す
    # テンソルを正規化する
    tensor = (tensor - min_vals) / (max_vals - min_vals + 1e-8)
    return tensor  # 正規化されたテンソルを返す

# 中間出力の保存処理
def save_intermediate_outputs(intermediate_outputs, save_dir, sample_id):
    os.makedirs(save_dir, exist_ok=True)  # 保存先のディレクトリを作成
    
    for name, tensor in intermediate_outputs.items():  # 中間出力を一つずつ処理
        if tensor.dim() == 5:  # 5次元のテンソルの場合
            print(f"[WARN] {name} has 5 dimensions {tensor.shape}. Skipping image save.")
            continue  # 保存をスキップ
        elif tensor.dim() == 4 and tensor.size(1) in [1, 3]:  # 4次元でチャネル数が1または3の場合
            normalized_tensor = normalize(tensor)  # テンソルを正規化
            grid = vutils.make_grid(normalized_tensor, nrow=8, normalize=False)  # 画像グリッドを作成
        elif tensor.dim() == 3 and tensor.size(0) in [1, 3]:  # 3次元でチャネル数が1または3の場合
            normalized_tensor = normalize(tensor)  # テンソルを正規化
            grid = vutils.make_grid(normalized_tensor.unsqueeze(0), nrow=8, normalize=False)  # グリッドを作成
        else:  # それ以外の形状の場合
            print(f"[WARN] {name} has unsupported shape {tensor.shape}. Saving histogram instead.")
            plt.figure()  # 新しいグラフを作成
            tensor_cpu = tensor.cpu().numpy().flatten()  # テンソルをNumPy配列に変換してフラット化
            plt.hist(tensor_cpu, bins=50, color='blue', alpha=0.7)  # ヒストグラムを描画
            plt.title(f"{name} Histogram")  # グラフのタイトル
            plt.xlabel("Value")  # X軸のラベル
            plt.ylabel("Frequency")  # Y軸のラベル
            save_path = os.path.join(save_dir, f"sample{sample_id}_{name}_histogram.png")  # 保存先のパス
            plt.savefig(save_path)  # ヒストグラムを保存
            plt.close()  # グラフを閉じる
            print(f"Saved histogram of {name} to {save_path}")  # 保存したヒストグラムの情報を表示
            continue  # 次の出力へ

        # 画像として保存
        ndarr = grid.mul(255).clamp(0, 255).byte().cpu().numpy()  # グリッドを255倍してクランプ
        if ndarr.shape[0] in [1, 3]:  # チャンネル数が1または3の場合
            ndarr = np.transpose(ndarr, (1, 2, 0))  # 形を変形
        else:
            print(f"[WARN] Unexpected channel size for {name}: {ndarr.shape[0]}. Skipping image save.")
            continue  # 保存をスキップ

        try:
            im = Image.fromarray(ndarr)  # NumPy配列から画像を作成
            save_path = os.path.join(save_dir, f"sample{sample_id}_{name}.png")  # 保存先のパス
            im.save(save_path)  # 画像を保存
            print(f"Saved {name} to {save_path}")  # 保存した画像の情報を表示
        except Exception as e:
            print(f"[ERROR] Failed to save {name} as image: {e}")  # 保存エラーを表示
            continue  # 次の出力へ

# メインプログラムの開始
if __name__ == "__main__":  # プログラムが直接実行されたとき
    parser = argparse.ArgumentParser()  # 引数を解析するためのオブジェクトを作成
    # 各引数を追加
    parser.add_argument("--logdir", default="/home/yamada_24/cwvae/logs/minerl_cwvae_20241017_145429/model", type=str, help="モデルのチェックポイントが保存されているディレクトリのパス")
    parser.add_argument("--num-examples", default=10, type=int, help="評価するサンプル数")
    parser.add_argument("--eval-seq-len", default=100, type=int, help="評価シーケンスの総長")
    parser.add_argument("--datadir", default="./minerl_navigate/", type=str, help="新しいデータセットのパス")
    parser.add_argument("--num-samples", default=64, type=int, help="各サンプルに対して生成する予測サンプルの数")
    parser.add_argument("--open-loop-ctx", default=36, type=int, help="コンテキストフレームの数")
    parser.add_argument("--no-save-grid", action="store_true", default=False, help="画像グリッドの保存を防止するためのフラグ")

    args = parser.parse_args()  # 引数を解析

    # ログディレクトリの存在を確認
    assert os.path.exists(args.logdir), f"ログディレクトリが存在しません: {args.logdir}"

    # 設定ファイルの読み込み
    exp_rootdir = str(pathlib.Path(args.logdir).resolve().parent)  # ルートディレクトリを取得
    eval_logdir = os.path.join(exp_rootdir, "eval_{}".format(datetime.now().strftime("%Y_%m_%d_%H_%M_%S")))  # 評価用のログディレクトリを作成
    os.makedirs(eval_logdir, exist_ok=True)  # 評価用ディレクトリを作成

    cfg = tools.read_configs(os.path.join(exp_rootdir, "config.yml"))  # 設定ファイルを読み込み
    cfg.batch_size = 1  # バッチサイズを1に設定
    cfg.open_loop_ctx = args.open_loop_ctx  # 引数からコンテキストフレーム数を設定

    # 評価シーケンス長とデータディレクトリの設定
    if args.eval_seq_len is not None:
        cfg.eval_seq_len = args.eval_seq_len
    if args.datadir:
        cfg.datadir = args.datadir

    train_loader, test_loader = load_dataset(args.datadir, cfg.batch_size)  # データセットを読み込む

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # GPUが使用可能かどうかを確認

    # モデルを構築する
    model_components = build_model(cfg)  # モデルの構築
    model = model_components["meta"]["model"]  # モデルを取得
    encoder = model_components["training"]["encoder"]  # エンコーダーを取得
    decoder = model_components["training"]["decoder"]  # デコーダーを取得
    model.to(device)  # モデルをデバイスに移動

    checkpoint = Checkpoint(exp_rootdir)  # チェックポイントを作成
    print(f"モデルを {args.logdir} から復元します")  # モデルの復元メッセージ
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, eps=1e-04)  # Adamオプティマイザーを作成
    checkpoint.restore(model, optimizer)  # モデルを復元

    ssim_all = []  # SSIMを保存するリスト
    psnr_all = []  # PSNRを保存するリスト

    model.eval()  # モデルを評価モードに設定
    with torch.no_grad():  # 勾配計算を無効にする
        for i_ex, data in enumerate(test_loader):  # テストデータローダーからデータを取得
            if i_ex >= args.num_examples:  # 指定されたサンプル数を超えたら終了
                break
            try:
                if isinstance(data, (tuple, list)):  # データがタプルまたはリストの場合
                    gts_tensor = data[0].to(device)  # 観測データをデバイスに移動
                elif isinstance(data, torch.Tensor):  # データがテンソルの場合
                    gts_tensor = data.to(device)  # デバイスに移動
                else:
                    print(f"Unexpected data type: {type(data)}")  # 想定外のデータタイプを表示
                    continue  # 次のデータへ

                # 評価シーケンス長に合わせてパディングまたはトリミングを行う
                if gts_tensor.shape[1] < cfg.eval_seq_len:
                    padding = cfg.eval_seq_len - gts_tensor.shape[1]  # 必要なパディングを計算
                    gts_tensor = torch.nn.functional.pad(gts_tensor, (0, 0, 0, 0, 0, padding))  # パディングを適用
                elif gts_tensor.shape[1] > cfg.eval_seq_len:
                    gts_tensor = gts_tensor[:, :cfg.eval_seq_len]  # シーケンスをトリミング

                context_frames = gts_tensor[:, :args.open_loop_ctx]  # コンテキストフレームを取得
                future_frames_gt = gts_tensor[:, args.open_loop_ctx:]  # 未来のフレームを取得

                # エンコーダーでコンテキストフレームをエンコード
                obs_encoded_context = encoder(context_frames)
                obs_encoded_full = []  # エンコードされたフルデータを格納するリスト
                future_length = future_frames_gt.shape[1]  # 未来のフレームの長さを取得
                for level_idx, obs_context_level in enumerate(obs_encoded_context):  # 各レベルごとに処理
                    obs_encoded_dim = obs_context_level.shape[2]  # エンコードされた次元を取得
                    batch_size = obs_context_level.shape[0]  # バッチサイズを取得

                    downsample_factor = 2 ** level_idx  # ダウンサンプリングの因子を計算
                    future_length_level = future_length // downsample_factor  # 各レベルの未来の長さを計算

                    if future_length_level == 0:  # 未来の長さが0の場合
                        future_length_level = 1  # 最低でも1に設定

                    # フルデータを初期化
                    obs_future_level = torch.zeros(
                        batch_size,
                        future_length_level,
                        obs_encoded_dim,
                        device=device
                    )

                    # コンテキストと未来フレームを結合
                    obs_full_level = torch.cat([obs_context_level, obs_future_level], dim=1)
                    obs_encoded_full.append(obs_full_level)  # フルデータを追加

                # AIが予測した未来をデコードする
                outputs_bot, _, priors, posteriors = model.hierarchical_unroll(obs_encoded_full)  # 予測を行う
                outputs_bot_future = outputs_bot[:, args.open_loop_ctx:]  # 未来の出力を取得
                preds, intermediate_outputs = decoder(outputs_bot_future)  # 予測と中間出力をデコード
                save_dir = os.path.join(eval_logdir, f"sample{i_ex}_intermediate_outputs")  # 保存先のディレクトリを設定

                # 予測した結果を保存する
                save_intermediate_outputs(intermediate_outputs, save_dir, sample_id=i_ex)  # 中間出力を保存
                preds_np = np.squeeze(preds.cpu().numpy(), axis=0)  # 予測結果をNumPy配列に変換
                future_frames_gt_np = np.squeeze(future_frames_gt.cpu().numpy(), axis=0)  # グラウンドトゥルースをNumPy配列に変換

                print(f"Before scaling: min={preds_np.min()}, max={preds_np.max()}")  # スケーリング前の値を表示
                # 予測結果を0-1の範囲にスケーリング
                preds_np = (preds_np - preds_np.min()) / (preds_np.max() - preds_np.min() + 1e-8)
                print(f"After scaling: min={preds_np.min()}, max={preds_np.max()}")  # スケーリング後の値を表示

                future_frames_gt_np = future_frames_gt_np / 255.0  # グラウンドトゥルースを0-1の範囲にスケーリング

                # メトリクスを計算
                try:
                    ssim, psnr = tools.compute_metrics(future_frames_gt_np, preds_np)  # SSIMとPSNRを計算
                except Exception as e:
                    print(f"メトリクス計算中のエラー: {str(e)}")  # エラーを表示
                    import traceback
                    traceback.print_exc()  # トレースバックを表示

                ssim_all.append(ssim)  # SSIMをリストに追加
                psnr_all.append(psnr)  # PSNRをリストに追加

                gts_np = np.uint8(np.clip(gts_tensor.cpu().numpy(), 0, 1) * 255)  # グラウンドトゥルースを0-255に変換
                preds_np_vis = np.uint8(np.clip(preds_np, 0, 1) * 255)  # 予測結果を0-255に変換

                # グラウンドトゥルースの保存先
                path_gt = os.path.join(eval_logdir, f"sample{i_ex}_gt/")
                os.makedirs(path_gt, exist_ok=True)  # ディレクトリを作成
                np.savez(os.path.join(path_gt, "gt_ctx.npz"), gts_np[0, : args.open_loop_ctx])  # コンテキストフレームを保存
                np.savez(os.path.join(path_gt, "gt_pred.npz"), gts_np[0, args.open_loop_ctx:])  # 予測フレームを保存
                if not args.no_save_grid:  # グリッドの保存を防止するフラグが立っていなければ
                    tools.save_as_grid(gts_np[0, : args.open_loop_ctx], path_gt, "gt_ctx.png")  # グリッドを保存
                    tools.save_as_grid(gts_np[0, args.open_loop_ctx:], path_gt, "gt_pred.png")  # グリッドを保存

                # 予測の保存先
                path_pred = os.path.join(eval_logdir, f"sample{i_ex}/")
                os.makedirs(path_pred, exist_ok=True)  # ディレクトリを作成
                np.savez(os.path.join(path_pred, "predictions.npz"), preds_np)  # 予測結果を保存
                if not args.no_save_grid:  # グリッドの保存を防止するフラグが立っていなければ
                    tools.save_as_grid(preds_np_vis[0], path_pred, "predictions.png")  # グリッドを保存

            except Exception as e:
                print(f"評価中のエラー: {str(e)}")  # エラーを表示
                import traceback
                traceback.print_exc()  # トレースバックを表示

    # AIの予測がどれだけ正確かをチェックする
    # SSIM（構造類似度指数）は、画像の構造がどれだけ似ているかを示します。
    # PSNR（ピーク信号対雑音比）は、画像の明るさがどれだけ近いかを測るものです。
    if ssim_all and psnr_all:
        ssim_all = np.array(ssim_all)  # SSIMをNumPy配列に変換
        psnr_all = np.array(psnr_all)  # PSNRをNumPy配列に変換
        mean_ssim = np.mean(ssim_all, axis=0)  # SSIMの平均を計算
        std_ssim = np.std(ssim_all, axis=0)  # SSIMの標準偏差を計算

        mean_psnr = np.mean(psnr_all, axis=0)  # PSNRの平均を計算
        std_psnr = np.std(psnr_all, axis=0)  # PSNRの標準偏差を計算

        x = np.arange(len(mean_ssim))  # X軸の値を生成

        mean_ssim = np.squeeze(mean_ssim)  # 平均を圧縮
        std_ssim = np.squeeze(std_ssim)  # 標準偏差を圧縮
        mean_psnr = np.squeeze(mean_psnr)  # 平均を圧縮
        std_psnr = np.squeeze(std_psnr)  # 標準偏差を圧縮

        # 予測の結果をグラフで可視化
        plt.fill_between(x, mean_ssim - std_ssim, mean_ssim + std_ssim, alpha=0.2)  # エリアを塗りつぶす
        plt.plot(x, mean_ssim, label="SSIM", color="blue")  # SSIMのプロット
        plt.xlabel("Frame")  # X軸のラベル
        plt.ylabel("SSIM")  # Y軸のラベル
        plt.title("SSIM Over Time")  # タイトルを設定
        plt.legend()  # 凡例を表示
        plt.savefig(os.path.join(eval_logdir, "ssim_plot.png"))  # SSIMプロットを保存
        plt.close()  # プロットを閉じる

        plt.fill_between(x, mean_psnr - std_psnr, mean_psnr + std_psnr, alpha=0.2)  # エリアを塗りつぶす
        plt.plot(x, mean_psnr, label="PSNR", color="red")  # PSNRのプロット
        plt.xlabel("Frame")  # X軸のラベル
        plt.ylabel("PSNR")  # Y軸のラベル
        plt.title("PSNR Over Time")  # タイトルを設定
        plt.legend()  # 凡例を表示
        plt.savefig(os.path.join(eval_logdir, "psnr_plot.png"))  # PSNRプロットを保存
        plt.close()  # プロットを閉じる
    else:
        print("メトリクスを計算できませんでした。評価中にエラーが発生した可能性があります。")  # エラーメッセージを表示
