import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import imageio

# npzファイルのパス
npz_file_path = '/home/yamada_24/cwvae/logs/minerl/minerl_cwvae_rssmcell_3l_f6_decsd0.4_enchl3_ences800_edchnlmult1_ss100_ds800_es800_seq100_lr0.0001_bs50/eval_2024_10_15_13_05_07/sample0_gt/gt_ctx.npz'

output_dir = '/home/yamada_24/cwvae/output_videos/'  # 保存先のディレクトリ
output_video_name = 'output_video.mp4'  # 保存する動画のファイル名
output_video_path = os.path.join(output_dir, output_video_name)  # フルパスで指定

# 出力フォルダが存在しない場合は作成する
os.makedirs(output_dir, exist_ok=True)

def load_npz(file_path):
    """
    npzファイルを読み込み、含まれる画像データを返す。
    """
    data = np.load(file_path)
    images = None

    # npzファイルに含まれる配列のキーを表示
    print("Keys in the npz file:", data.files)

    for key in data.files:
        images = data[key]
        print(f"\nData for {key}:")
        print(f"Shape: {images.shape}, Dtype: {images.dtype}")
        break  # 最初のキーの画像を使用

    return images

def show_video(images, save_path=None):
    """
    画像を動画として表示および保存する。
    """
    fig, ax = plt.subplots()
    
    # 画像の初期化
    if images.shape[1] == 3:  # カラー画像
        img = ax.imshow(images[0].transpose(1, 2, 0))
    elif images.shape[1] == 1:  # グレースケール画像
        img = ax.imshow(images[0, 0], cmap='gray')

    def update(frame):
        """フレームを更新するための関数"""
        if images.shape[1] == 3:  # カラー画像
            img.set_data(images[frame].transpose(1, 2, 0))
        elif images.shape[1] == 1:  # グレースケール画像
            img.set_data(images[frame, 0])
        return [img]

    # アニメーション作成
    ani = FuncAnimation(fig, update, frames=len(images), blit=True)

    # 動画として保存
    if save_path:
        print(f"Saving video to {save_path}")
        ani.save(save_path, writer='ffmpeg', fps=10)
    
    plt.show()

if __name__ == "__main__":
    # npzファイルをロード
    images = load_npz(npz_file_path)
    
    # 動画を表示および保存
    show_video(images, output_video_path)
