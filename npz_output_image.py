import numpy as np
import matplotlib.pyplot as plt
import os

# npzファイルのパス
npz_file_path = '/home/yamada_24/cwvae/logs/minerl/minerl_cwvae_rssmcell_3l_f6_decsd0.4_enchl3_ences800_edchnlmult1_ss100_ds800_es800_seq100_lr0.0001_bs50/eval_2024_10_15_13_05_07/sample0_gt/gt_ctx.npz'
output_dir = 'output_images'  # 保存するディレクトリ

# 出力ディレクトリが存在しない場合、作成
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def load_and_save_npz(file_path, output_dir):
    # npzファイルを読み込む
    data = np.load(file_path)

    # npzファイルに含まれる配列のキーを表示
    print("Keys in the npz file:", data.files)

    # 各キーに対応するデータを確認
    for key in data.files:
        print(f"\nData for {key}:")
        print(f"Shape: {data[key].shape}, Dtype: {data[key].dtype}")
        print(f"Min: {np.min(data[key])}, Max: {np.max(data[key])}")
        
        # 配列が3次元または4次元の場合（画像データと仮定）
        if len(data[key].shape) == 4:
            # 画像を保存する
            save_images(data[key], key, output_dir)
        else:
            print(f"Data for {key} is not a valid image array.")

def save_images(array, key, output_dir):
    # バッチ内のすべての画像を保存
    for i in range(array.shape[0]):
        # チャンネル数による表示調整
        if array.shape[1] == 3:  # カラー画像
            image = array[i].transpose(1, 2, 0)  # (channels, height, width) -> (height, width, channels)
        elif array.shape[1] == 1:  # グレースケール画像
            image = array[i, 0]  # (バッチ, チャンネル, 高さ, 幅) -> (高さ, 幅)
        else:
            print(f"Unexpected image format: {array.shape}")
            return

        # ピクセル値が1を超えていたら正規化 (0-255で表示)
        if np.max(image) > 1:
            image = np.clip(image, 0, 255).astype(np.uint8)
        else:
            image = np.clip(image * 255, 0, 255).astype(np.uint8)

        # 画像を保存
        output_path = os.path.join(output_dir, f"{key}_image_{i}.png")
        plt.imsave(output_path, image, cmap='gray' if image.ndim == 2 else None)
        print(f"Saved {output_path}")

if __name__ == "__main__":
    # npzファイルを読み込み、画像を保存
    load_and_save_npz(npz_file_path, output_dir)
