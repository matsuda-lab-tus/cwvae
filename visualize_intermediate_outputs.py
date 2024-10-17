import matplotlib.pyplot as plt
import os
from PIL import Image
import torch
import torchvision

def visualize_channel_outputs(save_dir, sample_id):
    layers = ['fc', 'reshaped', 'deconv1', 'deconv2', 'deconv3', 'deconv4', 'deconv5', 'tanh']
    
    for layer in layers:
        image_path = os.path.join(save_dir, f"sample{sample_id}_{layer}.png")
        if not os.path.exists(image_path):
            print(f"File {image_path} does not exist. Skipping...")
            continue
        image = Image.open(image_path)
        image = image.convert("RGB")
        plt.figure(figsize=(6,6))
        plt.imshow(image)
        plt.title(f"{layer} Output")
        plt.axis('off')
        plt.show()

        # 各チャネルを個別に表示
        tensor = torchvision.io.read_image(image_path).float() / 255.0  # [C, H, W]
        r, g, b = tensor[0], tensor[1], tensor[2]
        
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        
        axs[0].imshow(r, cmap='Reds')
        axs[0].set_title(f'{layer} - Red Channel')
        axs[0].axis('off')
        
        axs[1].imshow(g, cmap='Greens')
        axs[1].set_title(f'{layer} - Green Channel')
        axs[1].axis('off')
        
        axs[2].imshow(b, cmap='Blues')
        axs[2].set_title(f'{layer} - Blue Channel')
        axs[2].axis('off')
        
        plt.show()

# 例: sample1 の各層のチャネルを可視化
if __name__ == "__main__":
    save_dir = "/rda5/users/yamada_24/cwvae/logs/minerl_cwvae_20241017_145429/eval_2024_10_17_16_25_20/intermediate_outputs"
    sample_id = 1
    visualize_channel_outputs(save_dir, sample_id)
