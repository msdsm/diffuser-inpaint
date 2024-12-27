import os
import argparse
import numpy as np
import cv2

# PSNRを計算する関数 (cv2.PSNRを使用)
def calculate_psnr(src_image, dst_image):
    # 画像をNumPy配列に変換
    src_image = np.array(src_image)
    dst_image = np.array(dst_image)
    
    # PSNRをcv2.PSNRを使って計算
    return cv2.PSNR(src_image, dst_image)

# 引数のパーサーを設定
def parse_args():
    parser = argparse.ArgumentParser(description='Calculate PSNR between two image directories.')
    parser.add_argument('--src', type=str, required=True, help='Path to the source image directory.')
    parser.add_argument('--dst', type=str, required=True, help='Path to the destination image directory.')
    return parser.parse_args()

def print_psnr():
    # 引数をパース
    args = parse_args()

    # 画像ファイルを昇順で取得
    src_images = sorted(os.listdir(args.src))
    dst_images = sorted(os.listdir(args.dst))

    # PSNRの計算
    psnr_values = []
    for src_image_file, dst_image_file in zip(src_images, dst_images):
        # ファイルパスを作成
        src_image_path = os.path.join(args.src, src_image_file)
        dst_image_path = os.path.join(args.dst, dst_image_file)

        # 画像を読み込む (OpenCVを使って読み込む)
        src_image = cv2.imread(src_image_path)
        dst_image = cv2.imread(dst_image_path)

        src_image = cv2.resize(src_image, (128, 128))
        dst_image = cv2.resize(dst_image, (128, 128))

        # PSNRを計算
        psnr = calculate_psnr(src_image, dst_image)
        psnr_values.append(psnr)
        print(f'PSNR for {src_image_file} vs {dst_image_file}: {psnr:.2f} dB')

    # PSNRの平均を計算
    average_psnr = np.mean(psnr_values)
    print(f'Average PSNR: {average_psnr:.2f} dB')
    print(average_psnr)

if __name__ == '__main__':
    print_psnr()
