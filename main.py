import torch
from diffusers import StableDiffusionInpaintPipeline
from PIL import Image
import os
import argparse

def diffusersInpaint(original_images_folder, mask_images_folder, output_folder):
    # 事前学習済みのインペイントモデルをロード
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device", device)
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "runwayml/stable-diffusion-inpainting", 
        torch_dtype=torch.float16
    ).to(device)
    os.makedirs(output_folder, exist_ok=True)
    # 画像とマスクのリスト
    image_files = sorted(os.listdir(original_images_folder))
    mask_files = sorted(os.listdir(mask_images_folder))
    # インペイント処理
    for image_file, mask_file in zip(image_files, mask_files):
        # 画像とマスクのパス
        image_path = os.path.join(original_images_folder, image_file)
        mask_path = os.path.join(mask_images_folder, mask_file)
        # 画像とマスクを読み込み
        original_image = Image.open(image_path).convert("RGB").resize((128, 128))
        mask_image = Image.open(mask_path).convert("L").resize((128, 128))  # マスクはグレースケール
        # インペイント実行
        result = pipe(prompt="", image=original_image, mask_image=mask_image).images[0]
        # 結果を保存
        output_path = os.path.join(output_folder, image_file)
        result.resize((128, 128)).save(output_path)
        print(f"Saved inpainted image to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=str,
        help="input directory path",
    )
    parser.add_argument(
        "--mask",
        type=str,
        help="mask directory path",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="output directory path",
    )
    opt = parser.parse_args()
    diffusersInpaint(opt.input, opt.mask, opt.output)
