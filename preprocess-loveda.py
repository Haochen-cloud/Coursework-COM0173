import os
import numpy as np
from PIL import Image
import cv2
from tqdm import tqdm


ROOT = "./LoveDA"
OUT = "./loveda-processed"

os.makedirs(OUT, exist_ok=True)

splits = ["Urban", "Rural"]

# The original category is mapped to a binary classification (vegetation =1, non-vegetation =0)
VEGETATION_CLASSES = [5, 6]  # forest, agriculture

def convert_mask_to_binary(mask):
    # {5,6}=1，others: 0
    mask = np.array(mask)
    binary = np.isin(mask, VEGETATION_CLASSES).astype(np.uint8)
    return binary


def process_split(split):
    print(f"\nProcessing {split} ...")

    img_dir = os.path.join(ROOT, split, "images_png")
    mask_dir = os.path.join(ROOT, split, "masks_png")

    img_files = sorted(os.listdir(img_dir))

    out_img_dir = os.path.join(OUT, split, "images")
    out_msk_dir = os.path.join(OUT, split, "masks")
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_msk_dir, exist_ok=True)

    for fname in tqdm(img_files):
        if not fname.endswith(".png"):
            continue

        # read the images
        img_path = os.path.join(img_dir, fname)
        msk_path = os.path.join(mask_dir, fname)

        img = Image.open(img_path).convert("RGB")
        mask = Image.open(msk_path)

        # resize to（512×512）
        img = img.resize((512, 512), Image.BILINEAR)
        mask = mask.resize((512, 512), Image.NEAREST)

        # normalization 0-1
        img_np = np.array(img).astype(np.float32) / 255.0

        # 转成二分类
        mask_np = convert_mask_to_binary(mask).astype(np.uint8)

        # reshape to (1, H, W, C)
        img_np = np.expand_dims(img_np, axis=0)      # (1,512,512,3)
        mask_np = np.expand_dims(mask_np, axis=(0,3))  # (1,512,512,1)


        np.save(os.path.join(out_img_dir, fname.replace(".png", ".npy")), img_np)
        np.save(os.path.join(out_msk_dir, fname.replace(".png", ".npy")), mask_np)


if __name__ == "__main__":
    for s in splits:
        process_split(s)

    print("Finished preprocessing LoveDA!")

