"""HIFCIQA:Evaluate image quality using CLIP features and a KADID distortion gallery.

Run with the original defaults::

    python HIFC_IQA.py

Override individual settings with command-line arguments::

    python HIFC_IQA.py --patch-size 224 --top-n-types 10 --topn 1000 --alpha 0.8

Dependencies: torch, faiss-cpu (or faiss-gpu), opencv-python, open_clip_torch,
numpy, pandas, Pillow, scipy, and tqdm.

Use a separate cache file when changing the gallery data, model, or patch size.
"""

import argparse
import os
import random
from pathlib import Path

# Preserve the original endpoint while allowing an environment override.
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import cv2
import faiss
import numpy as np
import open_clip
import pandas as pd
import torch
from open_clip import tokenizer
from PIL import Image
from scipy.stats import pearsonr, spearmanr
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


# Settings outside the requested command-line parameters retain their defaults.
QUERY_IMAGE_DIR = "/home//datasets/iqa_dataset/PARA/imgs"
SEED = 42

DISTORTION_NAMES = {
    1: "Gaussian Blur", 2: "Lens Blur", 3: "Motion Blur",
    4: "Color Diffusion", 5: "Color Shift", 6: "Color Quant",
    7: "Color Sat. 1", 8: "Color Sat. 2", 9: "JPEG 2000",
    10: "JPEG", 11: "White Noise", 12: "White Noise Color",
    13: "Impulse Noise", 14: "Mult. Noise", 15: "Denoise",
    16: "Brighten", 17: "Darken", 18: "Mean Shift",
    19: "Jitter", 20: "Non-eccentricity", 21: "Pixelate",
    22: "Quantization", 23: "Color Block", 24: "High Sharp", 25: "Contrast",
}


def positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def alpha_value(value: str) -> float:
    number = float(value)
    if not 0.0 <= number <= 1.0:
        raise argparse.ArgumentTypeError("must be between 0 and 1")
    return number


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n", 1)[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--csv-path",
        default="/home/datasets/iqa_dataset/kadid10k/dmos.csv",
        help="Gallery CSV containing dist_img and dmos columns.",
    )
    parser.add_argument(
        "--image-dir",
        default="/home/datasets/iqa_dataset/kadid10k/images",
        help="Directory containing gallery images.",
    )
    parser.add_argument(
        "--cache-path",
        default="./cache/kadid_gallery_features.npz",
        help="Gallery feature cache (.npz); its parent directory is created.",
    )
    parser.add_argument(
        "--patch-size", type=positive_int, default=224,
        help="Size of semantic image patches in pixels.",
    )
    parser.add_argument(
        "--top-n-types", type=positive_int, default=10,
        help="Number of distortion types retained from the gallery.",
    )
    parser.add_argument(
        "--model-name", default="ViT-B-32",
        help="OpenCLIP model name, using the original openai pretrained weights.",
    )
    parser.add_argument(
        "--test-csv-path",
        default="/datasets/iqa_dataset/PARA/annotation/PARA-GiaaTest.csv",
        help="Query CSV containing sessionId, imageName, and aestheticScore_mean.",
    )
    parser.add_argument(
        "--topn", type=positive_int, default=1000,
        help="Semantic retrieval limit; effective count is min(topn, gallery_size // 2).",
    )
    parser.add_argument(
        "--alpha", type=alpha_value, default=0.8,
        help="Retrieval weight: alpha * retrieval_score + (1 - alpha) * CLIP_score.",
    )
    return parser


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def GMS(image, fragments_h=14, fragments_w=14, fsize_h=16, fsize_w=16):
    size_h = fragments_h * fsize_h
    size_w = fragments_w * fsize_w
    res_h, res_w = image.shape[:2]

    ratio = min(res_h / size_h, res_w / size_w)
    if ratio < 1:
        image = cv2.resize(image, (0, 0), fx=1 / ratio, fy=1 / ratio)

    res_h, res_w = image.shape[:2]
    hgrids = [min(res_h // fragments_h * i, res_h - fsize_h) for i in range(fragments_h)]
    wgrids = [min(res_w // fragments_w * i, res_w - fsize_w) for i in range(fragments_w)]
    hlength, wlength = res_h // fragments_h, res_w // fragments_w

    if hlength > fsize_h:
        rnd_h = np.random.randint(hlength - fsize_h, size=(fragments_h, fragments_w))
    else:
        rnd_h = np.zeros((fragments_h, fragments_w), dtype=int)
    if wlength > fsize_w:
        rnd_w = np.random.randint(wlength - fsize_w, size=(fragments_h, fragments_w))
    else:
        rnd_w = np.zeros((fragments_h, fragments_w), dtype=int)

    target = np.zeros((size_h, size_w, 3), dtype=np.uint8)
    for i, hs in enumerate(hgrids):
        for j, ws in enumerate(wgrids):
            h_s, h_e = i * fsize_h, (i + 1) * fsize_h
            w_s, w_e = j * fsize_w, (j + 1) * fsize_w
            h_so = hs + rnd_h[i, j]
            w_so = ws + rnd_w[i, j]
            target[h_s:h_e, w_s:w_e] = image[
                h_so:h_so + fsize_h, w_so:w_so + fsize_w
            ]
    return shuffle_gms_blocks(target, fragments_h, fragments_w, fsize_h, fsize_w)


def shuffle_gms_blocks(gms_image, fragments_h, fragments_w, fsize_h, fsize_w):
    height, width, channels = gms_image.shape
    assert height == fragments_h * fsize_h
    assert width == fragments_w * fsize_w

    num_blocks = fragments_h * fragments_w
    permutation = np.random.permutation(num_blocks)
    blocks = gms_image.reshape(
        fragments_h, fsize_h, fragments_w, fsize_w, channels
    ).transpose(0, 2, 1, 3, 4)
    blocks = blocks.reshape(num_blocks, fsize_h, fsize_w, channels)[permutation]
    blocks = blocks.reshape(
        fragments_h, fragments_w, fsize_h, fsize_w, channels
    ).transpose(0, 2, 1, 3, 4)
    return blocks.reshape(height, width, channels)


def get_image_patches(image, transform, patch_size=224, stride=224):
    """Extract semantic patches using the original resizing and stride rules."""
    width, height = image.size
    if min(height, width) > 448:
        scale = min(height, width) / 448
        image = image.resize((int(width / scale), int(height / scale)))
        width, height = image.size

    if width < patch_size or height < patch_size:
        scale = patch_size / min(width, height)
        image = image.resize(
            (int(width * scale) + 1, int(height * scale) + 1), Image.BICUBIC
        )
        width, height = image.size

    patches = []
    for y in range(0, height - patch_size + 1, stride):
        for x in range(0, width - patch_size + 1, stride):
            patches.append(transform(image.crop((x, y, x + patch_size, y + patch_size))))

    if width % stride != 0 and width > patch_size:
        for y in range(0, height - patch_size + 1, stride):
            patch = image.crop((width - patch_size, y, width, y + patch_size))
            patches.append(transform(patch))
    if height % stride != 0 and height > patch_size:
        for x in range(0, width - patch_size + 1, stride):
            patch = image.crop((x, height - patch_size, x + patch_size, height))
            patches.append(transform(patch))
    if not patches:
        patches.append(transform(image.resize((patch_size, patch_size), Image.BICUBIC)))

    return torch.stack(patches)


class ImageQuery(Dataset):
    """Load semantic patches, a GMS view, and normalized opinion scores."""

    def __init__(self, df, root, img_col, mos_col, transform, patch_size=224):
        self.df = df.reset_index(drop=True).copy()
        self.root = root
        self.img_col = img_col
        self.mos_col = mos_col
        self.transform = transform
        self.patch_size = patch_size

        mos = self.df[mos_col].to_numpy(dtype=np.float64)
        score_range = mos.max() - mos.min()
        self.df[mos_col] = (mos - mos.min()) / score_range if score_range else 0.0

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        with Image.open(os.path.join(self.root, row[self.img_col])) as source:
            image = source.convert("RGB")
        sem = get_image_patches(image, self.transform, self.patch_size)
        gms = self.transform(Image.fromarray(GMS(np.array(image))))
        return sem, gms.unsqueeze(0), float(row[self.mos_col])


@torch.no_grad()
def extract_dual_features(loader, model, device, tag=""):
    texts = ["a good image", "a bad image"]
    text_feat = model.encode_text(tokenizer.tokenize(texts).to(device))
    text_feat /= text_feat.norm(dim=-1, keepdim=True)
    sem_all, gms_all, mos_all, clip_mos = [], [], [], []

    for sem, gms, mos in tqdm(loader, desc=tag):
        sem = sem.squeeze(0).to(device)
        gms = gms.squeeze(0).to(device)
        f_sem = model.encode_image(sem).mean(0, keepdim=True)
        f_gms = model.encode_image(gms)
        f_sem /= f_sem.norm(dim=-1, keepdim=True)
        f_gms /= f_gms.norm(dim=-1, keepdim=True)

        text_probs = torch.softmax((f_sem @ text_feat.T) * 100, dim=-1)
        clip_mos.append(text_probs[0, 0].item())
        sem_all.append(f_sem.cpu().numpy())
        gms_all.append(f_gms.cpu().numpy())
        mos_all.extend(mos.tolist())

    return (
        np.concatenate(sem_all),
        np.concatenate(gms_all),
        np.asarray(mos_all, dtype=np.float64),
        np.asarray(clip_mos),
    )


def build_or_load_gallery(loader, model, device, df, cache_path):
    cache_path = Path(cache_path)
    if cache_path.exists():
        print(">>> Load cached gallery features")
        with np.load(cache_path, allow_pickle=False) as data:
            return tuple(data[key] for key in (
                "g_sem", "g_gms", "g_mos", "clip_mos", "dist_type"
            ))

    print(">>> Build gallery features")
    g_sem, g_gms, g_mos, clip_mos = extract_dual_features(loader, model, device, "Gallery")
    dist_type = df["dist_img"].str.extract(r"_(\d{2})_", expand=False).astype(int).to_numpy()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    # Passing a file handle preserves the exact requested cache filename.
    with cache_path.open("wb") as cache_file:
        np.savez(
            cache_file,
            g_sem=g_sem.astype(np.float32),
            g_gms=g_gms.astype(np.float32),
            g_mos=g_mos.astype(np.float32),
            clip_mos=clip_mos.astype(np.float32),
            dist_type=dist_type,
        )
    print(">>> Gallery cached")
    return g_sem, g_gms, g_mos, clip_mos, dist_type


def filter_by_target(q_gms, g_gms, dist_type, topk=10):
    """Keep the distortion types most similar to the query GMS centroid."""
    q_gms /= np.linalg.norm(q_gms, axis=1, keepdims=True)
    target_centroid = q_gms.mean(0)
    scores = []
    for distortion_type in np.unique(dist_type):
        indices = np.where(dist_type == distortion_type)[0]
        centroid = g_gms[indices].mean(0)
        centroid /= np.linalg.norm(centroid)
        scores.append((distortion_type, np.dot(target_centroid, centroid)))

    scores.sort(key=lambda item: item[1], reverse=True)
    selected = [distortion_type for distortion_type, _ in scores[:topk]]
    print(f"{'Rank':<5} | {'Type ID':<8} | {'Similarity':<10} | Status")
    print("-" * 45)
    for rank, (distortion_type, similarity) in enumerate(scores):
        status = "KEEP" if rank < topk else "DROP"
        description = DISTORTION_NAMES.get(distortion_type, "Unknown")
        print(
            f"{rank + 1:<5} | {distortion_type:<8} | {similarity:.4f}     | "
            f"{status} ({description})"
        )
    print(">>> Keep distortion types:", selected)
    return np.where(np.isin(dist_type, selected))[0]


def main(args: argparse.Namespace) -> None:
    """Run gallery extraction, target filtering, retrieval, and evaluation."""
    set_seed()
    if torch.cuda.is_available():
        # Keep cuda:2 when available and support machines with fewer GPUs.
        device = torch.device(f"cuda:{min(2, torch.cuda.device_count() - 1)}")
    else:
        device = torch.device("cpu")
    model, _, preprocess = open_clip.create_model_and_transforms(
        args.model_name, pretrained="openai"
    )
    model.to(device).eval()

    gallery_df = pd.read_csv(args.csv_path)
    gallery_loader = DataLoader(
        ImageQuery(
            gallery_df, args.image_dir, "dist_img", "dmos", preprocess,
            patch_size=args.patch_size,
        ),
        batch_size=1, shuffle=False,
    )
    g_sem_all, g_gms_all, g_mos_all, _, dist_type = build_or_load_gallery(
        gallery_loader, model, device, gallery_df, args.cache_path
    )

    test_df = pd.read_csv(args.test_csv_path)
    query_loader = DataLoader(
        ImageQuery(
            test_df, QUERY_IMAGE_DIR, "imageName", "MOS", preprocess,
            patch_size=args.patch_size,
        ),
        batch_size=1, shuffle=False,
    )
    q_sem, q_gms, q_mos, q_clip_mos = extract_dual_features(
        query_loader, model, device, "Query"
    )

    selected = filter_by_target(q_gms, g_gms_all, dist_type, args.top_n_types)
    g_sem = g_sem_all[selected]
    g_gms = g_gms_all[selected]
    g_mos = g_mos_all[selected].reshape(-1)

    topn = min(args.topn, len(g_mos) // 2)
    if topn < 1:
        raise ValueError("The filtered gallery must contain at least two images.")
    index_sem = faiss.IndexFlatIP(g_sem.shape[1])
    index_sem.add(g_sem.astype(np.float32))
    _, candidate_indices = index_sem.search(q_sem.astype(np.float32), topn)

    predictions = []
    for i in tqdm(range(len(q_mos)), desc="Retrieval"):
        candidates = candidate_indices[i]
        gms_sims = (g_gms[candidates] @ q_gms[i:i + 1].T).reshape(-1)
        ranked = np.argsort(-gms_sims)
        retained = ranked[gms_sims[ranked] > gms_sims.mean()]
        targets = candidates[retained]
        if targets.size == 0:
            # One candidate or tied similarities otherwise produce an empty mean.
            targets = candidates
        retrieval_score = g_mos[targets].mean()
        predictions.append(
            args.alpha * retrieval_score + (1 - args.alpha) * q_clip_mos[i]
        )

    predictions = np.asarray(predictions, dtype=np.float64)
    srcc, _ = spearmanr(predictions, q_mos)
    plcc, _ = pearsonr(predictions, q_mos)
    print(f"{'Top-N':<8} | {'SRCC':<8} | {'PLCC':<8}")
    print("-" * 30)
    print(f"{topn:<8} | {srcc:.4f}   | {plcc:.4f}")


if __name__ == "__main__":
    main(build_parser().parse_args())
