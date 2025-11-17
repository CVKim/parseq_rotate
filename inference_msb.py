import os
import csv
import glob
import argparse
from pathlib import Path
from typing import List, Optional, Union, Tuple
import math

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms.functional as F

from models.parseqra import PARSeqRotationAware
from utils import Tokenizer

_WEIGHTS_URL = {
    "parseq-tiny": "https://github.com/baudm/parseq/releases/download/v1.0.0/parseq_tiny-e7a21b54.pt",
    "parseq": "https://github.com/baudm/parseq/releases/download/v1.0.0/parseq-bb5792a6.pt",
    "abinet": "https://github.com/baudm/parseq/releases/download/v1.0.0/abinet-1d1e373e.pt",
    "trba": "https://github.com/baudm/parseq/releases/download/v1.0.0/trba-cfaed284.pt",
    "vitstr": "https://github.com/baudm/parseq/releases/download/v1.0.0/vitstr-26d0fcf4.pt",
    "crnn": "https://github.com/baudm/parseq/releases/download/v1.0.0/crnn-679d0e31.pt",
}

TensorOrPath = Union[str, np.ndarray, torch.Tensor]

# -----------------------
# Image + model utilities
# -----------------------

def load_for_infer(path: str, H: int = 32, W: int = 128, device: Optional[torch.device] = None, rotate_deg: Optional[int] = None) -> torch.Tensor:
    """Load image -> (1,3,H,W) float32 tensor in [0,1], optional rotation before resize."""
    img = Image.open(path).convert("RGB")
    if rotate_deg is not None:
        img = img.rotate(rotate_deg, resample=Image.BILINEAR, expand=False)
    img = img.resize((W, H), Image.BILINEAR)
    x = F.to_tensor(img).unsqueeze(0)
    return x.to(device) if device is not None else x

def norm_m11(x: torch.Tensor) -> torch.Tensor:
    """Normalize [0,1] -> [-1,1]"""
    return (x - 0.5) / 0.5

def safe_mean_conf(prob_like) -> float:
    """
    Try to make a single sequence-level confidence.
    If tokenizer.decode returns per-char probs, we average them.
    If it's already a scalar, just float() it.
    """
    try:
        if isinstance(prob_like, (list, tuple, np.ndarray, torch.Tensor)):
            if isinstance(prob_like, torch.Tensor):
                prob_like = prob_like.detach().cpu().numpy()
            arr = np.asarray(prob_like).astype(float)
            if arr.size == 0:
                return float("nan")
            return float(np.nanmean(arr))
        return float(prob_like)
    except Exception:
        return float("nan")

# -----------------------
# Visualization helpers
# -----------------------

def draw_label(draw: ImageDraw.ImageDraw, xy: Tuple[int, int], text: str, font: ImageFont.ImageFont):
    """Draw text with a small dark box behind it."""
    x, y = xy
    padding = 4
    tw, th = draw.textbbox((0, 0), text, font=font)[2:]
    draw.rectangle([x - padding, y - padding, x + tw + padding, y + th + padding], fill=(0, 0, 0, 200))
    draw.text((x, y), text, fill=(255, 255, 255), font=font)

def make_merged_preview(src_path: str,
                        out_path: str,
                        pred_left: str, conf_left: float, attn_left: float, bin_left: str,
                        pred_right: str, conf_right: float, attn_right: float, bin_right: str,
                        rotate_deg_right: int = 180,
                        max_side: int = 640):
    """
    Create a side-by-side preview: left = original, right = rotated(180).
    Scales images to fit within max_side and overlays predictions.
    """
    imgL = Image.open(src_path).convert("RGB")
    imgR = imgL.rotate(rotate_deg_right, resample=Image.BILINEAR, expand=False)

    # Fit height to max_side while keeping aspect; cap width too
    def _resize_keep_max(im: Image.Image, max_side: int) -> Image.Image:
        w, h = im.size
        scale = min(max_side / max(w, h), 1.0)
        if scale != 1.0:
            im = im.resize((int(w * scale), int(h * scale)), Image.BILINEAR)
        return im

    imgL = _resize_keep_max(imgL, max_side)
    imgR = _resize_keep_max(imgR, max_side)

    h = max(imgL.height, imgR.height)
    canvas = Image.new("RGB", (imgL.width + imgR.width, h), (30, 30, 30))
    canvas.paste(imgL, (0, 0))
    canvas.paste(imgR, (imgL.width, 0))

    draw = ImageDraw.Draw(canvas, "RGBA")
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", size=16)
    except Exception:
        font = ImageFont.load_default()

    # Left annotations
    left_text = f"orig: {pred_left} | seq_conf={conf_left:.3f} | attn0={attn_left:.3f} | {bin_left}"
    draw_label(draw, (8, 8), left_text, font)

    # Right annotations
    right_text = f"rot{rotate_deg_right}: {pred_right} | seq_conf={conf_right:.3f} | attn0={attn_right:.3f} | {bin_right}"
    draw_label(draw, (imgL.width + 8, 8), right_text, font)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)

# -----------------------
# Main runner
# -----------------------

def main():
    ap = argparse.ArgumentParser(description="Batch inference for PARSeqRotationAware with 180° twin & previews.")
    ap.add_argument("--input_dir", default='cropped_msb', required=False, help="Folder with images")
    ap.add_argument("--output_dir", default= 'output_msb', required=False, help="Folder to save results (images + CSV)")
    ap.add_argument("--weights", type=str, default="parseq", choices=list(_WEIGHTS_URL.keys()),
                    help="Pretrained weights key")
    ap.add_argument("--img_h", type=int, default=32)
    ap.add_argument("--img_w", type=int, default=128)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                    choices=["cpu", "cuda"])
    ap.add_argument("--thr", type=float, default=0.5, help="Threshold on attn_scores[:,0] for normal/defect")
    ap.add_argument("--exts", type=str, default="png,jpg,jpeg,bmp,webp,tif,tiff",
                    help="Comma-separated list of extensions to include")
    ap.add_argument("--max_preview_side", type=int, default=640, help="Max side for merged preview")
    args = ap.parse_args()

    device = torch.device(args.device)

    # Load model weights
    print(f"[INFO] Loading weights: {_WEIGHTS_URL[args.weights]}")
    checkpoint = torch.hub.load_state_dict_from_url(
        url=_WEIGHTS_URL[args.weights],
        map_location="cpu",
        check_hash=True,
    )

    # Build model (matches your config)
    model = PARSeqRotationAware(
        img_size=(args.img_h, args.img_w),
        charset="0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~",
        max_label_len=25,
        patch_size=(4, 8),  # [4,8] for [32,128]; [16,16] for [224,224]
        embed_dim=384,
        enc_num_heads=6,
        enc_mlp_ratio=4,
        enc_depth=12,
        dec_num_heads=12,
        dec_mlp_ratio=4,
        dec_depth=1,
        perm_num=6,
        perm_forward=True,
        perm_mirrored=True,
        decode_ar=True,
        refine_iters=1,
        dropout=0.0,
        device=str(device),
    )
    model.load_state_dict(checkpoint, strict=True)
    model = model.to(device).eval()
    tokenizer = Tokenizer(model.charset)

    # Collect images
    exts = [e.strip().lower() for e in args.exts.split(",")]
    img_paths: List[str] = []
    for e in exts:
        img_paths.extend(glob.glob(os.path.join(args.input_dir, f"**/*.{e}"), recursive=True))
    img_paths = sorted(img_paths)

    if not img_paths:
        print("[WARN] No images found.")
        return

    out_img_dir = os.path.join(args.output_dir, "merged")
    out_csv = os.path.join(args.output_dir, "results.csv")
    Path(out_img_dir).mkdir(parents=True, exist_ok=True)

    rows = []
    num_ok = 0
    num_ng = 0

    for idx, path in enumerate(img_paths, 1):
        rel = os.path.relpath(path, args.input_dir)
        print(f"[{idx}/{len(img_paths)}] {rel}")

        try:
            # Build 2-sample batch: original + 180
            x0 = load_for_infer(path, H=args.img_h, W=args.img_w, device=device, rotate_deg=None)
            x1 = load_for_infer(path, H=args.img_h, W=args.img_w, device=device, rotate_deg=180)
            x = torch.cat([x0, x1], dim=0)
            x = norm_m11(x)

            with torch.no_grad():
                output = model.forward_export(x)

            logits = output[..., :-1]        # (B, T, vocab)
            attn_scores = output[..., -1]    # (B, T)  — you used attn_scores[:,0]

            preds, probs = tokenizer.decode(logits)  # preds: list[str], probs: per-char probs or similar

            def bin_label(attn0: float, thr: float) -> str:
                return "normal" if attn0 > thr else "defect"

            # For each of the two views, compute numbers we’ll store/show
            results_per_view = []
            for b in range(2):
                attn0 = float(attn_scores[b, 0].item())
                seq_conf = safe_mean_conf(probs[b] if isinstance(probs, (list, tuple)) else probs)
                results_per_view.append((preds[b], seq_conf, attn0, bin_label(attn0, args.thr)))

            img_status = 'OK'
            if results_per_view[0][2] < 0.5 or not results_per_view[0][0].isdigit() or len(results_per_view[0][0]) != 4:
                img_status = 'NG'


            # Save merged preview
            out_img_path = os.path.join(out_img_dir, rel + "_merged.png")
            make_merged_preview(
                src_path=path,
                out_path=out_img_path,
                pred_left=results_per_view[0][0],
                conf_left=results_per_view[0][1],
                attn_left=results_per_view[0][2],
                bin_left=results_per_view[0][3],
                pred_right=results_per_view[1][0],
                conf_right=results_per_view[1][1],
                attn_right=results_per_view[1][2],
                bin_right=results_per_view[1][3],
                rotate_deg_right=180,
                max_side=args.max_preview_side,
            )

            # Row for CSV
            rows.append({
                "relpath": rel,
                "pred_orig": results_per_view[0][0],
                "seq_conf_orig": f"{results_per_view[0][1]:.6f}",
                "attn0_orig": f"{results_per_view[0][2]:.6f}",
                "binary_orig": results_per_view[0][3],
                "pred_rot180": results_per_view[1][0],
                "seq_conf_rot180": f"{results_per_view[1][1]:.6f}",
                "attn0_rot180": f"{results_per_view[1][2]:.6f}",
                "binary_rot180": results_per_view[1][3],
                "merged_path": os.path.relpath(out_img_path, args.output_dir),
                "img_status": img_status,
            })
            if img_status == 'OK':
                num_ok += 1
            else:
                num_ng += 1

        except Exception as e:
            print(f"  [ERROR] {e}")
            rows.append({
                "relpath": rel, "pred_orig": "", "seq_conf_orig": "", "attn0_orig": "",
                "binary_orig": "", "pred_rot180": "", "seq_conf_rot180": "", "attn0_rot180": "",
                "binary_rot180": "", "merged_path": "", "img_status": "", "error": str(e),
            })

    # Write CSV
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    fieldnames = ["relpath",
                  "pred_orig", "seq_conf_orig", "attn0_orig", "binary_orig",
                  "pred_rot180", "seq_conf_rot180", "attn0_rot180", "binary_rot180",
                  "merged_path", "img_status", "error"]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            if "error" not in r:
                r["error"] = ""
            writer.writerow(r)

    print(f"\n[DONE] Saved previews -> {out_img_dir}")
    print(f"[DONE] CSV -> {out_csv}")
    print(f"[STATS] ok={num_ok}  err={num_ng}  total={len(img_paths)}")

if __name__ == "__main__":
    main()