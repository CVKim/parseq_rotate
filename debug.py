from models.parseqra import PARSeqRotationAware
import torch
from PIL import Image
import torchvision.transforms.functional as F
from utils import Tokenizer

_WEIGHTS_URL = {
    "parseq-tiny": "https://github.com/baudm/parseq/releases/download/v1.0.0/parseq_tiny-e7a21b54.pt",
    "parseq": "https://github.com/baudm/parseq/releases/download/v1.0.0/parseq-bb5792a6.pt",
    "abinet": "https://github.com/baudm/parseq/releases/download/v1.0.0/abinet-1d1e373e.pt",
    "trba": "https://github.com/baudm/parseq/releases/download/v1.0.0/trba-cfaed284.pt",
    "vitstr": "https://github.com/baudm/parseq/releases/download/v1.0.0/vitstr-26d0fcf4.pt",
    "crnn": "https://github.com/baudm/parseq/releases/download/v1.0.0/crnn-679d0e31.pt",
}

import os
from typing import List, Optional, Union
import numpy as np
import torch

TensorOrPath = Union[str, np.ndarray, torch.Tensor]


def load_for_infer(path, H=32, W=128, device=None):
    """PIL 이미지 경로 -> (1, 3, H, W) 텐서"""
    img = Image.open(path).convert("RGB").resize((W, H), Image.BILINEAR)
    x = F.to_tensor(img).unsqueeze(0)  # (1, 3, H, W), [0,1] float32
    if device is not None:
        x = x.to(device)
    return x


if __name__ == "__main__":
    checkpoint = torch.hub.load_state_dict_from_url(
        url=_WEIGHTS_URL["parseq"],
        map_location="cpu",
        check_hash=True,
    )
    model = PARSeqRotationAware(
        img_size=(32, 128),
        charset="0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~",
        max_label_len=25,
        patch_size=(4, 8),  # [4,8]for [32,128] imgsize,[16,16] for [224,224] imgsize
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
        device="cpu",
    )
    model.load_state_dict(checkpoint, strict=True)
    model = model.eval()
    # 단일 이미지
    tokenizer = Tokenizer(model.charset)

    inputs = torch.cat([load_for_infer("msb_ng.png") for _ in range(1)], dim=0)

    inputs = (inputs - 0.5) / 0.5
    with torch.no_grad():
        output = model.forward_export(inputs)

    logits = output[..., :-1]
    attn_scores = output[..., -1]
    pred, p = tokenizer.decode(logits)
    if attn_scores[0, 0].item() > 0.5:
        single_pred = "normal"
    else:
        single_pred = "defect"
    print(pred, p)
    print(pred, single_pred)
    print(attn_scores[0, 0].item())
    print()
