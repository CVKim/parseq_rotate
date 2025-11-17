from typing import List, Optional, Union, Sequence

import os
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
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

class E2EONNX(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        mean: Sequence[float] = (0.0, 0.0, 0.0),
        std:  Sequence[float] = (1.0, 1.0, 1.0),
        scale255: bool = True,   # True -> input is 0..255; False -> input is 0..1
    ):
        super().__init__()
        self.model = model
        mean = torch.tensor(mean, dtype=torch.float32).view(1, -1, 1, 1)
        std  = torch.tensor(std,  dtype=torch.float32).view(1, -1, 1, 1)

        s = 255.0 if scale255 else 1.0
        # y = (x/s - mean)/std  ==  x * (1/(s*std)) + (-mean/std)
        self.register_buffer("mul",  1.0 / (s * std))
        self.register_buffer("add", -mean / std)

    def forward(self, x: torch.Tensor):
        x = x.to(torch.float32)           # Cast happens inside the graph
        x = x.mul(self.mul).add_(self.add)
        return self.model(x)
    
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
    img_size = (32,128)
    device="cpu"

    model = PARSeqRotationAware(
        img_size=img_size,
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
    model.forward = model.forward_export

    wrapped = E2EONNX(model, mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5), scale255=True)
    wrapped.eval()
    
    image_export = torch.zeros(1, 3, img_size[0], img_size[1]).to(device)   

    torch.onnx.export(
        wrapped,
        image_export,
        'model.onnx',
        export_params=True,
        do_constant_folding=True,
        verbose=False,
        opset_version=17,
        input_names=["data"],
        output_names=["output"],
    )