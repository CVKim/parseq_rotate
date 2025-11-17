from typing import Sequence, Union
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torchvision.transforms.functional as F

from models.parseqra import PARSeqRotationAware

_WEIGHTS_URL = {
    "parseq": "https://github.com/baudm/parseq/releases/download/v1.0.0/parseq-bb5792a6.pt",
}

TensorOrPath = Union[str, np.ndarray, torch.Tensor]

# ---------- simple loader (outside the graph), returns float32 [0,1] ----------
def load_for_infer(path, H=32, W=128, device=None):
    img = Image.open(path).convert("RGB").resize((W, H), Image.BILINEAR)
    x = F.to_tensor(img).unsqueeze(0)  # (1,3,H,W) in [0,1], float32
    return x.to(device) if device is not None else x

# ---------- Wrapper that runs model on orig & 180° and combines ----------
class E2EONNX_TTA(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        mean: Sequence[float] = (0.5, 0.5, 0.5),
        std:  Sequence[float] = (0.5, 0.5, 0.5),
        scale255: bool = True,     # expect floats in [0,1]
        combine: str = "concat",    # "concat" | "mean" | "max"
    ):
        super().__init__()
        assert combine in ("concat", "mean", "max")
        self.model = model
        self.combine = combine

        mean = torch.tensor(mean, dtype=torch.float32).view(1, -1, 1, 1)
        std  = torch.tensor(std,  dtype=torch.float32).view(1, -1, 1, 1)
        s = 255.0 if scale255 else 1.0
        # y = (x/s - mean)/std  ==  x * (1/(s*std)) + (-mean/std)
        self.register_buffer("mul",  1.0 / (s * std))
        self.register_buffer("add", -mean / std)

    def forward(self, x: torch.Tensor):
        # x: (N,3,H,W) float32 (0..1 if scale255=False)
        x = x.to(torch.float32)
        x = x.mul(self.mul).add_(self.add)    # normalize

        # Two passes: original and 180° rotated
        y_orig = self.model(x)                            # (N, ...)
        x_rot  = torch.flip(x, dims=(2, 3))               # 180°
        y_rot  = self.model(x_rot)                        # (N, ...)

        if self.combine == "concat":
            # Stack predictions in the batch dimension → (2N, ...)
            y = torch.cat([y_orig, y_rot], dim=0)
        else:
            # Reduce across a new "tta" dimension → (N, ...)
            y = torch.stack([y_orig, y_rot], dim=1)       # (N, 2, ...)
            if self.combine == "mean":
                y = torch.mean(y, dim=1)
            else:  # "max"
                # elementwise max across TTA dimension
                y = torch.amax(y, dim=1)
        return y

if __name__ == "__main__":
    # --- build model ---
    checkpoint = torch.hub.load_state_dict_from_url(
        url=_WEIGHTS_URL["parseq"], map_location="cpu", check_hash=True
    )

    img_size = (32, 128)
    device = "cpu"

    model = PARSeqRotationAware(
        img_size=img_size,
        charset="0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~",
        max_label_len=25,
        patch_size=(4, 8),
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
        device=device,
    )
    model.load_state_dict(checkpoint, strict=True)
    model.forward = model.forward_export  # export-friendly path

    # --- wrap for ONNX with TTA combining inside the graph ---
    wrapped = E2EONNX_TTA(
        model,
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5),
        scale255=True,
        combine="concat",        # change to "mean" or "max" if desired
    ).eval()

    # --- export ---
    dummy = torch.zeros(1, 3, img_size[0], img_size[1], dtype=torch.float32)  # [0,1]

    # dynamic batch; output batch depends on combine mode
    out_axes = {"output": {0: "N2"}} if wrapped.combine == "concat" else {"output": {0: "N"}}

    torch.onnx.export(
        wrapped,
        dummy,
        "model_tta.onnx",
        export_params=True,
        do_constant_folding=True,
        opset_version=17,
        input_names=["data"],
        output_names=["output"],
        dynamic_axes={
            "data": {0: "N"},
            **out_axes,
        },
    )
