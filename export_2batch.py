from typing import Sequence, Union
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torchvision.transforms.functional as F

from models.parseqra import PARSeqRotationAware
from utils import Tokenizer

_WEIGHTS_URL = {
    "parseq-tiny": "https://github.com/baudm/parseq/releases/download/v1.0.0/parseq_tiny-e7a21b54.pt",
    "parseq":      "https://github.com/baudm/parseq/releases/download/v1.0.0/parseq-bb5792a6.pt",
    "abinet":      "https://github.com/baudm/parseq/releases/download/v1.0.0/abinet-1d1e373e.pt",
    "trba":        "https://github.com/baudm/parseq/releases/download/v1.0.0/trba-cfaed284.pt",
    "vitstr":      "https://github.com/baudm/parseq/releases/download/v1.0.0/vitstr-26d0fcf4.pt",
    "crnn":        "https://github.com/baudm/parseq/releases/download/v1.0.0/crnn-679d0e31.pt",
}

TensorOrPath = Union[str, np.ndarray, torch.Tensor]

# ---------- I/O helper (outside the graph) ----------
def load_for_infer(path, H=32, W=128, device=None):
    img = Image.open(path).convert("RGB").resize((W, H), Image.BILINEAR)
    x = F.to_tensor(img).unsqueeze(0)  # (1,3,H,W) in [0,1]
    x = (x * 255.0).to(torch.uint8)    # make it 0..255 to match scale255=True
    return x.to(device) if device is not None else x

# ---------- Wrapper that augments inside the graph ----------
class E2EONNX(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        mean: Sequence[float] = (0.0, 0.0, 0.0),
        std:  Sequence[float] = (1.0, 1.0, 1.0),
        scale255: bool = True,       # input expected as 0..255 uint8/float
        add_rot180: bool = True,     # <— turn on rotation+concat inside graph
    ):
        super().__init__()
        self.model = model
        self.add_rot180 = add_rot180

        mean = torch.tensor(mean, dtype=torch.float32).view(1, -1, 1, 1)
        std  = torch.tensor(std,  dtype=torch.float32).view(1, -1, 1, 1)
        s = 255.0 if scale255 else 1.0
        # y = (x/s - mean)/std  ==  x * (1/(s*std)) + (-mean/std)
        self.register_buffer("mul",  1.0 / (s * std))
        self.register_buffer("add", -mean / std)

    def forward(self, x: torch.Tensor):
        # x: (N,3,H,W), typically uint8 0..255 or float 0..255
        x = x.to(torch.float32)
        x = x.mul(self.mul).add_(self.add)    # normalize to model’s expected range

        # 180° rotation = flip H and W; ONNX-friendly
        x_rot = torch.flip(x, dims=(2, 3))
        x = torch.cat([x, x_rot], dim=0)  # (2N,3,H,W)

        return self.model(x)                  # model.forward_export will run

if __name__ == "__main__":
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
    model.forward = model.forward_export  # use export path

    # Wrap with in-graph normalization + rotation concat
    wrapped = E2EONNX(
        model,
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5),
        scale255=True,
        add_rot180=True,   # <— builds rot+concat into the ONNX
    ).eval()

    # Dummy input: 0..255, shape (N,3,H,W). Use uint8 or float32—both fine.
    dummy = torch.zeros(1, 3, img_size[0], img_size[1], dtype=torch.float32)

    # If you want to verify:
    # out = wrapped(dummy)  # will be batch=2 because of rotation

    torch.onnx.export(
        wrapped,
        dummy,
        "model_2batch.onnx",
        export_params=True,
        do_constant_folding=True,
        opset_version=17,
        input_names=["data"],
        output_names=["output"],
        dynamic_axes={
            "data":   {0: "N"},
            "output": {0: "N_out"},  # will be 2*N at inference
        },
    )
