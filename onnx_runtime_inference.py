
from typing import List

import numpy as np
import onnxruntime
import torch
from utils import Tokenizer
from PIL import Image
import torchvision.transforms.functional as F

class ONNXModel:
    def __init__(self, onnx_file_path: str, device: torch.device = None):
        """
        Initialize the ONNXModel with the given ONNX file path.

        Args:
            onnx_file_path (str): Path to the ONNX model file.
            providers (list of str, optional): List of providers to use for inference.
                                               Defaults to None, which lets ONNX Runtime choose.
        """
        self.onnx_file_path = onnx_file_path
        self.providers = self._map_device_to_providers(device)
        self.session = self._create_session()
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [output.name for output in self.session.get_outputs()]

    def _map_device_to_providers(self, device: torch.device) -> List[str]:
        """
        Map the torch.device to ONNX Runtime providers.

        Args:
            device (torch.device): The device to map.

        Returns:
            List[str]: A list of providers for ONNX Runtime.
        """
        if device is None:
            return None

        if device.type == "cpu":
            return None
        elif device.type == "cuda":
            # Check if CUDA provider is available
            available_providers = onnxruntime.get_available_providers()
            if "CUDAExecutionProvider" in available_providers:
                return ["CUDAExecutionProvider"]
            else:
                raise ValueError(
                    "CUDAExecutionProvider is not available in ONNX Runtime."
                )
        else:
            raise ValueError(f"Unsupported device type: {device.type}")

    def _create_session(self):
        """
        Create an ONNX Runtime InferenceSession.

        Returns:
            onnxruntime.InferenceSession: The created inference session.
        """
        if self.providers:
            return onnxruntime.InferenceSession(
                self.onnx_file_path, providers=self.providers
            )
        else:
            return onnxruntime.InferenceSession(self.onnx_file_path)

    def __call__(self, x: torch.Tensor) -> np.ndarray:
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()

        # Create a dictionary for the inputs
        inputs = {self.input_name: x}

        # Run inference
        outputs = self.session.run(self.output_names, inputs)[0]
        return outputs

def load_for_infer(path, H=32, W=128, device=None):
    """PIL 이미지 경로 -> (1, 3, H, W) 텐서"""
    img = Image.open(path).convert("RGB").resize((W, H), Image.BILINEAR)
    x = F.to_tensor(np.array(img, dtype=np.float32)).unsqueeze(0)  # (1, 3, H, W), [0,1] float32
    if device is not None:
        x = x.to(device)
    return x

if __name__ == "__main__":

    # onnx_model = ONNXModel(onnx_file_path='./model_2batch.onnx', device=torch.device("cpu"))
    onnx_model = ONNXModel(onnx_file_path='./model_2batch.onnx', device=torch.device("cuda"))

    tokenizer = Tokenizer('0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~')

    inputs = torch.cat([load_for_infer("1669129696_125103015175724_8_SWD_stitch_MSB_OK_0.bmp") for _ in range(1)], dim=0)
    
    print(f"Input Type: {inputs.dtype}")
    print(f"Input Min Value: {inputs.min().item()}")
    print(f"Input Max Value: {inputs.max().item()}")
    
    input_shape = inputs.shape
    print(f"Input Shape (N, C, H, W): {input_shape}")
    print(f"Inference Height (H): {input_shape[2]}")
    print(f"Inference Width (W): {input_shape[3]}")
    
    onnx_output = onnx_model(inputs.cpu().numpy())
    
    logits = torch.tensor(onnx_output[..., :-1])
    attn_scores = torch.tensor(onnx_output[..., -1])
    pred, p = tokenizer.decode(logits)
    if attn_scores[0, 0].item() > 0.5:
        single_pred = "normal"
    else:
        single_pred = "defect"
    print(pred, p)
    print(pred, single_pred)
    print(attn_scores[0, 0].item())

    print()
