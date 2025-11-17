import os
import argparse
import pathlib
from typing import List
import numpy as np
import onnxruntime
import torch
from utils import Tokenizer  # 이 파일이 존재해야 합니다.
from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms.functional as F
from tqdm import tqdm # 진행률 표시를 위해 추가

# 
# -----------------------------------------------------------------
# ONNXModel 클래스 (제공해주신 코드 - 수정 없음)
# -----------------------------------------------------------------
# 
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
            # ONNX Runtime이 자동으로 CPU를 사용합니다.
            return ['CPUExecutionProvider']

        if device.type == "cpu":
            return ['CPUExecutionProvider']
        elif device.type == "cuda":
            # Check if CUDA provider is available
            available_providers = onnxruntime.get_available_providers()
            if "CUDAExecutionProvider" in available_providers:
                return ["CUDAExecutionProvider"]
            else:
                print("경고: CUDAExecutionProvider를 사용할 수 없습니다. CPU로 대체합니다.")
                return ["CPUExecutionProvider"]
        else:
            raise ValueError(f"Unsupported device type: {device.type}")

    def _create_session(self):
        """
        Create an ONNX Runtime InferenceSession.

        Returns:
            onnxruntime.InferenceSession: The created inference session.
        """
        return onnxruntime.InferenceSession(
            self.onnx_file_path, providers=self.providers
        )

    def __call__(self, x: torch.Tensor) -> np.ndarray:
        if isinstance(x, torch.Tensor):
            # ONNX Runtime은 CPU numpy 배열을 입력으로 받습니다.
            x = x.detach().cpu().numpy()

        # Create a dictionary for the inputs
        inputs = {self.input_name: x}

        # Run inference
        outputs = self.session.run(self.output_names, inputs)[0]
        return outputs

# 
# -----------------------------------------------------------------
# 이미지 로드 함수 (제공해주신 코드 - 수정 없음)
# -----------------------------------------------------------------
# 
def load_for_infer(path, H=32, W=128, device=None):
    """PIL 이미지 경로 -> (1, 3, H, W) 텐서"""
    img = Image.open(path).convert("RGB").resize((W, H), Image.BILINEAR)
    x = F.to_tensor(np.array(img, dtype=np.float32)).unsqueeze(0)  # (1, 3, H, W), [0,1] float32
    if device is not None:
        x = x.to(device)
    return x

# 
# -----------------------------------------------------------------
# (신규) 배치 처리, 시각화 및 저장 함수
# -----------------------------------------------------------------
# 
def process_and_save_batch(batch_outputs, batch_paths, tokenizer, output_dir, font):
    """
    추론된 배치 결과를 후처리하고, 원본 이미지에 시각화하여 저장합니다.
    """
    
    # 1. 배치 결과 후처리
    logits = torch.tensor(batch_outputs[..., :-1])
    attn_scores = torch.tensor(batch_outputs[..., -1])
    
    # Tokenizer가 배치 입력을 처리한다고 가정합니다.
    # preds: ["text1", "text2", ...], probs: [tensor1, tensor2, ...]
    preds, probs = tokenizer.decode(logits) 
    
    # 2. 배치의 각 이미지별로 시각화 및 저장
    for i in range(len(batch_paths)):
        try:
            image_path = batch_paths[i]
            text_pred = preds[i]
            # [i, 0] 인덱스는 모델 출력 구조에 따라 조정이 필요할 수 있습니다.
            attn_score = attn_scores[i, 0].item() 
            
            # 3. 분류 (Normal/Defect)
            if attn_score > 0.5:
                single_pred = "Normal"
                status_color = (0, 200, 0) # Green
            else:
                single_pred = "Defect"
                status_color = (255, 0, 0) # Red
            
            # 4. 시각화 (PIL 사용)
            # 원본 이미지를 다시 로드하여 고해상도로 시각화
            viz_image = Image.open(image_path).convert("RGB")
            draw = ImageDraw.Draw(viz_image)
            
            status_text = f"Status: {single_pred} ({attn_score:.2f})"
            pred_text = f"Pred: {text_pred}"
            
            # 텍스트가 잘 보이도록 배경 사각형 그리기
            draw.rectangle((0, 0, 250, 20), fill=status_color)
            draw.text((5, 2), status_text, fill="black", font=font)
            
            draw.rectangle((0, 22, 250, 42), fill=(255, 255, 255))
            draw.text((5, 24), pred_text, fill="black", font=font)
            
            # 5. 저장
            output_filename = os.path.basename(image_path)
            output_path = os.path.join(output_dir, output_filename)
            viz_image.save(output_path)
        
        except Exception as e:
            print(f"Error visualizing/saving {os.path.basename(image_path)}: {e}")

# 
# -----------------------------------------------------------------
# (신규) 메인 실행 함수 (argparse 및 배치 처리 로직)
# -----------------------------------------------------------------
# 
def main(args):
    """
    메인 로직: 이미지 로드, 배치 추론, 시각화
    """
    
    # 1. 환경 설정
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # 2. 모델 및 토크나이저 로드
    print(f"Loading ONNX model from: {args.onnx_path}")
    onnx_model = ONNXModel(onnx_file_path=args.onnx_path, device=device)
    tokenizer = Tokenizer('0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~')
    
    # 3. 출력 폴더 생성
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Results will be saved to: {args.output_dir}")
    
    # 4. 시각화 폰트 로드
    try:
        # Arial 폰트 시도 (더 깔끔함)
        font = ImageFont.truetype("arial.ttf", size=14)
    except IOError:
        print("Arial font not found, using default font.")
        font = ImageFont.load_default()
        
    # 5. 입력 폴더에서 이미지 검색
    supported_exts = ['.png', '.jpg', '.jpeg', '.bmp']
    image_paths = []
    for ext in supported_exts:
        image_paths.extend(list(pathlib.Path(args.input_dir).glob(f'*{ext}')))
    
    if not image_paths:
        print(f"No images (png, jpg, bmp) found in {args.input_dir}")
        return
        
    print(f"Found {len(image_paths)} images. Processing in batches of {args.batch_size}...")
    
    # 6. 배치 처리 루프
    batch_inputs = []
    batch_paths = []
    
    # tqdm으로 진행률 표시
    for image_path in tqdm(image_paths, desc="Processing Images"):
        try:
            str_path = str(image_path)
            # 이미지 로드 및 전처리
            inputs = load_for_infer(str_path, H=args.height, W=args.width, device=device)
            
            batch_inputs.append(inputs)
            batch_paths.append(str_path)
            
            # 배치가 꽉 차면 추론 실행
            if len(batch_inputs) == args.batch_size:
                batch_tensor = torch.cat(batch_inputs, dim=0)
                batch_outputs = onnx_model(batch_tensor) # (B, Seq, Vocab)
                
                # 결과 처리 및 저장
                process_and_save_batch(batch_outputs, batch_paths, tokenizer, args.output_dir, font)
                
                # 배치 리스트 초기화
                batch_inputs.clear()
                batch_paths.clear()
                
        except Exception as e:
            print(f"Error loading {image_path}: {e}")

    # 7. 마지막에 남은 배치 처리
    if batch_inputs:
        batch_tensor = torch.cat(batch_inputs, dim=0)
        batch_outputs = onnx_model(batch_tensor)
        
        process_and_save_batch(batch_outputs, batch_paths, tokenizer, args.output_dir, font)

    print(f"\nProcessing complete.")

# 
# -----------------------------------------------------------------
# (수정) __main__ 블록 -> argparse 사용
# -----------------------------------------------------------------
# 
if __name__ == "__main__":
    # 터미널에서 인자를 받기 위한 설정
    parser = argparse.ArgumentParser(description="Run ONNX OCR inference on a folder of images.")
    
    parser.add_argument("--input_dir", type=str, required=True, 
                        help="Directory containing input images. (예: ./cropped_msb_test)")
                        
    parser.add_argument("--output_dir", type=str, required=True, 
                        help="Directory to save visualized output images. (예: ./results_output)")
                        
    parser.add_argument("--onnx_path", type=str, default="./model_2batch.onnx", 
                        help="Path to the ONNX model file.")
                        
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], 
                        help="Device to run inference on (cuda or cpu).")
                        
    parser.add_argument("--batch_size", type=int, default=2, 
                        help="Batch size for inference. (모델 이름에 맞게 기본값 2로 설정)")
                        
    parser.add_argument("--height", type=int, default=32, 
                        help="Image height for model input.")
                        
    parser.add_argument("--width", type=int, default=128, 
                        help="Image width for model input.")
    
    args = parser.parse_args()
    
    # 메인 함수 실행
    main(args)