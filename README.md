# README

## Overview
This repository provides a complete workflow for exporting a PARSeq-based text recognition model to ONNX, including:

- Rotation-aware OCR model (PARSeqRotationAware)  
- In-graph preprocessing (normalization + 180° rotation augmentation)
- ONNX export that doubles the batch size (N → 2N)
- Inference using ONNX Runtime
- Tokenizer-based decoding

The exported ONNX model performs:
1. Input normalization  
2. 180° rotated image generation  
3. Batch concatenation  
4. Forward pass of PARSeq  

All operations are embedded inside the ONNX graph.

---

## Files

### 1. `export_2batch.py`
Exports a rotation-aware PARSeq model to ONNX.

Key features:
- Loads pretrained PARSeq weights  
- Wraps the model with `E2EONNX` to include:
  - Input normalization  
  - Auto-scaling from 0–255  
  - 180° rotation + concatenation  
- Exports to `model_2batch.onnx`
- Output batch dimension becomes **2× larger** (original + rotated)

**Input expected:**  
Shape `(N, 3, H, W)` with values in **0–255**, dtype `uint8` or `float32`.

---

### 2. `onnx_runtime_inference.py`
A lightweight ONNX Runtime inference wrapper.

Features:
- Device-aware provider selection (CPU or CUDA)
- Loads ONNX model and runs inference
- Includes a helper function for image preprocessing
- Tokenizer-based decoding of logits
- Extracts:
  - logits (character predictions)
  - attention scores
  - defect/rotation classification

---

## Model Architecture

### PARSeqRotationAware
The base model performs:
- Vision Transformer encoder–decoder text recognition
- Rotation-aware decoding  
- Outputs:
  - Logits: `(batch, seq_len, vocab_size)`
  - A final attention score (used for normal/defect classification)

### In-Graph Wrapper (`E2EONNX`)
Embedded in the exported ONNX model and performs:
1. Normalization  
   \[
   y = \frac{x / 255 - \text{mean}}{\text{std}}
   \]
2. Generates a 180° rotated copy  
3. Concatenates along the batch dimension (resulting in 2×N)

This enables a fully self-contained ONNX model.

---

## Exporting the ONNX Model

Run:

```bash
python export_2batch.py



## Inference Example
from onnx_runtime_inference import ONNXModel, load_for_infer
from utils import Tokenizer
import torch

onnx_model = ONNXModel(
    onnx_file_path="./model_2batch.onnx",
    device=torch.device("cuda")
)

tokenizer = Tokenizer("0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~")

inputs = load_for_infer("example.bmp", H=32, W=128)
onnx_output = onnx_model(inputs.numpy())

logits = torch.tensor(onnx_output[..., :-1])
attn_scores = torch.tensor(onnx_output[..., -1])
pred, prob = tokenizer.decode(logits)

rotation_flag = "normal" if attn_scores[0, 0] > 0.5 else "defect"

print(pred, prob)
print("Rotation:", rotation_flag)

