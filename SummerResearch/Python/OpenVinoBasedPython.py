import torch
import torchvision.models as models
import openvino as ov
import numpy as np
import os
import time
from PIL import Image
from torchvision import transforms

# --- Constants ---
MODEL_DIR = "alexnet_ir"
MODEL_XML = os.path.join(MODEL_DIR, "alexnet.xml")
IMAGE_PATH = "C:/Users/OEM/Documents/part-4-project/SummerResearch/Python/cat.jpg"
N_GRID = 32 

# --- CONFIG ---
MOCK_HARDWARE = True  
TARGET_CONV = "__module.features.0/aten::_convolution/Convolution"
TARGET_WEIGHTS = "self.features.0.weight_compressed"

def setup_openvino():
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
        model_torch = models.alexnet(pretrained=True).eval()
        ov_model = ov.convert_model(model_torch, example_input=torch.randn(1, 3, 224, 224))
        ov.save_model(ov_model, MODEL_XML)
    core = ov.Core()
    return core.read_model(MODEL_XML), core

def preprocess_image(path):
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img = Image.open(path).convert('RGB')
    return preprocess(img).unsqueeze(0).numpy()

def extract_layer_data(ov_model, core, input_data, layer_name, weight_name, precision="INT16"):
    # 1. Weights
    weights = next(op.get_data() for op in ov_model.get_ops() if weight_name == op.get_friendly_name())
    
    # 2. Balanced Quantization
    limit = 32767 if precision == "INT16" else 127
    dtype = np.int16 if precision == "INT16" else np.int8
    
    # We use a power-of-two divisor to keep math clean
    w_scale = limit / (np.max(np.abs(weights)) * 4) 
    a_scale = limit / (np.max(np.abs(input_data)) * 4)
    
    w_matrix = (weights * w_scale).reshape(weights.shape[0], -1).astype(dtype)
    a_quant_f = (input_data * a_scale).astype(np.float32)

    # 3. im2col
    input_tensor = torch.from_numpy(a_quant_f)
    unfolded = torch.nn.functional.unfold(input_tensor, kernel_size=11, stride=4, padding=2)
    a_matrix = unfolded.squeeze(0).transpose(0, 1).numpy().astype(dtype)

    return w_matrix, a_matrix

def coordinated_row_removal(data_matrix, weight_matrix):
    active_m = np.where(np.any(data_matrix, axis=1))[0]
    active_n = np.where(np.any(weight_matrix, axis=0))[0]
    data_k = set(np.where(np.any(data_matrix, axis=0))[0])
    weight_k = set(np.where(np.any(weight_matrix, axis=1))[0])
    common_k = sorted(list(data_k & weight_k))
    if not common_k or len(active_m) == 0 or len(active_n) == 0: return None
    return data_matrix[np.ix_(active_m, common_k)], weight_matrix[np.ix_(common_k, active_n)], active_m, active_n


def run_jtag_inference(m, n, k, s_data, s_weight):
    if MOCK_HARDWARE:
        # Full 64-bit precision internal accumulation
        full_p = np.matmul(s_data.astype(np.float64), s_weight.astype(np.float64))
        
        # --- THE TRUNCATION FIX ---
        # We simulate a 'Right Shift' (e.g., >> 8) to fit results into a readable range
        # This is what you must do in VHDL/NIOS to prevent overflow
        truncated = (full_p / 256).astype(np.int32)
        return truncated
    return None

def main():
    ov_model, core = setup_openvino()
    real_input = preprocess_image(IMAGE_PATH)
    
    w_matrix, a_matrix = extract_layer_data(ov_model, core, real_input, TARGET_CONV, TARGET_WEIGHTS)
    
    # Tile 0
    t_size = 32
    raw_d, raw_w = a_matrix[:t_size, :], w_matrix[:, :t_size].T
    
    result = coordinated_row_removal(raw_d, raw_w)
    if result:
        s_data, s_weight, m_idx, n_idx = result
        hw_match = run_jtag_inference(len(m_idx), len(n_idx), s_data.shape[1], s_data, s_weight)
        
        print(f"--- Results for REAL Cat (Truncated for Stability) ---")
        print(hw_match[:4, :4])
        print(f"\nMax value in tile: {np.max(hw_match)}")

if __name__ == "__main__":
    main() 