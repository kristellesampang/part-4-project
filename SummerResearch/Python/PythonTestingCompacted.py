import torch
import torch.nn.functional as F
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import numpy as np
import torch.quantization
import time
import json
import requests
import os
import sys
import re

# --- Constants ---
IMAGE_PATH = "C:/Users/pchh520/Documents/GitHub/part-4-project/SummerResearch/Python/cat.jpg"
MIF_OUTPUT_DIR = "C:/Users/pchh520/Documents/GitHub/part-4-project/SummerResearch/Python/mif_results"
N_GRID = 32  # Physical NPU Grid Size

# --- Model Classes ---
class QuantizableAlexNet(torch.nn.Module):
    def __init__(self, model_fp32):
        super(QuantizableAlexNet, self).__init__() 
        self.quant = torch.quantization.QuantStub() 
        self.dequant = torch.quantization.DeQuantStub()
        self.model_fp32 = model_fp32

    def forward(self, x, return_conv=False, im2col=False):
        x = self.quant(x)
        conv_out = self.model_fp32.features(x)
        if return_conv:
            if im2col:
                unfolded = torch.nn.functional.unfold(conv_out, kernel_size=3, stride=1, padding=1)
                return unfolded
            return conv_out
        x = self.model_fp32.avgpool(conv_out)
        x = torch.flatten(x, 1)
        x = self.model_fp32.classifier(x)
        x = self.dequant(x)
        return x

# --- Utility Functions ---
def load_quantized_alexnet():
    model_fp32 = models.alexnet(pretrained=True)
    model_fp32.eval()
    modules_to_fuse = [['0', '1'], ['3', '4'], ['6', '7'], ['8', '9'], ['10', '11']]
    torch.quantization.fuse_modules(model_fp32.features, modules_to_fuse, inplace=True)
    model = QuantizableAlexNet(model_fp32)
    model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    torch.quantization.prepare(model, inplace=True)
    model(torch.randn(1, 3, 224, 224))
    torch.quantization.convert(model, inplace=True)
    return model

def preprocess_image(image_path):
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img = Image.open(image_path).convert('RGB')
    return preprocess(img).unsqueeze(0)

def extract_conv_weights_and_activations(model, input_tensor, conv_idx, relu_idx):
    conv_layer = model.model_fp32.features[conv_idx]
    quantized_weight_tensor = conv_layer.weight() if callable(conv_layer.weight) else conv_layer.weight
    weights_int8 = quantized_weight_tensor.int_repr().cpu().numpy()
    weights_2d = weights_int8.reshape(weights_int8.shape[0], -1)

    activation = {}
    def hook(module, input, output): activation['relu'] = output
    relu_layer = model.model_fp32.features[relu_idx]
    h = relu_layer.register_forward_hook(hook)
    with torch.no_grad(): model(input_tensor)
    h.remove()

    q_act = activation['relu']
    act_float = q_act.dequantize().cpu() if hasattr(q_act, 'dequantize') else torch.tensor(q_act.int_repr().cpu().numpy(), dtype=torch.float32).unsqueeze(0)
    unfolded = F.unfold(act_float, kernel_size=3, stride=1, padding=1)
    act_2d = unfolded.squeeze(0).transpose(0, 1).numpy().astype(np.int8)
    return weights_2d, act_2d

# --- Fixed Core Logic ---
def coordinated_row_removal(data_matrix, weight_matrix):
    data_matrix = np.array(data_matrix)
    weight_matrix = np.array(weight_matrix)
    active_m = np.where(np.any(data_matrix, axis=1))[0]
    active_n = np.where(np.any(weight_matrix, axis=0))[0]
    data_k = set(np.where(np.any(data_matrix, axis=0))[0])
    weight_k = set(np.where(np.any(weight_matrix, axis=1))[0])
    common_k = sorted(list(data_k & weight_k))
    if not common_k or len(active_m) == 0 or len(active_n) == 0: return None
    
    compact_data = data_matrix[np.ix_(active_m, common_k)]
    compact_weight = weight_matrix[np.ix_(common_k, active_n)]
    return compact_data, compact_weight, len(active_m), len(active_n), len(common_k), active_m, active_n

def run_jtag_inference(m, n, k, s_data, s_weight, m_idx, n_idx, config=0):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    bin_file = os.path.join(current_dir, "tile.bin")
    res_file = os.path.join(current_dir, "result.bin")
    
    if os.path.exists(res_file): os.remove(res_file)

    dtype = np.int8 if config == 1 else np.int16
    header = bytes([int(m), int(n), int(k), int(config)])
    payload = s_data.astype(dtype).tobytes() + s_weight.astype(dtype).tobytes()

    with open(bin_file, "wb") as f:
        f.write(header + payload)

    start = time.time()
    while not os.path.exists(res_file):
        if time.time() - start > 20: return None
        time.sleep(0.1)
    
    raw_res = np.fromfile(res_file, dtype='<i4')
    os.remove(res_file)

    sparse_32x32 = np.zeros((32, 32), dtype=np.int32)
    dense_res = raw_res.reshape(m, n)
    
    for i, orig_row in enumerate(m_idx):
        for j, orig_col in enumerate(n_idx):
            sparse_32x32[orig_row, orig_col] = dense_res[i, j]
            
    return sparse_32x32


def prepare_simulation_case(model, layer_name, conv_idx, relu_idx, tile_idx, t_size, config=0):
    input_t = preprocess_image(IMAGE_PATH)
    weights_all, acts_all = extract_conv_weights_and_activations(model, input_t, conv_idx, relu_idx)
    
    raw_d = acts_all[tile_idx*t_size : (tile_idx+1)*t_size, :t_size]
    raw_w = weights_all[:t_size, :t_size]
    
    res_stripped = coordinated_row_removal(raw_d, raw_w)
    if not res_stripped: return None
    s_data, s_weight, m, n, k, m_idx, n_idx = res_stripped

    print(f"\nconstant M_VAL : integer := {m};")
    print(f"constant N_VAL : integer := {n};")
    print(f"constant K_VAL : integer := {k};")
    print(f"Config: {'Int8 8x8' if config == 1 else 'Int16 32x32'}")

    print(f"m_idx = {m_idx}")
    print(f"n_idx = {n_idx}")
    print(f"Compact Shapes: Data={s_data.shape}, Weight={s_weight.shape} | M={m}, N={n}, K={k}")
    print("Zero Rows in s_data:", np.where(~s_data.any(axis=1))[0])

    hw_reconstructed = run_jtag_inference(m, n, k, s_data, s_weight, m_idx, n_idx, config=config)
    if hw_reconstructed is None: return None

    # SW Bit-Accurate Reference
    if config == 1:
        # Int8 reference — 32-bit accumulator, no overflow wrapping needed
        full_precision = np.matmul(s_data.astype(np.int32), s_weight.astype(np.int32))
        sw_dense = full_precision.astype(np.int32)
    else:
        # Int16 reference — 64-bit then wrap to 32-bit
        full_precision = np.matmul(s_data.astype(np.int64), s_weight.astype(np.int64))
        sw_dense = ((full_precision + 2**31) % 2**32 - 2**31).astype(np.int32)

    hw_dense = hw_reconstructed[np.ix_(m_idx, n_idx)]

    print("\n" + "="*20 + " VERIFICATION " + "="*20)
    print(f"DENSE HARDWARE RESULT ({m}x{n}):")
    print(hw_dense)
    print("\nDENSE SOFTWARE REFERENCE:")
    print(sw_dense)
    
    if np.array_equal(sw_dense, hw_dense):
        print("\nSTATUS: 100% BIT-ACCURATE MATCH.")
    else:
        diff = sw_dense - hw_dense
        print(f"\nSTATUS: DISCREPANCY FOUND. Non-zero diffs: {np.count_nonzero(diff)}")
        print("Diff Matrix Sample (Top 5x5):")
        print(diff[:5, :5])

    return hw_reconstructed


def main():
    model = load_quantized_alexnet()
    # config=0 for Int16, config=1 for Int8
    prepare_simulation_case(model, "Conv5", 10, 11, tile_idx=0, t_size=16, config=0)

if __name__ == '__main__':
    main()