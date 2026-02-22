import torch
import torchvision.models as models
import openvino as ov
import numpy as np
import os
import time
import sys

# --- Constants ---
MODEL_DIR = "alexnet_ir"
MODEL_XML = os.path.join(MODEL_DIR, "alexnet.xml")
IMAGE_PATH = "C:/Users/pchh520/Documents/GitHub/part-4-project/SummerResearch/Python/cat.jpg"
N_GRID = 32  # Physical NPU Grid Size

# --- 1. OpenVINO Setup & Conversion ---
def setup_openvino():
    """Converts PyTorch AlexNet to OpenVINO IR if not already present."""
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
        print("--- Converting PyTorch AlexNet to OpenVINO IR (FP32) ---")
        model_torch = models.alexnet(pretrained=True).eval()
        # We export in FP32 so we can manually cast to INT8 or INT16 later
        ov_model = ov.convert_model(model_torch, example_input=torch.randn(1, 3, 224, 224))
        ov.save_model(ov_model, MODEL_XML)
    
    core = ov.Core()
    return core.read_model(MODEL_XML), core

# --- 2. Data Extraction (Replaces Torch Hooks) ---
def extract_layer_data(ov_model, core, input_data, layer_name, precision="INT16"):
    """Extracts weights and activations, then quantizes to research specs."""
    # Add the specific layer as an output so we can 'peek' at activations
    ov_model.add_outputs([layer_name])
    compiled = core.compile_model(ov_model, "CPU")
    infer_req = compiled.create_infer_request()
    
    # Run Inference
    results = infer_req.infer({0: input_data})
    raw_acts = results[layer_name]

    # Extract Weights from the Graph
    weights = None
    for op in ov_model.get_ops():
        if layer_name in op.get_friendly_name() and "weights" in op.get_friendly_name():
            weights = op.get_data()
            break
            
    if weights is None:
        raise ValueError(f"Could not find weights for layer: {layer_name}")

    # --- Manual Quantization Logic (INT16 vs INT8) ---
    # We use a simple symmetric scaling to fit your signed ranges
    if precision == "INT16":
        limit = 32767
        dtype = np.int16
    else:
        limit = 127
        dtype = np.int8

    w_scale = limit / np.max(np.abs(weights))
    a_scale = limit / np.max(np.abs(raw_acts))
    
    w_quant = (weights * w_scale).astype(dtype)
    # Reshape weights to 2D (Out_Channels, Flattened_Kernel)
    w_quant = w_quant.reshape(w_quant.shape[0], -1)
    
    # Simple flatten for activations to match your existing 2D tile logic
    # Note: For real Conv, you'd use im2col here. 
    # For now, we mimic your previous 2D extraction:
    a_quant = raw_acts.reshape(raw_acts.shape[1], -1).T.astype(dtype)

    return w_quant, a_quant

# --- 3. Your Verified Research Logic (DO NOT CHANGE) ---
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

def run_jtag_inference(m, n, k, s_data, s_weight):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    bin_file = os.path.join(current_dir, "tile.bin")
    res_file = os.path.join(current_dir, "result.bin")
    if os.path.exists(res_file): os.remove(res_file)

    grid_data = np.zeros((N_GRID, N_GRID), dtype=np.int16)
    grid_weight = np.zeros((N_GRID, N_GRID), dtype=np.int16)
    grid_data[:s_data.shape[0], :s_data.shape[1]] = s_data.astype(np.int16)
    grid_weight[:s_weight.shape[0], :s_weight.shape[1]] = s_weight.astype(np.int16)

    header = bytes([int(m), int(n), int(k), 0]) 
    payload = grid_data.tobytes() + grid_weight.tobytes()

    with open(bin_file, "wb") as f:
        f.write(header + payload)

    print(f"JTAG: Sent 32x32 Grid. M={m}, N={n}, K={k}. Waiting...")
    start = time.time()
    while not os.path.exists(res_file):
        if time.time() - start > 20: return None
        time.sleep(0.1)
    
    raw_res = np.fromfile(res_file, dtype='<i4')
    return raw_res.reshape(N_GRID, N_GRID)[:m, :n]

# --- 4. Main Execution Pipeline ---
def main():
    # A. Initial Setup
    ov_model, core = setup_openvino()
    
    # B. Input Preprocessing (Dummy or Real)
    dummy_input = np.random.randn(1, 3, 224, 224).astype(np.float32)
    
    # C. Extract Data (Choose Precision here!)
    # Layer Names for AlexNet: 'features.0', 'features.3', 'features.6', etc.
    target_layer = "features.0" 
    precision_mode = "INT16" # OR "INT8"
    
    weights, activations = extract_layer_data(ov_model, core, dummy_input, target_layer, precision=precision_mode)
    
    # D. Slicing a Tile (Example: Tile 0)
    t_size = 32
    raw_d = activations[:t_size, :t_size]
    raw_w = weights[:t_size, :t_size]
    
    # E. Stripping & Hardware Inference
    res_stripped = coordinated_row_removal(raw_d, raw_w)
    if res_stripped:
        s_data, s_weight, m, n, k, m_idx, n_idx = res_stripped
        
        # BIT-ACCURATE SOFTWARE REF
        full_p = np.matmul(s_data.astype(np.int64), s_weight.astype(np.int64))
        sw_match = (full_p + 2**31) % 2**32 - 2**31
        
        # HARDWARE INFERENCE (Ensure TCL script is running!)
        hw_match = run_jtag_inference(m, n, k, s_data, s_weight)
        
        if hw_match is not None:
            print(f"\nHardware Result for {target_layer} ({precision_mode}):")
            print(hw_match[:4, :4])
            print(f"Match: {np.array_equal(sw_match.astype(np.int32), hw_match)}")

if __name__ == "__main__":
    main()