import openvino as ov
import torch
import torchvision.models as models
import torch.nn.functional as F
import numpy as np
import os
import time
from PIL import Image
from torchvision import transforms

# --- Constants ---
IMAGE_PATH = "C:/Users/pchh520/Documents/GitHub/part-4-project/SummerResearch/Python/cat.jpg"
MODEL_DIR  = "C:/Users/pchh520/Documents/GitHub/part-4-project/SummerResearch/Python/ov_models"

SUPPORTED_MODELS = {
    "alexnet":    (models.alexnet,    (1, 3, 224, 224)),
    "vgg16":      (models.vgg16,      (1, 3, 224, 224)),
    "resnet50":   (models.resnet50,   (1, 3, 224, 224)),
    "mobilenetv2":(models.mobilenet_v2,(1, 3, 224, 224)),
}

def get_or_convert_model(model_name):
    os.makedirs(MODEL_DIR, exist_ok=True)
    xml_path = os.path.join(MODEL_DIR, f"{model_name}.xml")
    if not os.path.exists(xml_path):
        print(f"Converting {model_name} to OpenVINO IR...")
        model_fn, input_shape = SUPPORTED_MODELS[model_name]
        torch_model = model_fn(pretrained=True).eval()
        dummy = torch.randn(*input_shape)
        ov_model = ov.convert_model(torch_model, example_input=dummy)
        ov.save_model(ov_model, xml_path)
        print(f"Saved to {xml_path}")
    core = ov.Core()
    ov_model = core.read_model(xml_path)
    return ov_model, core

def discover_conv_layers(ov_model):
    conv_layers = []
    for op in ov_model.get_ops():
        if op.get_type_name() != "Convolution":
            continue
        attrs        = op.get_attributes()
        strides      = attrs["strides"]
        pads         = attrs["pads_begin"]
        dilations    = attrs["dilations"]
        weight_node  = op.input(1).get_source_output().get_node()
        weight_shape = weight_node.get_output_shape(0)
        conv_layers.append({
            "name":         op.get_friendly_name(),
            "op":           op,
            "weight_node":  weight_node,
            "out_channels": int(weight_shape[0]),
            "in_channels":  int(weight_shape[1]),
            "kernel_h":     int(weight_shape[2]),
            "kernel_w":     int(weight_shape[3]),
            "strides":      list(strides),
            "pads":         list(pads),
            "dilations":    list(dilations),
            "input_shape":  str(op.input(0).get_partial_shape()),
            "layer_idx":    len(conv_layers),
        })
    print(f"\nDiscovered {len(conv_layers)} conv layers:")
    for i, l in enumerate(conv_layers):
        print(f"  [{i}] {l['name']}")
        print(f"      Filters: {l['out_channels']} x ({l['in_channels']} x {l['kernel_h']} x {l['kernel_w']})")
        print(f"      Stride: {l['strides']}  Pad: {l['pads']}  Dilation: {l['dilations']}")
    return conv_layers

def extract_all_activations(ov_model, core, input_data, conv_layers):
    augmented  = ov_model.clone()
    name_to_op = {op.get_friendly_name(): op for op in augmented.get_ops()}
    new_outputs = []
    for layer in conv_layers:
        op = name_to_op.get(layer["name"])
        if op is not None:
            new_outputs.append(op.output(0))
    augmented.add_outputs(new_outputs)
    compiled      = core.compile_model(augmented, "CPU")
    infer_request = compiled.create_infer_request()
    infer_request.infer({0: input_data})
    n_original  = len(ov_model.outputs)
    activations = {}
    for i, layer in enumerate(conv_layers):
        out_idx = n_original + i
        if out_idx < len(compiled.outputs):
            activations[layer["name"]] = np.maximum(0, infer_request.get_output_tensor(out_idx).data.copy())
    return activations

def extract_weights(layer):
    weight_node = layer["weight_node"]
    const_node  = weight_node.input(0).get_source_output().get_node()
    weights     = const_node.data
    weights_2d  = weights.reshape(weights.shape[0], -1)
    max_per_filter = np.max(np.abs(weights_2d), axis=1, keepdims=True)
    max_per_filter = np.where(max_per_filter == 0, 1.0, max_per_filter)
    weights_int8   = np.clip(np.round(weights_2d / max_per_filter * 127), -128, 127).astype(np.int8)
    return weights_int8

def im2col(activation, kernel_h, kernel_w, strides, pads):
    t = torch.from_numpy(activation.astype(np.float32))
    unfolded = F.unfold(t, kernel_size=(kernel_h, kernel_w), stride=strides, padding=pads)
    return unfolded.squeeze(0).transpose(0, 1).numpy()

def quantize_to_int8(matrix):
    max_val = np.max(np.abs(matrix))
    if max_val == 0:
        return matrix.astype(np.int8)
    scale = 127.0 / max_val
    return np.clip(np.round(matrix * scale), -128, 127).astype(np.int8)

def compute_sparsity(matrix):
    return np.sum(matrix == 0) / matrix.size

def decide_config(m, n, k):
    """
    Routes to one of three hardware modes based on actual stripped dimensions.
    Int8  8x8  — m<=8  and n<=8  (small sparse tile)
    Int16 16x16 — m<=16 and n<=16 (medium tile)
    Int16 32x32 — everything else  (large dense tile)
    Config byte: 1=Int8 8x8, 0=Int16 (both 16x16 and 32x32 use same SA)
    Tile size reported separately for logging.
    """
    if m <= 8 and n <= 8:
        return 1, 8,  "Int8  8x8"
    elif m <= 16 and n <= 16:
        return 0, 16, "Int16 16x16"
    else:
        return 0, 32, "Int16 32x32"

def coordinated_row_removal(data_matrix, weight_matrix):
    data_matrix   = np.array(data_matrix)
    weight_matrix = np.array(weight_matrix)
    active_m  = np.where(np.any(data_matrix,   axis=1))[0]
    active_n  = np.where(np.any(weight_matrix, axis=0))[0]
    data_k    = set(np.where(np.any(data_matrix,   axis=0))[0])
    weight_k  = set(np.where(np.any(weight_matrix, axis=1))[0])
    common_k  = sorted(list(data_k & weight_k))
    if not common_k or len(active_m) == 0 or len(active_n) == 0:
        return None
    compact_data   = data_matrix[np.ix_(active_m, common_k)]
    compact_weight = weight_matrix[np.ix_(common_k, active_n)]
    return compact_data, compact_weight, len(active_m), len(active_n), len(common_k), active_m, active_n

def run_jtag_inference(m, n, k, s_data, s_weight, m_idx, n_idx, config=0):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    bin_file = os.path.join(current_dir, "tile.bin")
    res_file = os.path.join(current_dir, "result.bin")

    if os.path.exists(res_file):
        os.remove(res_file)

    dtype   = np.int8 if config == 1 else np.int16
    header  = bytes([int(m), int(n), int(k), int(config)])
    payload = s_data.astype(dtype).tobytes() + s_weight.astype(dtype).tobytes()

    for attempt in range(10):
        try:
            with open(bin_file, "wb") as f:
                f.write(header + payload)
            break
        except PermissionError:
            time.sleep(0.2)
    else:
        print("  ERROR: Could not write tile.bin after 10 attempts")
        return None

    start = time.time()
    while not os.path.exists(res_file):
        if time.time() - start > 30:
            print("  ERROR: Timeout waiting for result.bin")
            return None
        time.sleep(0.1)

    prev_size    = -1
    stable_count = 0
    while stable_count < 3:
        try:
            curr_size = os.path.getsize(res_file)
        except OSError:
            time.sleep(0.05)
            continue
        if curr_size == prev_size and curr_size > 0:
            stable_count += 1
        else:
            stable_count = 0
        prev_size = curr_size
        time.sleep(0.05)

    raw_res    = np.fromfile(res_file, dtype='<i4')
    os.remove(res_file)

    sparse_out = np.zeros((32, 32), dtype=np.int32)
    dense_res  = raw_res.reshape(m, n)
    for i, orig_row in enumerate(m_idx):
        for j, orig_col in enumerate(n_idx):
            sparse_out[orig_row, orig_col] = dense_res[i, j]

    return sparse_out

def sw_reference(s_data, s_weight, config):
    if config == 1:
        full = np.matmul(s_data.astype(np.int32), s_weight.astype(np.int32))
        return full.astype(np.int32)
    else:
        full = np.matmul(s_data.astype(np.int64), s_weight.astype(np.int64))
        return ((full + 2**31) % 2**32 - 2**31).astype(np.int32)

def run_layer(layer_idx, layer, activation, t_size):
    name     = layer["name"]
    kernel_h = layer["kernel_h"]
    kernel_w = layer["kernel_w"]
    strides  = layer["strides"]
    pads     = layer["pads"]

    print(f"\n{'='*60}")
    print(f"Layer [{layer_idx}]: {name}")
    print(f"  Kernel: {kernel_h}x{kernel_w}  Stride: {strides}  Pad: {pads}")
    print(f"  Output channels: {layer['out_channels']}  Input channels: {layer['in_channels']}")

    weights_2d = extract_weights(layer)
    act_2d     = im2col(activation, kernel_h, kernel_w, strides, pads[0] if len(pads) == 1 else pads)

    total_rows = act_2d.shape[0]
    total_cols = weights_2d.shape[0]

    print(f"  Activation shape: {act_2d.shape}  Weight shape: {weights_2d.shape}")

    act_sparsity = compute_sparsity(act_2d)
    w_sparsity   = compute_sparsity(weights_2d)
    print(f"  Activation sparsity: {act_sparsity*100:.1f}%  Weight sparsity: {w_sparsity*100:.1f}%")

    num_tiles_row = max(1, (total_rows + t_size - 1) // t_size)
    num_tiles_col = max(1, (total_cols + t_size - 1) // t_size)
    total_tiles   = num_tiles_row * num_tiles_col
    print(f"  Tile size: {t_size}  Total tiles: {total_tiles}")

    layer_results = []
    tile_num    = 0
    int8_count  = 0
    int16_16_count = 0
    int16_32_count = 0
    match_count = 0

    for tr in range(0, total_rows, t_size):
        r_start  = tr
        r_end    = min(r_start + t_size, total_rows)
        raw_d    = act_2d[r_start:r_end, :t_size]
        raw_d_q8 = quantize_to_int8(raw_d)

        for tc in range(0, total_cols, t_size):
            tile_num += 1
            c_start  = tc
            c_end    = min(c_start + t_size, total_cols)
            raw_w    = weights_2d[c_start:c_end, :t_size].T
            raw_w_q8 = raw_w

            result = coordinated_row_removal(raw_d_q8, raw_w_q8)
            if result is None:
                print(f"  Tile {tile_num}/{total_tiles}: skipped (all zeros)")
                continue

            s_data, s_weight, m, n, k, m_idx, n_idx = result
            config, sa_size, reason = decide_config(m, n, k)

            if config == 1:
                int8_count += 1
            elif sa_size == 16:
                int16_16_count += 1
            else:
                int16_32_count += 1

            tile_sparsity = 1.0 - (m * k + k * n) / (raw_d.size + raw_w.size)
            print(f"  Tile {tile_num}/{total_tiles}: M={m} N={n} K={k}  "
                  f"Config={reason}  Sparsity={tile_sparsity*100:.1f}%")

            hw = run_jtag_inference(m, n, k, s_data, s_weight, m_idx, n_idx, config=config)
            if hw is None:
                print(f"    TIMEOUT — skipping")
                continue

            sw       = sw_reference(s_data, s_weight, config)
            hw_dense = hw[np.ix_(m_idx, n_idx)]
            match    = np.array_equal(sw, hw_dense)

            if match:
                match_count += 1
                print(f"    MATCH ✓")
            else:
                diff = sw - hw_dense
                print(f"    MISMATCH — {np.count_nonzero(diff)} non-zero diffs")
                print(f"    SW sample: {sw[:2, :4]}")
                print(f"    HW sample: {hw_dense[:2, :4]}")
                print(f"    Diff sample: {diff[:2, :4]}")

            layer_results.append({
                "tile":     tile_num,
                "m": m, "n": n, "k": k,
                "config":   config,
                "sa_size":  sa_size,
                "sparsity": tile_sparsity,
                "match":    match,
            })

    print(f"\n  Layer summary: {match_count}/{len(layer_results)} tiles matched")
    print(f"  Int8 8x8: {int8_count}  Int16 16x16: {int16_16_count}  Int16 32x32: {int16_32_count}")
    return layer_results

def preprocess_image(path):
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img = Image.open(path).convert('RGB')
    return preprocess(img).unsqueeze(0).numpy()

def run_single_tile(model_name="alexnet", image_path=IMAGE_PATH, layer_idx=0, tile_row=0, tile_col=0, t_size=32):
    print(f"\nSingle tile test: model={model_name} layer={layer_idx} tile=({tile_row},{tile_col})")
    ov_model, core = get_or_convert_model(model_name)
    conv_layers    = discover_conv_layers(ov_model)
    if layer_idx >= len(conv_layers):
        print(f"ERROR: layer_idx {layer_idx} out of range")
        return
    layer      = conv_layers[layer_idx]
    input_data = preprocess_image(image_path)
    print("\nExtracting activations...")
    activations = extract_all_activations(ov_model, core, input_data, [layer])
    name = layer["name"]
    if name not in activations:
        print(f"ERROR: no activation for {name}")
        return
    activation = activations[name]
    weights_2d = extract_weights(layer)
    act_2d     = im2col(activation, layer["kernel_h"], layer["kernel_w"],
                        layer["strides"], layer["pads"][0] if len(layer["pads"]) == 1 else layer["pads"])
    total_rows = act_2d.shape[0]
    total_cols = weights_2d.shape[0]
    r_start    = tile_row * t_size
    c_start    = tile_col * t_size
    r_end      = min(r_start + t_size, total_rows)
    c_end      = min(c_start + t_size, total_cols)
    raw_d      = act_2d[r_start:r_end, :t_size]
    raw_w      = weights_2d[c_start:c_end, :t_size].T
    raw_d_q8   = quantize_to_int8(raw_d)
    result     = coordinated_row_removal(raw_d_q8, raw_w)
    if result is None:
        print("Tile is all zeros after stripping")
        return
    s_data, s_weight, m, n, k, m_idx, n_idx = result
    config, sa_size, reason = decide_config(m, n, k)
    print(f"M={m} N={n} K={k}  Config={reason}")
    print(f"Sparsity: {(1.0 - (m*k + k*n)/(raw_d.size + raw_w.size))*100:.1f}%")
    hw = run_jtag_inference(m, n, k, s_data, s_weight, m_idx, n_idx, config=config)
    if hw is None:
        print("TIMEOUT")
        return
    sw       = sw_reference(s_data, s_weight, config)
    hw_dense = hw[np.ix_(m_idx, n_idx)]
    print(f"\nHARDWARE:\n{hw_dense}")
    print(f"\nSOFTWARE:\n{sw}")
    if np.array_equal(sw, hw_dense):
        print("\nSTATUS: BIT-ACCURATE MATCH")
    else:
        diff = sw - hw_dense
        print(f"\nSTATUS: MISMATCH — {np.count_nonzero(diff)} non-zero diffs")
        print(diff[:5, :5])

def main(model_name="alexnet", image_path=IMAGE_PATH, t_size=32):
    print(f"\n{'='*60}")
    print(f"NPU OpenVINO Pipeline")
    print(f"Model: {model_name}  Image: {os.path.basename(image_path)}  Tile size: {t_size}")
    print(f"{'='*60}")

    ov_model, core = get_or_convert_model(model_name)
    conv_layers    = discover_conv_layers(ov_model)
    input_data     = preprocess_image(image_path)
    print(f"\nInput shape: {input_data.shape}")

    print("\nRunning inference to extract activations...")
    activations = extract_all_activations(ov_model, core, input_data, conv_layers)
    print(f"Extracted activations for {len(activations)} layers")

    all_results = {}
    for i, layer in enumerate(conv_layers):
        name = layer["name"]
        if name not in activations:
            print(f"\nLayer [{i}] {name}: no activation captured, skipping")
            continue
        layer_results     = run_layer(i, layer, activations[name], t_size)
        all_results[name] = layer_results

    print(f"\n{'='*60}")
    print(f"FINAL SUMMARY — {model_name} on {os.path.basename(image_path)}")
    print(f"{'='*60}")
    total_tiles    = sum(len(v) for v in all_results.values())
    total_match    = sum(sum(1 for t in v if t["match"]) for v in all_results.values())
    total_int8     = sum(sum(1 for t in v if t["config"] == 1) for v in all_results.values())
    total_int16_16 = sum(sum(1 for t in v if t["config"] == 0 and t["sa_size"] == 16) for v in all_results.values())
    total_int16_32 = sum(sum(1 for t in v if t["config"] == 0 and t["sa_size"] == 32) for v in all_results.values())
    avg_sparsity   = np.mean([t["sparsity"] for v in all_results.values() for t in v]) if total_tiles > 0 else 0

    print(f"Total tiles processed : {total_tiles}")
    print(f"Bit-accurate matches  : {total_match}/{total_tiles}")
    print(f"Int8  8x8  tiles      : {total_int8}")
    print(f"Int16 16x16 tiles     : {total_int16_16}")
    print(f"Int16 32x32 tiles     : {total_int16_32}")
    print(f"Average tile sparsity : {avg_sparsity*100:.1f}%")

if __name__ == "__main__":
    run_single_tile(model_name="alexnet", image_path=IMAGE_PATH, layer_idx=0, tile_row=1, tile_col=0, t_size=32)
    #main(model_name="alexnet", image_path=IMAGE_PATH, t_size=32)