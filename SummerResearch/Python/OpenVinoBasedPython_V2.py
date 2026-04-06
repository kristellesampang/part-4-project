import openvino as ov
import torch
import torchvision.models as models
import torch.nn.functional as F
import torch.nn.utils.prune as prune
import numpy as np
import os
import time
import csv
from PIL import Image
from torchvision import transforms

# paths to the test image and where the converted OpenVINO models get saved
IMAGE_PATH = "C:/Users/pchh520/Documents/GitHub/part-4-project/SummerResearch/Python/cat.jpg"
MODEL_DIR  = "C:/Users/pchh520/Documents/GitHub/part-4-project/SummerResearch/Python/ov_models"

# the four models supported, paired with their expected input shape (batch=1, RGB, 224x224)
SUPPORTED_MODELS = {
    "alexnet":    (models.alexnet,    (1, 3, 224, 224)),
    "vgg16":      (models.vgg16,      (1, 3, 224, 224)),
    "resnet50":   (models.resnet50,   (1, 3, 224, 224)),
    "mobilenetv2":(models.mobilenet_v2,(1, 3, 224, 224)),
}

def get_or_convert_model(model_name, pruned=False, prune_amount=0.5):
    # make sure the output folder exists before anything gets written into it
    os.makedirs(MODEL_DIR, exist_ok=True)

    # the suffix distinguishes pruned and unpruned versions so they don't overwrite each other
    suffix   = f"_pruned{int(prune_amount*100)}" if pruned else ""
    xml_path = os.path.join(MODEL_DIR, f"{model_name}{suffix}.xml")

    # conversion takes a while, so skip it if the IR files are already on disk
    if not os.path.exists(xml_path):
        print(f"Converting {model_name}{suffix} to OpenVINO IR...")
        model_fn, input_shape = SUPPORTED_MODELS[model_name]

        # load pretrained ImageNet weights and switch to eval mode (disables dropout, batchnorm tracking, etc.)
        torch_model = model_fn(pretrained=True).eval()

        if pruned:
            # structured L2 pruning across output channels (dim=0) for every conv layer
            # prune.remove bakes the mask into the actual weights permanently — no sparse wrapper left over
            for _, module in torch_model.named_modules():
                if isinstance(module, torch.nn.Conv2d):
                    prune.ln_structured(module, name='weight', amount=prune_amount, n=2, dim=0)
                    prune.remove(module, 'weight')
            print(f"  Applied structured pruning ({int(prune_amount*100)}% filters zeroed)")

        # OpenVINO needs a real tensor to trace the graph, so a random dummy input does the job
        dummy    = torch.randn(*input_shape)
        ov_model = ov.convert_model(torch_model, example_input=dummy)
        ov.save_model(ov_model, xml_path)
        print(f"Saved to {xml_path}")

    # load from disk regardless — same path whether just converted or already existed
    core     = ov.Core()
    ov_model = core.read_model(xml_path)
    return ov_model, core


def discover_conv_layers(ov_model):
    # walk the computation graph and collect every Convolution node with its metadata
    conv_layers = []
    for op in ov_model.get_ops():
        if op.get_type_name() != "Convolution":
            continue

        attrs        = op.get_attributes()
        strides      = attrs["strides"]
        pads         = attrs["pads_begin"]
        dilations    = attrs["dilations"]

        # input(1) is the weight tensor — follow the edge back to the constant node holding the actual data
        weight_node  = op.input(1).get_source_output().get_node()
        weight_shape = weight_node.get_output_shape(0)  # (out_ch, in_ch, kH, kW)

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

    # print a readable summary before tiling starts — good sanity check
    print(f"\nDiscovered {len(conv_layers)} conv layers:")
    for i, l in enumerate(conv_layers):
        print(f"  [{i}] {l['name']}")
        print(f"      Filters: {l['out_channels']} x ({l['in_channels']} x {l['kernel_h']} x {l['kernel_w']})")
        print(f"      Stride: {l['strides']}  Pad: {l['pads']}  Dilation: {l['dilations']}")

    return conv_layers


def extract_all_activations(ov_model, core, input_data, conv_layers):
    # clone the model first so the original stays untouched
    augmented  = ov_model.clone()
    name_to_op = {op.get_friendly_name(): op for op in augmented.get_ops()}

    # tap each conv layer's output by adding it as an extra model output
    # this way a single forward pass captures all intermediate activations at once
    new_outputs = []
    for layer in conv_layers:
        op = name_to_op.get(layer["name"])
        if op is not None:
            new_outputs.append(op.output(0))
    augmented.add_outputs(new_outputs)

    # compile and run inference once
    compiled      = core.compile_model(augmented, "CPU")
    infer_request = compiled.create_infer_request()
    infer_request.infer({0: input_data})

    # the original model outputs (e.g. class scores) come first, then the tapped activations
    n_original  = len(ov_model.outputs)
    activations = {}
    for i, layer in enumerate(conv_layers):
        out_idx = n_original + i
        if out_idx < len(compiled.outputs):
            # apply ReLU here since conv layers are almost always followed by it
            activations[layer["name"]] = np.maximum(0, infer_request.get_output_tensor(out_idx).data.copy())

    return activations


def extract_weights(layer):
    # trace back through the graph to get the raw weight tensor from the constant node
    weight_node = layer["weight_node"]
    const_node  = weight_node.input(0).get_source_output().get_node()
    weights     = const_node.data

    # flatten each filter into a row — shape becomes (out_ch, in_ch * kH * kW)
    weights_2d  = weights.reshape(weights.shape[0], -1)

    # per-filter max-abs scaling into int8 — each filter gets its own scale factor
    # all-zero filters (from pruning) get scale=1 to avoid divide-by-zero
    max_per_filter = np.max(np.abs(weights_2d), axis=1, keepdims=True)
    max_per_filter = np.where(max_per_filter == 0, 1.0, max_per_filter)
    weights_int8   = np.clip(np.round(weights_2d / max_per_filter * 127), -128, 127).astype(np.int8)

    return weights_int8


def im2col(activation, kernel_h, kernel_w, strides, pads):
    # im2col converts a convolution into a standard matrix multiply
    # F.unfold extracts all overlapping kernel-sized patches and lays them out as columns
    # result shape: (num_patches, in_ch * kH * kW)
    t = torch.from_numpy(activation.astype(np.float32))
    unfolded = F.unfold(t, kernel_size=(kernel_h, kernel_w), stride=strides, padding=pads)
    return unfolded.squeeze(0).transpose(0, 1).numpy()


def quantize_to_int8(matrix):
    # symmetric int8 quantization — scale so the largest absolute value maps to 127
    max_val = np.max(np.abs(matrix))
    if max_val == 0:
        return matrix.astype(np.int8)
    scale = 127.0 / max_val
    return np.clip(np.round(matrix * scale), -128, 127).astype(np.int8)


def compute_sparsity(matrix):
    # fraction of zeros in the matrix — 0.0 = fully dense, 1.0 = fully sparse
    return np.sum(matrix == 0) / matrix.size


def decide_config(m, n, k):
    # routes the tile to the right hardware mode based on stripped dimensions
    # small tiles (m and n both <= 16) fit in the int8 16x16 systolic array (config=1)
    # anything larger goes to the int16 32x32 array (config=0)
    if m <= 16 and n <= 16:
        return 1, 16, "Int8 16x16"
    else:
        return 0, 32, "Int16 32x32"


def coordinated_row_removal(data_matrix, weight_matrix):
    data_matrix   = np.array(data_matrix)
    weight_matrix = np.array(weight_matrix)

    # find which rows in data and columns in weights have at least one non-zero entry
    active_m  = np.where(np.any(data_matrix,   axis=1))[0]
    active_n  = np.where(np.any(weight_matrix, axis=0))[0]

    # K dimension: only keep indices that are non-zero in BOTH matrices
    # a K index that's zero in either side contributes nothing to the output
    data_k    = set(np.where(np.any(data_matrix,   axis=0))[0])
    weight_k  = set(np.where(np.any(weight_matrix, axis=1))[0])
    common_k  = sorted(list(data_k & weight_k))

    # if everything got stripped out, skip the tile entirely
    if not common_k or len(active_m) == 0 or len(active_n) == 0:
        return None

    # extract the compact dense submatrix for just the non-zero rows/cols
    compact_data   = data_matrix[np.ix_(active_m, common_k)]
    compact_weight = weight_matrix[np.ix_(common_k, active_n)]

    # return the original index arrays too — needed to scatter results back into the full output
    return compact_data, compact_weight, len(active_m), len(active_n), len(common_k), active_m, active_n


def run_jtag_inference(m, n, k, s_data, s_weight, m_idx, n_idx, config=0):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    bin_file = os.path.join(current_dir, "tile.bin")
    res_file = os.path.join(current_dir, "result.bin")

    # remove any stale result file so old data doesn't get picked up accidentally
    if os.path.exists(res_file):
        os.remove(res_file)

    # pack the tile: 4-byte header (m, n, k, config) followed by the raw matrix data
    # int8 for the small array, int16 for the larger one
    dtype   = np.int8 if config == 1 else np.int16
    header  = bytes([int(m), int(n), int(k), int(config)])
    payload = s_data.astype(dtype).tobytes() + s_weight.astype(dtype).tobytes()

    # retry the file write up to 10 times in case the file is briefly locked by another process
    for attempt in range(10):
        try:
            with open(bin_file, "wb") as f:
                f.write(header + payload)
            break
        except PermissionError:
            time.sleep(0.2)
    else:
        print("  ERROR: Could not write tile.bin after 10 attempts")
        return None, -1

    # the TCL/JTAG side writes back m*n int32 results + one extra int32 cycle count at the end
    expected_bytes = (m * n + 1) * 4

    # poll for result.bin to appear and verify it's fully written before reading
    start = time.time()
    while True:
        if time.time() - start > 30:
            print("  ERROR: Timeout waiting for result.bin")
            return None, -1
        if os.path.exists(res_file):
            sz = os.path.getsize(res_file)
            if sz == expected_bytes:
                # double-check the size hasn't changed — confirms the write is done
                time.sleep(0.02)
                if os.path.getsize(res_file) == sz:
                    break
        time.sleep(0.05)

    # read the result file and clean it up — retry if still briefly locked
    for _ in range(20):
        try:
            raw_res = np.fromfile(res_file, dtype='<i4')
            if raw_res.size != m * n + 1:
                time.sleep(0.05)
                continue
            os.remove(res_file)
            break
        except (PermissionError, OSError):
            time.sleep(0.05)
    else:
        print("  ERROR: Could not read/delete result.bin")
        return None, -1

    # last word is cycle count appended by TCL
    cycles  = int(raw_res[-1])
    raw_res = raw_res[:-1]

    # hardware only computed the stripped submatrix, so scatter those results back into
    # a full 32x32 output using the original row/col indices (everything else stays zero)
    sparse_out = np.zeros((32, 32), dtype=np.int32)
    dense_res  = raw_res.reshape(m, n)
    for i, orig_row in enumerate(m_idx):
        for j, orig_col in enumerate(n_idx):
            sparse_out[orig_row, orig_col] = dense_res[i, j]

    return sparse_out, cycles


def sw_reference(s_data, s_weight, config):
    # software golden reference for bit-accuracy checking against hardware results
    # int8 mode: accumulate in int32 — no overflow risk at these sizes
    # int16 mode: accumulate in int64 first, then wrap to int32 to match hardware overflow behavior
    if config == 1:
        full = np.matmul(s_data.astype(np.int32), s_weight.astype(np.int32))
        return full.astype(np.int32)
    else:
        full = np.matmul(s_data.astype(np.int64), s_weight.astype(np.int64))
        return ((full + 2**31) % 2**32 - 2**31).astype(np.int32)


def run_layer(layer_idx, layer, activation, t_size, model_name):
    name     = layer["name"]
    kernel_h = layer["kernel_h"]
    kernel_w = layer["kernel_w"]
    strides  = layer["strides"]
    pads     = layer["pads"]

    print(f"\n{'='*60}")
    print(f"Layer [{layer_idx}]: {name}")
    print(f"  Kernel: {kernel_h}x{kernel_w}  Stride: {strides}  Pad: {pads}")
    print(f"  Output channels: {layer['out_channels']}  Input channels: {layer['in_channels']}")

    # get the weight matrix (out_ch x kernel_elements) and the im2col activation matrix
    weights_2d = extract_weights(layer)
    act_2d     = im2col(activation, kernel_h, kernel_w, strides, pads[0] if len(pads) == 1 else pads)

    total_rows = act_2d.shape[0]
    total_cols = weights_2d.shape[0]

    print(f"  Activation shape: {act_2d.shape}  Weight shape: {weights_2d.shape}")

    act_sparsity = compute_sparsity(act_2d)
    w_sparsity   = compute_sparsity(weights_2d)
    print(f"  Activation sparsity: {act_sparsity*100:.1f}%  Weight sparsity: {w_sparsity*100:.1f}%")

    # ceiling division to get the number of tiles in each dimension
    num_tiles_row = max(1, (total_rows + t_size - 1) // t_size)
    num_tiles_col = max(1, (total_cols + t_size - 1) // t_size)
    total_tiles   = num_tiles_row * num_tiles_col
    print(f"  Tile size: {t_size}  Total tiles: {total_tiles}")

    layer_results  = []
    tile_num       = 0
    int8_count     = 0
    int16_32_count = 0
    match_count    = 0

    # slide the tile window across the M dimension (activation output patches)
    for tr in range(0, total_rows, t_size):
        r_start  = tr
        r_end    = min(r_start + t_size, total_rows)
        orig_m   = r_end - r_start

        # slice out this row-tile and quantize it to int8
        raw_d    = act_2d[r_start:r_end, :t_size]
        raw_d_q8 = quantize_to_int8(raw_d)

        # slide across the N dimension (output filters)
        for tc in range(0, total_cols, t_size):
            tile_num += 1
            c_start  = tc
            c_end    = min(c_start + t_size, total_cols)
            orig_n   = c_end - c_start
            orig_k   = min(t_size, act_2d.shape[1], weights_2d.shape[1])

            # transpose the weight tile to get it into (K x N) layout for the matmul
            raw_w    = weights_2d[c_start:c_end, :t_size].T
            raw_w_q8 = raw_w

            # strip all-zero rows and columns before sending to hardware
            result = coordinated_row_removal(raw_d_q8, raw_w_q8)
            if result is None:
                print(f"  Tile {tile_num}/{total_tiles}: skipped (all zeros)")
                continue

            s_data_t, s_weight_t, m, n, k, m_idx, n_idx = result
            config, sa_size, reason = decide_config(m, n, k)

            if config == 1:
                int8_count += 1
            else:
                int16_32_count += 1

            # sparsity = how much of the raw tile got skipped after row removal
            tile_sparsity = 1.0 - (m * k + k * n) / (raw_d.size + raw_w.size)
            orig_macs     = 2 * orig_m * orig_n * orig_k
            eff_macs      = 2 * m * n * k
            mac_reduction = 1.0 - (eff_macs / orig_macs) if orig_macs > 0 else 0.0

            # CPU baseline: time the raw dense matmul (no sparsity) as a reference point
            cpu_d = raw_d_q8[:, :orig_k].astype(np.int32)
            cpu_w = raw_w_q8[:orig_k, :].astype(np.int32)
            t0 = time.perf_counter()
            _ = cpu_d @ cpu_w
            cpu_us = (time.perf_counter() - t0) * 1e6

            print(f"  Tile {tile_num}/{total_tiles}: M={m} N={n} K={k}  "
                  f"Config={reason}  Sparsity={tile_sparsity*100:.1f}%  "
                  f"MAC reduction={mac_reduction*100:.1f}%  CPU={cpu_us:.1f}us")

            # send the stripped tile to hardware via JTAG and wait for the result
            hw, cycles = run_jtag_inference(m, n, k, s_data_t, s_weight_t, m_idx, n_idx, config=config)
            if hw is None:
                print(f"    TIMEOUT — skipping")
                continue

            # compare hardware output against the software reference
            sw       = sw_reference(s_data_t, s_weight_t, config)
            hw_dense = hw[np.ix_(m_idx, n_idx)]
            match    = np.array_equal(sw, hw_dense)

            if match:
                match_count += 1
                print(f"    MATCH ✓  cycles={cycles}")
            else:
                diff = sw - hw_dense
                print(f"    MISMATCH — {np.count_nonzero(diff)} non-zero diffs  cycles={cycles}")
                print(f"    SW sample: {sw[:2, :4]}")
                print(f"    HW sample: {hw_dense[:2, :4]}")
                print(f"    Diff sample: {diff[:2, :4]}")

            layer_results.append({
                "model":         model_name,
                "layer":         name,
                "layer_idx":     layer_idx,
                "tile":          tile_num,
                "orig_m":        orig_m,
                "orig_n":        orig_n,
                "orig_k":        orig_k,
                "m":             m,
                "n":             n,
                "k":             k,
                "config":        config,
                "sa_size":       sa_size,
                "sparsity":      round(tile_sparsity, 4),
                "orig_macs":     orig_macs,
                "eff_macs":      eff_macs,
                "mac_reduction": round(mac_reduction, 4),
                "cycles":        cycles,
                "cpu_us":        round(cpu_us, 2),
                "match":         match,
            })

    print(f"\n  Layer summary: {match_count}/{len(layer_results)} tiles matched")
    print(f"  Int8 16x16: {int8_count}  Int16 32x32: {int16_32_count}")
    return layer_results


def preprocess_image(path):
    # standard ImageNet preprocessing pipeline
    # resize to 256 then center-crop to 224 to match what the models were trained on
    # normalize with ImageNet mean and std
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img = Image.open(path).convert('RGB')
    return preprocess(img).unsqueeze(0).numpy()  # unsqueeze adds the batch dimension


def get_csv_path(model_name, image_path, pruned=False, prune_amount=0.5):
    # build a unique CSV filename based on model, pruning config, and image name
    # placed next to this script so it's easy to find
    current_dir = os.path.dirname(os.path.abspath(__file__))
    suffix = f"_pruned{int(prune_amount*100)}" if pruned else ""
    return os.path.join(current_dir, f"results_{model_name}{suffix}_{os.path.splitext(os.path.basename(image_path))[0]}.csv")


def append_layer_csv(csv_path, layer_tiles, write_header=False):
    # appends tile results for one layer to the CSV
    # write_header is only True on the very first layer call so column names appear once at the top
    fieldnames = [
        "model", "layer", "layer_idx", "tile",
        "orig_m", "orig_n", "orig_k",
        "m", "n", "k",
        "config", "sa_size",
        "sparsity", "orig_macs", "eff_macs", "mac_reduction",
        "cycles", "cpu_us", "match"
    ]
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        for t in layer_tiles:
            writer.writerow(t)


def export_csv(all_results, model_name, image_path, pruned=False, prune_amount=0.5):
    # bulk export of all results at once — overwrites any existing file
    # this is the "dump everything at the end" version; append_layer_csv is the incremental one
    current_dir = os.path.dirname(os.path.abspath(__file__))
    suffix   = f"_pruned{int(prune_amount*100)}" if pruned else ""
    csv_path = os.path.join(current_dir, f"results_{model_name}{suffix}_{os.path.splitext(os.path.basename(image_path))[0]}.csv")
    fieldnames = [
        "model", "layer", "layer_idx", "tile",
        "orig_m", "orig_n", "orig_k",
        "m", "n", "k",
        "config", "sa_size",
        "sparsity", "orig_macs", "eff_macs", "mac_reduction",
        "cycles", "cpu_us", "match"
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for tiles in all_results.values():
            for t in tiles:
                writer.writerow(t)
    print(f"\nCSV saved to: {csv_path}")
    return csv_path


def run_single_tile(model_name="alexnet", image_path=IMAGE_PATH, layer_idx=0, tile_row=0, tile_col=0, t_size=32, pruned=False, prune_amount=0.5):
    # debug helper — tests one specific tile from one specific layer without running the whole model
    print(f"\nSingle tile test: model={model_name} layer={layer_idx} tile=({tile_row},{tile_col}) pruned={pruned}")

    ov_model, core = get_or_convert_model(model_name, pruned=pruned, prune_amount=prune_amount)
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

    # compute start/end indices for the requested tile position
    r_start = tile_row * t_size
    c_start = tile_col * t_size
    r_end   = min(r_start + t_size, total_rows)
    c_end   = min(c_start + t_size, total_cols)

    raw_d    = act_2d[r_start:r_end, :t_size]
    raw_w    = weights_2d[c_start:c_end, :t_size].T
    raw_d_q8 = quantize_to_int8(raw_d)

    result = coordinated_row_removal(raw_d_q8, raw_w)
    if result is None:
        print("Tile is all zeros after stripping")
        return

    s_data_t, s_weight_t, m, n, k, m_idx, n_idx = result
    config, sa_size, reason = decide_config(m, n, k)

    print(f"M={m} N={n} K={k}  Config={reason}")
    print(f"Sparsity: {(1.0 - (m*k + k*n)/(raw_d.size + raw_w.size))*100:.1f}%")

    hw, cycles = run_jtag_inference(m, n, k, s_data_t, s_weight_t, m_idx, n_idx, config=config)
    if hw is None:
        print("TIMEOUT")
        return

    sw       = sw_reference(s_data_t, s_weight_t, config)
    hw_dense = hw[np.ix_(m_idx, n_idx)]

    print(f"\nHARDWARE:\n{hw_dense}")
    print(f"\nSOFTWARE:\n{sw}")
    print(f"\nCycles: {cycles}")

    if np.array_equal(sw, hw_dense):
        print("\nSTATUS: BIT-ACCURATE MATCH")
    else:
        diff = sw - hw_dense
        print(f"\nSTATUS: MISMATCH — {np.count_nonzero(diff)} non-zero diffs")
        print(diff[:5, :5])


def main(model_name="alexnet", image_path=IMAGE_PATH, t_size=32, pruned=False, prune_amount=0.5, start_layer=0):
    print(f"\n{'='*60}")
    print(f"NPU OpenVINO Pipeline")
    print(f"Model: {model_name}  Image: {os.path.basename(image_path)}  Tile size: {t_size}  Pruned: {pruned}")
    print(f"{'='*60}")

    ov_model, core = get_or_convert_model(model_name, pruned=pruned, prune_amount=prune_amount)
    conv_layers    = discover_conv_layers(ov_model)
    input_data     = preprocess_image(image_path)
    print(f"\nInput shape: {input_data.shape}")

    # run the full model once up front to get all intermediate activations in one shot
    print("\nRunning inference to extract activations...")
    activations = extract_all_activations(ov_model, core, input_data, conv_layers)
    print(f"Extracted activations for {len(activations)} layers")

    # if the CSV already exists, append to it; otherwise start fresh with a header
    csv_path     = get_csv_path(model_name, image_path, pruned, prune_amount)
    write_header = not os.path.exists(csv_path)
    all_results  = {}

    for i, layer in enumerate(conv_layers):
        # start_layer allows resuming from a specific layer if a previous run was cut short
        if i < start_layer:
            continue

        name = layer["name"]
        if name not in activations:
            print(f"\nLayer [{i}] {name}: no activation captured, skipping")
            continue

        layer_results     = run_layer(i, layer, activations[name], t_size, model_name)
        all_results[name] = layer_results

        # write results after each layer so nothing is lost if the script crashes or gets killed
        append_layer_csv(csv_path, layer_results, write_header=write_header)
        write_header = False  # header only goes in once

    # aggregate stats across all layers
    print(f"\n{'='*60}")
    print(f"FINAL SUMMARY — {model_name} on {os.path.basename(image_path)}")
    print(f"{'='*60}")

    total_tiles    = sum(len(v) for v in all_results.values())
    total_match    = sum(sum(1 for t in v if t["match"]) for v in all_results.values())
    total_int8     = sum(sum(1 for t in v if t["config"] == 1) for v in all_results.values())
    total_int16_32 = sum(sum(1 for t in v if t["config"] == 0) for v in all_results.values())
    avg_sparsity   = np.mean([t["sparsity"] for v in all_results.values() for t in v]) if total_tiles > 0 else 0
    total_orig_macs = sum(t["orig_macs"] for v in all_results.values() for t in v)
    total_eff_macs  = sum(t["eff_macs"]  for v in all_results.values() for t in v)
    overall_mac_red = 1.0 - (total_eff_macs / total_orig_macs) if total_orig_macs > 0 else 0

    print(f"Total tiles processed  : {total_tiles}")
    print(f"Bit-accurate matches   : {total_match}/{total_tiles}")
    print(f"Int8  16x16 tiles      : {total_int8}")
    print(f"Int16 32x32 tiles      : {total_int16_32}")
    print(f"Average tile sparsity  : {avg_sparsity*100:.1f}%")
    print(f"Overall MAC reduction  : {overall_mac_red*100:.1f}%")

    print(f"\nCSV saved to: {csv_path}")


if __name__ == "__main__":
    #run_single_tile(model_name="alexnet", image_path=IMAGE_PATH, layer_idx=4, tile_row=0, tile_col=0, t_size=16)
    main(model_name="vgg16", image_path=IMAGE_PATH, t_size=32, pruned=True, prune_amount=0.45, start_layer=8)
