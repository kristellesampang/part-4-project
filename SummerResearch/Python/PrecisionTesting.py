
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
import serial
import subprocess
import pandas as pd
 

# --- Constants ---
IMAGE_PATH = "C:/Users/pchh520/Documents/GitHub/part-4-project/SummerResearch/Python/cat.jpg"
# IMAGE_PATH = 'C:/Users/iamkr/Documents/part-4-project/Final/Python/hand_xray.jpg'
# IMAGE_PATH = 'C:/Users/iamkr/Documents/part-4-project/Final/Python/patella_alta.jpg'
# MIF_OUTPUT_DIR = "C:/Users/iamkr/Documents/part-4-project/Final/mif/pipeline_v2"
MIF_OUTPUT_DIR = "C:/Users/pchh520/Documents/GitHub/part-4-project/SummerResearch/Python/mif_results"
# TEST_DATA_MIF_DIR = 'C:/Users/iamkr/Documents/part-4-project/Final/testing/v2_alexnet/run_1/tile_1/activation_tile_1.mif'
# TEST_WEIGHT_MIF_DIR = 'C:/Users/iamkr/Documents/part-4-project/Final/testing/v2_alexnet/run_1/tile_1/weight_tile_1.mif'
STRIPPED_DATA_MIF_DIR = 'C:/Users/iamkr/Documents/part-4-project/Final/testing/v2_alexnet/run_2/tile_1/stripped_activation.mif'
STRIPPED_WEIGHT_MIF_DIR = 'C:/Users/iamkr/Documents/part-4-project/Final/testing/v2_alexnet/run_2/tile_1/stripped_weight.mif'
LAYER_SIZE = 64
TILE_SIZE = 8


# Quantise Alexnet to int8
class QuantizableAlexNet(torch.nn.Module):
    """A wrapper class to make AlexNet quantizable."""
    
    # Initialize with a pre-trained AlexNet model
    def __init__(self, model_fp32):
        super(QuantizableAlexNet, self).__init__() 
        self.quant = torch.quantization.QuantStub() 
        self.dequant = torch.quantization.DeQuantStub()
        self.model_fp32 = model_fp32 # Pre-trained AlexNet model

    # Forward pass with options to return convolutional outputs and apply im2col
    def forward(self, x, return_conv=False, im2col=False):
        x = self.quant(x) # Quantize the input
        conv_out = self.model_fp32.features(x) # Feature extractor (convolutional output)
        if return_conv:
            # Return convolutional activations (4D: N, C, H, W)
            if im2col:
                # Convert activations to 2D using unfold (im2col)
                # im2col: each column is a patch for the next layer
                # Example: kernel_size=3, stride=1, padding=1
                unfolded = torch.nn.functional.unfold(conv_out, kernel_size=3, stride=1, padding=1)
                # unfolded shape: (N, C*kernel_size*kernel_size, L) -> transpose to (L, C*ks*ks)
                return unfolded
            return conv_out
        x = self.model_fp32.avgpool(conv_out) # Average pooling
        x = torch.flatten(x, 1) # Flatten the tensor
        x = self.model_fp32.classifier(x) # Classifier
        x = self.dequant(x) # Dequantize the output
        return x

    def get_conv_weights(self, layer_idx=0):
        # Get weights of a convolutional layer in 4D (out_channels, in_channels, kH, kW)
        conv_layer = [m for m in self.model_fp32.features if isinstance(m, torch.nn.Conv2d)][layer_idx]
        return conv_layer.weight.data
    
# Retrieve ImageNet Labels
def get_imagenet_labels():
    """Downloads and loads the ImageNet class labels."""
    labels_path = 'imagenet_class_index.json'
    labels_url = 'https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json'

    if not os.path.exists(labels_path):
        print(f"Downloading ImageNet labels from {labels_url}...")
        try:
            response = requests.get(labels_url)
            response.raise_for_status()
            with open(labels_path, 'w') as f:
                json.dump(response.json(), f)
            print("Labels downloaded.")
        except requests.exceptions.RequestException as e:
            print(f"Error downloading labels: {e}")
            sys.exit(1)

    with open(labels_path) as f:
        labels = json.load(f)
    return labels

# Load and Quantise AlexNet Model
def load_quantized_alexnet():
    """Loads and quantizes a pre-trained AlexNet model."""
    model_fp32 = models.alexnet(pretrained=True) # Load pre-trained AlexNet
    model_fp32.eval() # Set to evaluation mode

    # Fuse Conv, ReLU, and MaxPool layers for quantization
    modules_to_fuse = [['0', '1'], ['3', '4'], ['6', '7'], ['8', '9'], ['10', '11']]
    torch.quantization.fuse_modules(model_fp32.features, modules_to_fuse, inplace=True)


    model = QuantizableAlexNet(model_fp32)
    model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    torch.quantization.prepare(model, inplace=True)

    # Calibrate with a dummy input
    model(torch.randn(1, 3, 224, 224))

    torch.quantization.convert(model, inplace=True)
    return model

# Preprocess the Input Image
def preprocess_image(image_path):
    """Preprocesses the input image for AlexNet."""
    preprocess = transforms.Compose([
        transforms.Resize(256), # Resize to 256x256
        transforms.CenterCrop(224), # Center crop to 224x224
        transforms.ToTensor(), # Convert to tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # Normalize
    ])
    img = Image.open(image_path).convert('RGB') # Open and convert image to RGB
    img_t = preprocess(img) # Apply preprocessing
    img_t = img_t.unsqueeze(0) # Add batch dimension
    return img_t


def extract_conv_weights_and_activations(model, input_tensor, conv_idx, relu_idx):
    """Extracts quantized weights and activations from a convolutional layer."""
    # 1. Extract quantized weights
    conv_layer = model.model_fp32.features[conv_idx]
    quantized_weight_tensor = conv_layer.weight() if callable(conv_layer.weight) else conv_layer.weight
    weights_int8 = quantized_weight_tensor.int_repr().cpu().numpy()
    weights_2d = weights_int8.reshape(weights_int8.shape[0], -1)

    # 2. Extract activations after ReLU using a forward hook
    activation = {}
    def get_quantized_activation(name):
        def hook(module, input, output):
            activation[name] = output
        return hook

    # The hook captures the output of the ReLU layer during the forward pass
    # and stores it in the 'activation' dictionary.
    # This allows us to access the quantized activation values later.
    # Register the hook on the specified ReLU layer
    relu_layer = model.model_fp32.features[relu_idx]
    hook_handle = relu_layer.register_forward_hook(get_quantized_activation('relu'))
    with torch.no_grad():
        model(input_tensor)
    hook_handle.remove()
    # Convert the quantized activation tensor to int8 numpy array
    quantized_activation_tensor = activation['relu']
    activations_int8 = quantized_activation_tensor.int_repr().cpu().numpy()
        
    # Save as a 2D matrix using im2col
    # Convert to float32 for unfold (im2col), then back to int8 for hardware
    if hasattr(quantized_activation_tensor, 'dequantize'):
        activations_float = quantized_activation_tensor.dequantize().cpu()
    else:
        activations_float = torch.tensor(activations_int8, dtype=torch.float32).unsqueeze(0)
    activations_unfolded = F.unfold(
        activations_float,
        kernel_size=3,
        stride=1,
        padding=1
    )
    activations_2d = activations_unfolded.squeeze(0).transpose(0, 1).numpy().astype(np.int8)

    print(f"Conv weights shape: {weights_2d.shape}")
    print(f"Activation shape: {activations_2d.shape}")
    return weights_2d, activations_2d


# Save 2D Matrix to MIF File
def save_matrix_to_mif(matrix, filename, depth, width, m, n, k):
    """Saves a 2D numpy array to a Memory Initialization File (MIF)."""
    
    depth += 3; # extra 3 bits for m, n, k
    
    with open(filename, 'w') as f:
        f.write(f"WIDTH = {width};\n")
        f.write(f"DEPTH = {depth};\n")
        f.write("ADDRESS_RADIX = UNS;\n")
        f.write("DATA_RADIX = HEX;\n")
        f.write("\nCONTENT BEGIN\n")
        
        flat_matrix = matrix.flatten()
        
        f.write(f"\t0\t:\t{m:02X};\n")
        f.write(f"\t1\t:\t{n:02X};\n")
        f.write(f"\t2\t:\t{k:02X};\n")
        for i, val in enumerate(flat_matrix):
            hex_val = f"{val:02X}"
            f.write(f"\t{(i+3)}\t:\t{hex_val};\n")
        f.write("END;\n")
        # for i, val in enumerate(flat_matrix):
        #     # for the first three lines, add m, n, k
        #     if i == 0:
        #         hex_val = f"{m:02X}"
        #         f.write(f"\t{i}\t:\t{hex_val};\n")
        #     elif i == 1:
        #         hex_val = f"{n:02X}"
        #         f.write(f"\t{i}\t:\t{hex_val};\n")
        #     elif i == 2:
        #         hex_val = f"{k:02X}"
        #         f.write(f"\t{i}\t:\t{hex_val};\n")
        #     else:
        #         hex_val = f"{val:02X}"
        #         f.write(f"\t{i}\t:\t{hex_val};\n")

        # f.write("END;\n")
    print(f"Matrix saved to {filename}")
    
def generate_and_save_tiles(weights, activations, output_dir, layer_size, tile_size):
    """Slices matrices, generates tiles, and saves them as .mif files."""
    print(f"\n--- GENERATING {tile_size}x{tile_size} TILES ---")
    
    # Ensure the base output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    start_r, start_c = 0, 0 
    tile_counter = 0
    
    for c in range(start_c, start_c + layer_size, tile_size):
        w_tile = weights[start_r:start_r+tile_size, c:c+tile_size]
        a_tile = activations[start_r:start_r+tile_size, c:c+tile_size]

        # --- NEW: Create the tile-specific sub-folder ---
        tile_dir = os.path.join(output_dir, f"tile_{tile_counter}")
        if not os.path.exists(tile_dir):
            os.makedirs(tile_dir)
            
        print(f"Processing Tile {tile_counter}...")

        # Save to the new sub-folder
        w_filename = os.path.join(tile_dir, f"weight_tile_{tile_counter}.mif")
        a_filename = os.path.join(tile_dir, f"activation_tile_{tile_counter}.mif")
        
        save_matrix_to_mif(w_tile, w_filename, LAYER_SIZE, tile_size, m=tile_size, n=tile_size, k=tile_size)
        save_matrix_to_mif(a_tile, a_filename, LAYER_SIZE, tile_size, m=tile_size, n=tile_size, k=tile_size)   
        
        tile_counter += 1

    print(f"Successfully saved {tile_counter} tiles in: {output_dir}")
    

# def find_joint_non_sparse_tile_start(weights, activations, tile_size, min_nonzero=8):
#     rows, cols = weights.shape
#     for r in range(0, rows - tile_size + 1):
#         for c in range(0, cols - tile_size + 1):
#             w_tile = weights[r:r+tile_size, c:c+tile_size]
#             a_tile = activations[r:r+tile_size, c:c+tile_size]
#             if np.count_nonzero(w_tile) > min_nonzero and np.count_nonzero(a_tile) > min_nonzero:
#                 return r, c
#     # If no suitable tile is found, return (0,0)
#     return 0, 0

    
def run_inference(model, input_tensor, labels):
    """Performs inference on the input tensor and prints the prediction."""
    print("\n--- FULL MODEL INFERENCE (FOR REFERENCE) ---")
    
    output_logits = model(input_tensor)
    prediction_index = torch.argmax(output_logits, 1).item()
    
    probabilities = torch.nn.functional.softmax(output_logits, dim=1)
    confidence = probabilities[0, prediction_index].item() * 100
    
    predicted_class = labels[str(prediction_index)][1]
    
    print(f"Prediction: {predicted_class.replace('_', ' ').title()}")
    print(f"Confidence: {confidence:.2f}%")
    
def mif_to_matrix(filename, rows, cols):
    """
    Reads a Quartus Memory Initialization File (.mif) and converts it
    into a 2D NumPy array.
    """
    data_values = []
    
    try:
        with open(filename, 'r') as f:
            in_content_section = False
            for line in f:
                if 'CONTENT BEGIN' in line:
                    in_content_section = True
                    continue
                if 'END;' in line:
                    break
                if in_content_section:
                    match = re.search(r'\d+\s*:\s*(-?[0-9a-fA-F]+);', line)
                    if match:
                        hex_value = match.group(1)
                        data_values.append(int(hex_value, 16))
        # Skip the first 3 values (m, n, k)
        data_values = data_values[3:]
                        
    except FileNotFoundError:
        print(f"Error: The file '{filename}' was not found.")
        return None

    try:
        matrix = np.array(data_values, dtype=np.int32).reshape((rows, cols))
        return matrix
    except ValueError as e:
        print(f"Error: Could not reshape data into a {rows}x{cols} matrix. {e}")
        return None

def generate_vhdl_stimulus(compact_data, compact_weight, m, k, n, N=8):
    """
    Generates VHDL 'constant' declarations for the compacted matrices.
    Updated for 16-bit (INT16) research.
    """
    print(f"-- VHDL stimulus for compacted matrices (INT16)")
    print(f"constant ACTIVE_ROWS : integer := {m};")
    print(f"constant ACTIVE_K : integer := {k};")
    print(f"constant ACTIVE_COLS : integer := {n};")

    # Generate VHDL for the Data Matrix (using s16 instead of u8)
    vhdl_data = f"\nconstant MATRIX_DATA_STIMULUS : systolic_array_matrix_input := (\n"
    for r in range(m):
        # Convert each value to int to ensure clean VHDL output
        row_elements = [f"s16({int(compact_data[r, c])})" for c in range(k)]
        padding = [f"s16(0)"] * (N - k)
        vhdl_data += f"    ({', '.join(row_elements + padding)}),\n"
    vhdl_data += "    others => (others => s16(0))\n);"
    print(vhdl_data)

    # Generate VHDL for the Weight Matrix (using s16 instead of u8)
    vhdl_weight = f"\nconstant MATRIX_WEIGHT_STIMULUS : systolic_array_matrix_input := (\n"
    for r in range(k):
        row_elements = [f"s16({int(compact_weight[r, c])})" for c in range(n)]
        padding = [f"s16(0)"] * (N - n)
        vhdl_weight += f"    ({', '.join(row_elements + padding)}),\n"
    vhdl_weight += "    others => (others => s16(0))\n);"
    print(vhdl_weight)
    print("-" * 40)
    
def mac_calculator(data_mif_path, weight_mif_path, data_m, data_n, weight_m, weight_n):
    """Calculates the matrix multiplication using a MAC approach."""
    data_matrix = mif_to_matrix(data_mif_path, data_m, data_n)
    weight_matrix = mif_to_matrix(weight_mif_path, weight_m, weight_n)

    if data_matrix is None or weight_matrix is None:
        print("\nAborting MAC calculation due to file read error.")
        return
    
    if data_matrix.shape[1] != weight_matrix.shape[0]:
        print("Error: Matrix dimensions are not compatible for multiplication.")
        return
    
    result_matrix = np.matmul(data_matrix, weight_matrix)
    
    print("\n--- MAC Calculator Result ---")
    print(f"Input A shape: {data_matrix.shape}, Input B shape: {weight_matrix.shape}")
    print("\nResult Matrix (C):")
    print(result_matrix)
    print("\n" + "="*40 + "\n")

def simulate_systolic_array(matrix_A, matrix_B, m,n,k):
    """Simulates the behavior of a systolic array for matrix multiplication."""
    if matrix_A is None or matrix_B is None:
        return
        
    rows_A, cols_A = matrix_A.shape
    rows_B, cols_B = matrix_B.shape
    
    if cols_A != rows_B:
        print("Error: Matrix dimensions are not compatible for multiplication.")
        return

    # Simulate the MAC operation
    result_matrix = np.matmul(matrix_A, matrix_B)
    
    # Calculate the latency (Total Clock Cycles)
    # latency = (rows_A - 1) + (cols_B - 1) + cols_A # !! change
    latency = m + n + k - 1
    
    print("\n--- 4. Systolic Array Simulation ---")
    print(f"Input A shape: {matrix_A.shape}, Input B shape: {matrix_B.shape}")
    # print both input and output matrices
    print("\nInput Matrix (A):")
    print(matrix_A)
    print("\nInput Matrix (B):")
    print(matrix_B)
    print("\nResult Matrix (C):")
    print(result_matrix)
    print(f"\nSimulated Total Clock Cycles (Latency): {latency}")
    print(f"Active Rows (m): {m}, Active Columns (n): {n}, Active K (k): {k}")
    print("\n" + "="*40)


def coordinated_row_removal(data_matrix, weight_matrix):
    """
    This function correctly implements your original goal. It finds all active
    rows and columns for each matrix but coordinates the inner dimension 'k'
    to ensure the multiplication is always valid.
    """
    data_matrix = np.array(data_matrix)
    weight_matrix = np.array(weight_matrix)

    # 1. Find the active rows for data (m) and active columns for weight (n).
    active_m_indices = np.where(np.any(data_matrix, axis=1))[0]
    active_n_indices = np.where(np.any(weight_matrix, axis=0))[0]

    # 2. Find the active inner dimension 'k' by taking the UNION of active
    #    data columns and active weight rows. This captures all contributing parts.
    data_k_indices = np.where(np.any(data_matrix, axis=0))[0]
    weight_k_indices = np.where(np.any(weight_matrix, axis=1))[0]
    # Using a set union ensures we have a sorted list of unique indices
    common_k_indices = sorted(list(set(data_k_indices) | set(weight_k_indices)))

    # 3. Create the new, dense matrices by stripping all zero-axes using these indices.
    compact_data = data_matrix[np.ix_(active_m_indices, common_k_indices)]
    compact_weight = weight_matrix[np.ix_(common_k_indices, active_n_indices)]

    # 4. Extract the final, correct dimensions.
    m_new = compact_data.shape[0]
    k_new = compact_data.shape[1]
    n_new = compact_weight.shape[1]

    return compact_data, compact_weight, m_new, k_new, n_new

def analyze_optimization(model, image_dir):
    results_log = []
    
    # Define AlexNet Layers
    layers = [
        {'name': 'Conv1', 'c': 0, 'r': 1},
        {'name': 'Conv2', 'c': 3, 'r': 4},
        {'name': 'Conv3', 'c': 6, 'r': 7},
        {'name': 'Conv5', 'c': 10, 'r': 11}
    ]

    # Throughput Multipliers for Arria 10
    precision_factors = {
        'FP32': 1.0,   # 1 DSP = 1 MAC
        'INT16': 2.0,  # 1 DSP = 2 MACs (Packed)
        'INT8': 2.0    # 1 DSP = 2 MACs (Standard A10 mode)
    }

    # Iterate through every image in your folder
    image_files = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    for layer in layers:
        print(f"\n--- Analysing {layer['name']} ---")
        
        # Extract data for this layer (using one image as a representative sample for weights)
        # In a full study, you'd average the activations across all images
        w, a = extract_conv_weights_and_activations(model, preprocess_image(image_files[0]), layer['c'], layer['r'])

        for t_size in [8, 16, 32]:
            # Calculate Sparsity Speedup (Avg cycles across all images)
            total_reduction = []
            
            for img_path in image_files[:20]: # Process up to 10 images for speed | Can try 20 for better avg
                input_t = preprocess_image(img_path)
                _, act = extract_conv_weights_and_activations(model, input_t, layer['c'], layer['r'])
                
                # Sample a few tiles to find average sparsity
                for i in range(0, 20): 
                    # Slice a random tile
                    data_tile = act[i*t_size:(i+1)*t_size, :t_size]
                    weight_tile = w[:t_size, :t_size]
                    
                    if data_tile.shape[0] == t_size:
                        _, _, m, k, n = coordinated_row_removal(data_tile, weight_tile)
                        baseline = (t_size * 3) - 1
                        actual = (m + n + k - 1) if m > 0 else 0
                        total_reduction.append(baseline / actual if actual > 0 else baseline)

            avg_sparsity_speedup = np.mean(total_reduction)

            # Compare Precisions
            for prec, hw_factor in precision_factors.items():
                # OVERALL SCORE = (Sparsity Gain) * (Hardware Parallelism Gain)
                # This represents how many 'Effective OPS' you get per Arria 10 DSP block
                efficiency_score = avg_sparsity_speedup * hw_factor
                
                results_log.append({
                    'Layer': layer['name'],
                    'Tile': t_size,
                    'Precision': prec,
                    'Sparsity_Gain': round(avg_sparsity_speedup, 2),
                    'Total_Efficiency': round(efficiency_score, 2)
                })

    return pd.DataFrame(results_log)

def twos_complement_to_uint8(arr):
    return arr.astype(np.int8).astype(np.uint8)

import subprocess
import os
import numpy as np

def run_jtag_inference(m, n, k, data_matrix, weight_matrix):

    # 1. THE EXECUTABLE
    java_exe = "C:/intelfpga_pro/24.1/quartus/bin64/java17/jre64/bin\java.exe"
    
    # 2. THE FULL CLASSPATH (Copied exactly from your CMD output)
    # We use a raw string to ensure Windows backslashes don't break the path
    classpath = (
        "C:/intelfpga_pro/24.1/quartus/../syscon/lib/FastInfoset.jar;"
        "C:/intelfpga_pro/24.1/quartus/../syscon/lib/animal-sniffer-annotations-1.23.jar;"
        "C:/intelfpga_pro/24.1/quartus/../syscon/lib/annotations-4.1.1.4.jar;"
        "C:/intelfpga_pro/24.1/quartus/../syscon/lib/bare-bones-browser-launch-3.2.jar;"
        "C:/intelfpga_pro/24.1/quartus/../syscon/lib/checker-qual-3.33.0.jar;"
        "C:/intelfpga_pro/24.1/quartus/../syscon/lib/codemodel.jar;"
        "C:/intelfpga_pro/24.1/quartus/../syscon/lib/com.altera.privateinterfaces.jar;"
        "C:/intelfpga_pro/24.1/quartus/../syscon/lib/com.altera.sopcreport.jar;"
        "C:/intelfpga_pro/24.1/quartus/../syscon/lib/com.altera.systemconsole.app.jar;"
        "C:/intelfpga_pro/24.1/quartus/../syscon/lib/com.altera.systemconsole.configrom.jar;"
        "C:/intelfpga_pro/24.1/quartus/../syscon/lib/com.altera.systemconsole.dashboard.jar;"
        "C:/intelfpga_pro/24.1/quartus/../syscon/lib/com.altera.systemconsole.designs.jar;"
        "C:/intelfpga_pro/24.1/quartus/../syscon/lib/com.altera.systemconsole.driver.jar;"
        "C:/intelfpga_pro/24.1/quartus/../syscon/lib/com.altera.systemconsole.executor.jar;"
        "C:/intelfpga_pro/24.1/quartus/../syscon/lib/com.altera.systemconsole.gdbserver.jar;"
        "C:/intelfpga_pro/24.1/quartus/../syscon/lib/com.altera.systemconsole.grpc.jar;"
        "C:/intelfpga_pro/24.1/quartus/../syscon/lib/com.altera.systemconsole.gui.jar;"
        "C:/intelfpga_pro/24.1/quartus/../syscon/lib/com.altera.systemconsole.helper.jar;"
        "C:/intelfpga_pro/24.1/quartus/../syscon/lib/com.altera.systemconsole.iobus.jar;"
        "C:/intelfpga_pro/24.1/quartus/../syscon/lib/com.altera.systemconsole.issp.jar;"
        "C:/intelfpga_pro/24.1/quartus/../syscon/lib/com.altera.systemconsole.jar;"
        "C:/intelfpga_pro/24.1/quartus/../syscon/lib/com.altera.systemconsole.lightclient.jar;"
        "C:/intelfpga_pro/24.1/quartus/../syscon/lib/com.altera.systemconsole.matlabapi.jar;"
        "C:/intelfpga_pro/24.1/quartus/../syscon/lib/com.altera.systemconsole.monitor.jar;"
        "C:/intelfpga_pro/24.1/quartus/../syscon/lib/com.altera.systemconsole.plugin.elf.jar;"
        "C:/intelfpga_pro/24.1/quartus/../syscon/lib/com.altera.systemconsole.plugin.jtag.jar;"
        "C:/intelfpga_pro/24.1/quartus/../syscon/lib/com.altera.systemconsole.plugin.memory.jar;"
        "C:/intelfpga_pro/24.1/quartus/../syscon/lib/com.altera.systemconsole.plugin.pli.jar;"
        "C:/intelfpga_pro/24.1/quartus/../syscon/lib/com.altera.systemconsole.scripting.jar;"
        "C:/intelfpga_pro/24.1/quartus/../syscon/lib/com.altera.systemconsole.sldhubchannel.jar;"
        "C:/intelfpga_pro/24.1/quartus/../syscon/lib/com.altera.systemconsole.stp.jar;"
        "C:/intelfpga_pro/24.1/quartus/../syscon/lib/com.altera.systemconsole.tcpchannel.jar;"
        "C:/intelfpga_pro/24.1/quartus/../syscon/lib/com.altera.systemconsole.trace.database.jar;"
        "C:/intelfpga_pro/24.1/quartus/../syscon/lib/com.altera.systemconsole.trace.drivers.jar;"
        "C:/intelfpga_pro/24.1/quartus/../syscon/lib/com.altera.systemconsole.trace.gui.jar;"
        "C:/intelfpga_pro/24.1/quartus/../syscon/lib/com.altera.systemconsole.transceivers.crete3.jar;"
        "C:/intelfpga_pro/24.1/quartus/../syscon/lib/com.altera.systemconsole.transceivers.jar;"
        "C:/intelfpga_pro/24.1/quartus/../syscon/lib/com.altera.systemconsole.ui.jar;"
        "C:/intelfpga_pro/24.1/quartus/../syscon/lib/com.altera.systemconsole.unifiedtoolkit.jar;"
        "C:/intelfpga_pro/24.1/quartus/../syscon/lib/com.altera.systemconsole.usb.jar;"
        "C:/intelfpga_pro/24.1/quartus/../syscon/lib/com.altera.systemconsole.usb.tcp.jar;"
        "C:/intelfpga_pro/24.1/quartus/../syscon/lib/com.altera.tcl.interpreter.jar;"
        "C:/intelfpga_pro/24.1/quartus/../syscon/lib/com.altera.tcl.ui.jar;"
        "C:/intelfpga_pro/24.1/quartus/../syscon/lib/com.intel.shared.jar;"
        "C:/intelfpga_pro/24.1/quartus/../syscon/lib/commons-collections4-4.3.jar;"
        "C:/intelfpga_pro/24.1/quartus/../syscon/lib/commons-lang3-3.12.0.jar;"
        "C:/intelfpga_pro/24.1/quartus/../syscon/lib/commons-text-1.10.0.jar;"
        "C:/intelfpga_pro/24.1/quartus/../syscon/lib/docking-frames-common-1.1.2_20c.jar;"
        "C:/intelfpga_pro/24.1/quartus/../syscon/lib/docking-frames-core-1.1.2_20c.jar;"
        "C:/intelfpga_pro/24.1/quartus/../syscon/lib/dtd-parser.jar;"
        "C:/intelfpga_pro/24.1/quartus/../syscon/lib/error_prone_annotations-2.20.0.jar;"
        "C:/intelfpga_pro/24.1/quartus/../syscon/lib/failureaccess-1.0.1.jar;"
        "C:/intelfpga_pro/24.1/quartus/../syscon/lib/grpc-api-1.60.1.jar;"
        "C:/intelfpga_pro/24.1/quartus/../syscon/lib/grpc-context-1.60.1.jar;"
        "C:/intelfpga_pro/24.1/quartus/../syscon/lib/grpc-core-1.60.1.jar;"
        "C:/intelfpga_pro/24.1/quartus/../syscon/lib/grpc-netty-shaded-1.60.1.jar;"
        "C:/intelfpga_pro/24.1/quartus/../syscon/lib/grpc-protobuf-1.60.1.jar;"
        "C:/intelfpga_pro/24.1/quartus/../syscon/lib/grpc-protobuf-lite-1.60.1.jar;"
        "C:/intelfpga_pro/24.1/quartus/../syscon/lib/grpc-stub-1.60.1.jar;"
        "C:/intelfpga_pro/24.1/quartus/../syscon/lib/grpc-util-1.60.1.jar;"
        "C:/intelfpga_pro/24.1/quartus/../syscon/lib/gson-2.10.1.jar;"
        "C:/intelfpga_pro/24.1/quartus/../syscon/lib/guava-32.0.1-jre.jar;"
        "C:/intelfpga_pro/24.1/quartus/../syscon/lib/istack-commons-runtime.jar;"
        "C:/intelfpga_pro/24.1/quartus/../syscon/lib/istack-commons-tools.jar;"
        "C:/intelfpga_pro/24.1/quartus/../syscon/lib/j2objc-annotations-2.8.jar;"
        "C:/intelfpga_pro/24.1/quartus/../syscon/lib/jacl1.3.2a.jar;"
        "C:/intelfpga_pro/24.1/quartus/../syscon/lib/javax.activation-api.jar;"
        "C:/intelfpga_pro/24.1/quartus/../syscon/lib/jaxb-api.jar;"
        "C:/intelfpga_pro/24.1/quartus/../syscon/lib/jaxb-jxc.jar;"
        "C:/intelfpga_pro/24.1/quartus/../syscon/lib/jaxb-runtime.jar;"
        "C:/intelfpga_pro/24.1/quartus/../syscon/lib/jaxb-xjc.jar;"
        "C:/intelfpga_pro/24.1/quartus/../syscon/lib/jfreechart-1.5.0.jar;"
        "C:/intelfpga_pro/24.1/quartus/../syscon/lib/jline-reader-3.11.0.jar;"
        "C:/intelfpga_pro/24.1/quartus/../syscon/lib/jline-terminal-3.11.0.jar;"
        "C:/intelfpga_pro/24.1/quartus/../syscon/lib/jsr305-3.0.2.jar;"
        "C:/intelfpga_pro/24.1/quartus/../syscon/lib/listenablefuture-9999.0-empty-to-avoid-conflict-with-guava.jar;"
        "C:/intelfpga_pro/24.1/quartus/../syscon/lib/miglayout-core-5.2.jar;"
        "C:/intelfpga_pro/24.1/quartus/../syscon/lib/miglayout-swing-5.2.jar;"
        "C:/intelfpga_pro/24.1/quartus/../syscon/lib/perfmark-api-0.26.0.jar;"
        "C:/intelfpga_pro/24.1/quartus/../syscon/lib/proto-google-common-protos-2.22.0.jar;"
        "C:/intelfpga_pro/24.1/quartus/../syscon/lib/protobuf-java-3.24.0.jar;"
        "C:/intelfpga_pro/24.1/quartus/../syscon/lib/relaxng-datatype.jar;"
        "C:/intelfpga_pro/24.1/quartus/../syscon/lib/rngom.jar;"
        "C:/intelfpga_pro/24.1/quartus/../syscon/lib/stax-ex.jar;"
        "C:/intelfpga_pro/24.1/quartus/../syscon/lib/txw2.jar;"
        "C:/intelfpga_pro/24.1/quartus/../syscon/lib/xsom.jar;"
    )

    # Write the data to a file instead of passing it as a "string" argument
    with open("tile_data.txt", "w") as f:
        # Line 1: Activations
        f.write(" ".join(map(str, data_matrix.flatten().astype(int))) + "\n")
        # Line 2: Weights
        f.write(" ".join(map(str, weight_matrix.flatten().astype(int))) + "\n")
    
    tcl_path = "C:/Users/pchh520/Documents/GitHub/part-4-project/SummerResearch/Python/PythonAutomation.tcl"
    quartus_bin64 = "C:/intelfpga_pro/24.1/quartus/bin64"

    # 4. CONSTRUCT COMMAND
    # We switch the entry point to "Shell" and add "--cli" and "--script"
    cmd = [
        java_exe,
        f"-Djava.library.path={quartus_bin64}", 
        "-Dawt.useSystemAAFontSettings=on",
        "-cp", classpath,
        "com.altera.systemconsole.app.Shell",
        "--cli",
        f"--script={tcl_path}",
        str(m), str(n), str(k)
    ]

    print(f"\n[SYSTEM] Calling Intel JVM directly for Tile (M={m}, N={n}, K={k})...")

    try:
            # We set cwd to the bin64 folder so Java finds its native dependencies (.dlls)
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                shell=True,
                cwd="C:/intelfpga_pro/24.1/quartus/bin64" 
            )
        
            # Log the output to see if Java is complaining about memory or paths
            print("\n--- JVM STDOUT ---")
            print(result.stdout)
            print("\n--- JVM STDERR ---")
            print(result.stderr)

            # 5. PARSE DATA
            if "FPGA_DATA:" in result.stdout:
                line = [l for l in result.stdout.split('\n') if "FPGA_DATA:" in l][0]
                raw_vals = line.replace("FPGA_DATA:", "").strip().split()
                clean_list = [int(x) for x in raw_vals]
                return np.array(clean_list).reshape(m, n)
            else:
                print("[ERROR] JTAG session succeeded, but no 'FPGA_DATA:' found in output.")
                print("Full Output:", result.stdout)
                return None

    except Exception as e:
        print(f"[CRITICAL] Failed to launch JVM: {str(e)}")
        return None
    
    # 4. Return the output so we can see what the FPGA did
    #return result.stdout

def prepare_simulation_case(model, layer_name, conv_idx, relu_idx, tile_idx, t_size):
    """Generates tiles, triggers hardware, and reconstructs the output."""
    print(f"\n=== PROCESSING: {layer_name} (Tile {tile_idx}, Size {t_size}) ===")
    
    # 1. Extract data from the AlexNet model
    input_tensor = preprocess_image(IMAGE_PATH)
    w, a = extract_conv_weights_and_activations(model, input_tensor, conv_idx, relu_idx)
    
    # 2. Slice the specific tile
    # (Assuming we are looking at a square tile based on tile_idx)
    sim_data = a[tile_idx*t_size:(tile_idx+1)*t_size, :t_size]
    sim_weight = w[:t_size, :t_size]
    
    if sim_data is not None and sim_weight is not None:
        # 3. SPARSITY GATHER: Get 'Stripped' data and the Index Maps
        s_data, s_weight, m, k, n = coordinated_row_removal(sim_data, sim_weight)
        
        # Identify WHICH rows and columns are active for reconstruction
        active_m_indices = np.where(np.any(sim_data, axis=1))[0]
        active_n_indices = np.where(np.any(sim_weight, axis=0))[0]
        
        # 4. EXECUTE ON ARRIA 10
        hw_stripped = run_jtag_inference(m, n, k, s_data, s_weight)
        
        if hw_stripped is not None:
            # 5. SOFTWARE VERIFICATION (The Golden Reference)
            sw_stripped = np.matmul(s_data.astype(np.int32), s_weight.astype(np.int32))
            
            # 6. ACCURACY ANALYSIS (MSE)
            mse = np.mean((sw_stripped - hw_stripped)**2)
            print(f"\n--- ACCURACY RESULTS ---")
            print(f"Mean Squared Error (MSE): {mse:.4f}")
            if mse == 0:
                print(">>> SUCCESS: Bit-accurate match between Python and Hardware.")
            else:
                print(">>> WARNING: Discrepancy detected. Check bit-widths or flush cycles.")

            # 7. SPATIAL RECONSTRUCTION (The 'Scatter' phase)
            # Re-expand the m x n result back into a 32x32 grid
            reconstructed = np.zeros((t_size, t_size), dtype=np.int32)
            for i, orig_row in enumerate(active_m_indices):
                for j, orig_col in enumerate(active_n_indices):
                    reconstructed[orig_row, orig_col] = hw_stripped[i, j]

            print(f"\n--- RECONSTRUCTION COMPLETE ---")
            print(f"Original Tile Size: {t_size}x{t_size}")
            print(f"Computed Tile Size: {m}x{n} (Saved {((1 - (m*n)/(t_size*t_size))*100):.1f}% Work)")
            print("Preview of Reconstructed Output (First 4x4):")
            print(reconstructed[:4, :4])
            
            # Return this for the next layer in the pipeline
            return reconstructed

    return None

def main():
    model = load_quantized_alexnet()
    image_dir = 'C:/Users/pchh520/Documents/GitHub/part-4-project/SummerResearch/Python/sparsity_analysis_images'
    
    df = analyze_optimization(model, image_dir)
    
    # Find the best combination for each layer
    print("\n\n=== OPTIMAL HARDWARE CONFIGURATIONS PER LAYER ===")
    for layer in df['Layer'].unique():
        layer_df = df[df['Layer'] == layer]
        best_row = layer_df.loc[layer_df['Total_Efficiency'].idxmax()]
        
        print(f"Layer: {layer}")
        print(f"  Best Config: {best_row['Tile']}x{best_row['Tile']} at {best_row['Precision']}")
        print(f"  Reason: Total Efficiency of {best_row['Total_Efficiency']}x over baseline")
        print("-" * 40)
        
    # Optional: Save the whole CSV for your report
    df.to_csv('alexnet_optimization_results_2.csv', index=False)

    # # --- SIMULATION STIMULUS GENERATOR ---
    # # Pick a "Golden Tile" to test in QuestaSim
    # # Example: Conv2, 8x8 Tile Size, Tile index 0
    # target_layer = "Conv1"
    # target_tile = 0
    # target_size = 32
    
    # print(f"\n\n--- GENERATING VHDL STIMULUS FOR {target_layer} TILE {target_tile} ---")
    
    # # Locate the MIFs saved earlier
    # layer_dir = os.path.join(MIF_OUTPUT_DIR, f"{target_layer}_tile_{target_size}")
    # a_path = os.path.join(layer_dir, f"tile_{target_tile}", f"activation_tile_{target_tile}.mif")
    # w_path = os.path.join(layer_dir, f"tile_{target_tile}", f"weight_tile_{target_tile}.mif")
    
    # # Load them back into Python
    # sim_data = mif_to_matrix(a_path, target_size, target_size)
    # sim_weight = mif_to_matrix(w_path, target_size, target_size)
    
    # if sim_data is not None and sim_weight is not None:
    #     # Get the "Stripped" version
    #     s_data, s_weight, m, k, n = coordinated_row_removal(sim_data, sim_weight)
        
    #     # Convert weight to unsigned for VHDL if necessary
    #     s_weight_uint = twos_complement_to_uint8(s_weight)
        
    #     # This prints the code you copy-paste into your testbench
    #     generate_vhdl_stimulus(s_data, s_weight_uint, m, k, n, N=target_size)

    # To run Case #1 (The 32x32 Hero):
    model = load_quantized_alexnet()
    prepare_simulation_case(model, "Conv1", 0, 1, tile_idx=0, t_size=32)

if __name__ == '__main__':
    main()