
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

# --- Constants ---
# IMAGE_PATH = 'C:/Users/iamkr/Documents/part-4-project/Final/Python/cat.jpg'
IMAGE_PATH = 'C:/Users/iamkr/Documents/part-4-project/Final/Python/hand_xray.jpg'
MIF_OUTPUT_DIR = "C:/Users/iamkr/Documents/part-4-project/Final/mif/pipeline_v2"
TEST_DATA_MIF_DIR = 'C:/Users/iamkr/Documents/part-4-project/Final/mif/pipeline_v2/activation_tile_4.mif'
TEST_WEIGHT_MIF_DIR = 'C:/Users/iamkr/Documents/part-4-project/Final/mif/pipeline_v2/weight_tile_4.mif'
STRIPPED_DATA_MIF_DIR = 'C:/Users/iamkr/Documents/part-4-project/Final/mif/pipeline_v2/stripped_activation.mif'
STRIPPED_WEIGHT_MIF_DIR = 'C:/Users/iamkr/Documents/part-4-project/Final/mif/pipeline_v2/stripped_weight.mif'
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

# Extract Weights and Activation (with ReLU)
# def extract_weights_and_activations(model, input_tensor):
#     """Extracts weights and activations from the model."""
#     # Extract weights from the first convolutional layer as 4D tensor
#     quantized_weight_tensor = model.get_conv_weights(layer_idx=0)
#     # Convert Weight to 2D by shaping it to (filter, channel*height*width)
#     weights_2d = quantized_weight_tensor.view(quantized_weight_tensor.size(0), -1).detach().cpu().numpy()   

#     # Extract the Activation Tensor after ReLU
#     with torch.no_grad():
#         activation_tensor = model(input_tensor, return_conv=True) # Get conv output
#         relu_activation_tensor = torch.nn.functional.relu(activation_tensor) # Apply ReLU
#         # Convert activation to 2D using im2col
#         activations_2d = relu_activation_tensor.squeeze(0).transpose(0, 1).detach().cpu().numpy() # Shape (L, C*ks*ks)
        
#     return weights_2d, activations_2d


# def extract_matched_matrices(model, input_tensor, target_feature_idx, next_conv_feature_idx):
#     """ Extracts matched 2D weight and activation matrices from a quantized model. """
#     # Use a dictionary to store the captured activation tensor
#     captured_activation = {}
#     # Hook: ggrabbing the output of a layer during a forward pass
#     def hook_fn(module, input, output):
#         captured_activation['value'] = output

#     # 1. Identify the target layer (to hook) and the next conv layer (for weights/params)
#     target_layer = model.model_fp32.features[target_feature_idx]
#     next_conv_layer = model.model_fp32.features[next_conv_feature_idx]

#     # 2. Register the forward hook on the target layer
#     handle = target_layer.register_forward_hook(hook_fn)

#     # 3. Run a forward pass to trigger the hook
#     # This means we need to pass the input tensor through the model
#     with torch.no_grad():
#         model(input_tensor)

#     # 4. Remove the hook immediately to keep things clean
#     handle.remove()

#     # 5. Retrieve the captured activation tensor
#     activation_4d = captured_activation['value'] # This is the output of the target layer after the forward pass.
#     if hasattr(activation_4d, 'dequantize'):
#         activation_4d = activation_4d.dequantize()
#         # quantise to int8
#         activation_4d = activation_4d.init_repr().to(torch.int8)
        

#     # 6. Extract the weights from the NEXT conv layer and reshape to 2D
#     weight_4d = model.model_fp32.classifier[1].weight()
#     fc1_weights_int8 = weight_4d.int_repr().numpy() 
#     # convert to 2D by shaping it to (filter, channel*height*width)
#     weights_2d = fc1_weights_int8.reshape(fc1_weights_int8.shape[0], -1)

#     # 7. Apply im2col (`unfold`) to the captured activation tensor
#     #    using the parameters from the NEXT conv layer.
#     activations_unfolded = F.unfold( 
#         activation_4d,
#         kernel_size=next_conv_layer.kernel_size,
#         stride=next_conv_layer.stride,
#         padding=next_conv_layer.padding
#     )
        
#     # Reshape to the final 2D activation matrix
#     activations_2d = activations_unfolded.squeeze(0)
#     # convert this from a tensor to a numpy array
#     activations_2d = activations_2d.detach().cpu().numpy()
    
#     print(f"Extracted weight matrix shape: {weights_2d.shape}")
#     print(weights_2d)
#     print(f"Extracted activation matrix shape: {activations_2d.shape}")
#     print(activations_2d)

#     return weights_2d, activations_2d



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
def save_matrix_to_mif(matrix, filename, depth, width):
    """Saves a 2D numpy array to a Memory Initialization File (MIF)."""
    
    with open(filename, 'w') as f:
        f.write(f"DEPTH = {depth};\n")
        f.write(f"WIDTH = {width};\n")
        f.write("ADDRESS_RADIX = HEX;\n")
        f.write("DATA_RADIX = HEX;\n")
        f.write("CONTENT BEGIN\n")
        
        flat_matrix = matrix.flatten()
        for i, val in enumerate(flat_matrix):
            hex_val = f"{val:02X}"
            f.write(f"\t{i}\t:\t{hex_val};\n")

        f.write("END;\n")
    # print(f"Matrix saved to {filename}")
    
# Slices the matrices into NxN tiles and saves each tile as a separate MIF file
def generate_and_save_tiles(weights, activations, output_dir, layer_size, tile_size):
    """Slices matrices, generates 8x8 tiles, and saves them as .mif files."""
    # --- 1. Slice Matrices to the specified layer size ---
    print(f"\n--- FINDING CONSISTENT NON-SPARSE STARTING POINT FOR TILING ---")
    # start_r, start_c = find_joint_non_sparse_tile_start(weights, activations, tile_size)
    start_r, start_c = 0, 0  # Default to (0,0) 
    print(f"Tiling starts at row {start_r}, col {start_c} for both weights and activations.")

    # --- 2. Generate and save tiles ---
    print(f"\n--- GENERATING {tile_size}x{tile_size} TILES AND SAVING AS .MIF ---")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    tile_counter = 0
    for c in range(start_c, start_c + layer_size, tile_size):
        w_tile = weights[start_r:start_r+tile_size, c:c+tile_size]
        a_tile = activations[start_r:start_r+tile_size, c:c+tile_size]
        # Ensure a_tile is tile_size x tile_size
        # if a_tile.shape[0] < tile_size or a_tile.shape[1] < tile_size:
        #     padded_tile = np.zeros((tile_size, tile_size), dtype=weights.dtype)
        #     padded_tile[:a_tile.shape[0], :a_tile.shape[1]] = a_tile
        #     a_tile = padded_tile
        print(f"\Tile {tile_counter}:")
        print(w_tile)
        print(a_tile)

        save_matrix_to_mif(w_tile, os.path.join(output_dir, f"weight_tile_{tile_counter}.mif"), tile_size, tile_size)
        save_matrix_to_mif(a_tile, os.path.join(output_dir, f"activation_tile_{tile_counter}.mif"), tile_size, tile_size)

        tile_counter += 1

    print(f"Generated and saved {tile_counter} pairs of {tile_size}x{tile_size} tiles.")
    print(f"Files are saved in the '{output_dir}' directory.")
    

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
                    match = re.search(r'\d+\s*:\s*([0-9a-fA-F]+);', line)
                    if match:
                        hex_value = match.group(1)
                        data_values.append(int(hex_value, 16))
                        
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
    The padding is a VHDL requirement to fit the smaller logical matrix
    into the fixed-size 8x8 physical type.
    """
    print(f"-- VHDL stimulus for compacted matrices")
    print(f"constant ACTIVE_ROWS : integer := {m};")
    print(f"constant ACTIVE_K : integer := {k};")
    print(f"constant ACTIVE_COLS : integer := {n};")

    # Generate VHDL for the Data Matrix
    vhdl_data = f"\nconstant MATRIX_DATA_STIMULUS : systolic_array_matrix_input := (\n"
    for r in range(m):
        row_elements = [f"u8({compact_data[r, c]})" for c in range(k)]
        padding = [f"u8(0)"] * (N - k)
        vhdl_data += f"    ({', '.join(row_elements + padding)}),\n"
    vhdl_data += "    others => (others => u8(0))\n);"
    print(vhdl_data)

    # Generate VHDL for the Weight Matrix
    vhdl_weight = f"\nconstant MATRIX_WEIGHT_STIMULUS : systolic_array_matrix_input := (\n"
    for r in range(k):
        row_elements = [f"u8({compact_weight[r, c]})" for c in range(n)]
        padding = [f"u8(0)"] * (N - n)
        vhdl_weight += f"    ({', '.join(row_elements + padding)}),\n"
    vhdl_weight += "    others => (others => u8(0))\n);"
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
    latency = m + n + k - 2  
    
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

# Main 
def main():
    # Load and Quantise AlexNet
    model = load_quantized_alexnet()
    print("\n---  MODEL LOADED AND QUANTISED ---")
    
    # Preprocess the image
    input_tensor = preprocess_image(IMAGE_PATH)
    if input_tensor is None:
        return # Exit if image was not found
    print("\n--- IMAGE PREPROCESSED ---")

    # --- Extract and print quantized conv weights and activations (first conv layer) ---
    print("\n--- EXTRACTING QUANTISED CONV WEIGHTS AND ACTIVATION (CONV0, RELU1) ---")
    conv_weights_2d, conv_activations_2d = extract_conv_weights_and_activations(model, input_tensor, conv_idx=0, relu_idx=1)

    # Tile the matrices and save as MIF files
    generate_and_save_tiles(conv_weights_2d, conv_activations_2d, MIF_OUTPUT_DIR, LAYER_SIZE, TILE_SIZE)
    # Run Inference
    labels = get_imagenet_labels()
    run_inference(model, input_tensor, labels)




if __name__ == '__main__':
    main()