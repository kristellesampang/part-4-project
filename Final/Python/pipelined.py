import torch
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
IMAGE_PATH = 'C:/Users/iamkr/Documents/part-4-project/Final/Python/cat.jpg'
MIF_OUTPUT_DIR = "C:/Users/iamkr/Documents/part-4-project/Final/mif/pipeline"
LAYER_SIZE = 64
TILE_SIZE = 8

# --- Model Wrapper and Utility Functions (Unchanged) ---

class QuantizableAlexNet(torch.nn.Module):
    """A wrapper class to make AlexNet quantizable."""
    def __init__(self, model_fp32):
        super(QuantizableAlexNet, self).__init__()
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()
        self.model_fp32 = model_fp32

    def forward(self, x):
        x = self.quant(x)
        x = self.model_fp32.features(x)
        x = self.model_fp32.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.model_fp32.classifier(x)
        x = self.dequant(x)
        return x

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

def save_matrix_as_mif(matrix, filename, data_width=8):
    """Saves a 2D NumPy matrix into a Quartus Memory Initialization File (.mif)."""
    depth = matrix.size
    with open(filename, 'w') as f:
        f.write(f"WIDTH={data_width};\n")
        f.write(f"DEPTH={depth};\n")
        f.write("ADDRESS_RADIX=UNS;\n")
        f.write("DATA_RADIX=HEX;\n\n")
        f.write("CONTENT BEGIN\n")

        flat_matrix = matrix.flatten()
        for i, val in enumerate(flat_matrix):
            hex_val = f"{val:02X}"
            f.write(f"\t{i}\t:\t{hex_val};\n")

        f.write("END;\n")

# --- Refactored Helper Functions ---

def load_and_quantize_model():
    """Loads a pre-trained AlexNet model and applies dynamic quantization."""
    print("Loading and quantizing AlexNet model...")
    alexnet_fp32 = models.alexnet(weights=models.AlexNet_Weights.DEFAULT)
    quantizable_model = QuantizableAlexNet(model_fp32=alexnet_fp32)
    quantizable_model.eval()
    quantizable_model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    model_prepared = torch.quantization.prepare(quantizable_model, inplace=False)

    # Calibrate the model with dummy data
    with torch.no_grad():
        for _ in range(10):
            model_prepared(torch.randn(1, 3, 224, 224))
            
    model_quantized = torch.quantization.convert(model_prepared, inplace=False)
    print("Model successfully quantized.")
    return model_quantized

def preprocess_image(image_path):
    """Loads an image from the given path and preprocesses it for the model."""
    try:
        img = Image.open(image_path).convert('RGB')
    except FileNotFoundError:
        print(f"ERROR: The image file '{image_path}' was not found.")
        return None

    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return preprocess(img).unsqueeze(0)

def extract_and_process_tensors(model, input_tensor):
    """Extracts weight and activation tensors, and applies ReLU to weights."""
    # 1. Extract and process weights
    quantized_weight_tensor = model.model_fp32.classifier[1].weight()
    fc1_weights_int8 = quantized_weight_tensor.int_repr().numpy()
    weights_after_relu = np.maximum(0, fc1_weights_int8)
    
    # 2. Extract activations using a forward hook after the first ReLU
    activation = {}
    def get_quantized_activation(name):
        def hook(model, input, output):
            activation[name] = output
        return hook

    # --- THIS IS THE CORRECTED LINE ---
    # Hook the first ReLU layer in the classifier (at index 2)
    hook_handle = model.model_fp32.classifier[2].register_forward_hook(get_quantized_activation('relu1'))
    
    with torch.no_grad():
        model(input_tensor)
    hook_handle.remove()
    
    quantized_activation_tensor = activation['relu1']
    # The output is already post-ReLU, so no need for np.maximum(0, ...)
    activations_int8 = torch.flatten(quantized_activation_tensor.int_repr(), 1).numpy()
    
    return weights_after_relu, activations_int8

def generate_and_save_mif_tiles(weights, activations, output_dir, layer_size, tile_size):
    """Slices matrices, generates 8x8 tiles, and saves them as .mif files."""
    # --- 1. Slice Matrices to the specified layer size ---
    print(f"\n--- SLICING MATRICES TO {layer_size}x{layer_size} ---")
    weights_slice = weights[0:layer_size, 0:layer_size]
    activations_slice = activations[:, 0:layer_size]
    print(f"Created a {weights_slice.shape} weight matrix.")
    print(f"Created a {activations_slice.shape} activation matrix.")
    
    # --- 2. Generate and save tiles ---
    print(f"\n--- GENERATING {tile_size}x{tile_size} TILES AND SAVING AS .MIF ---")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    tile_counter = 0
    for c in range(0, layer_size, tile_size):
        w_tile = weights_slice[0:tile_size, c:c+tile_size]
        
        a_vector_slice = activations_slice[:, c:c+tile_size]
        a_tile = np.zeros((tile_size, tile_size), dtype=weights.dtype)
        a_tile[0, :] = a_vector_slice
        
        save_matrix_as_mif(w_tile, os.path.join(output_dir, f"weight_tile_{tile_counter}.mif"))
        save_matrix_as_mif(a_tile, os.path.join(output_dir, f"activation_tile_{tile_counter}.mif"))
        tile_counter += 1

    print(f"\nGenerated and saved {tile_counter} pairs of {tile_size}x{tile_size} tiles.")
    print(f"Files are saved in the '{output_dir}' directory.")

def run_inference(model, input_tensor, labels):
    """Performs inference on the input tensor and prints the prediction."""
    print("\n--- FULL MODEL INFERENCE (FOR REFERENCE) ---")
    
    output_logits = model(input_tensor)
    prediction_index = torch.argmax(output_logits, 1).item()
    
    probabilities = torch.nn.functional.softmax(output_logits, dim=1)
    confidence = probabilities[0, prediction_index].item() * 100
    
    predicted_class = labels[str(prediction_index)][1]
    
    print(f"\nPrediction: {predicted_class.replace('_', ' ').title()}")
    print(f"Confidence: {confidence:.2f}%")
    

def mif_to_matrix(filename, rows=8, cols=8):
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

def strip_matrices(matrix_A, matrix_B):
    """
    Strips all-zero rows and columns from two matrices while keeping their
    inner dimensions compatible for multiplication (A * B).
    """
    # Find rows in A that are NOT all-zero
    nonzero_rows_A = {i for i in range(matrix_A.shape[0]) if np.any(matrix_A[i, :])}
    
    # Find columns in B that are NOT all-zero
    nonzero_cols_B = {j for j in range(matrix_B.shape[1]) if np.any(matrix_B[:, j])}
    
    # Find inner dimensions (columns of A, rows of B) that are NOT all-zero in EITHER matrix
    nonzero_cols_A = {j for j in range(matrix_A.shape[1]) if np.any(matrix_A[:, j])}
    nonzero_rows_B = {i for i in range(matrix_B.shape[0]) if np.any(matrix_B[i, :])}
    
    # The indices to keep for the inner dimension is the union of both
    keep_indices = sorted(list(nonzero_cols_A.union(nonzero_rows_B)))
    
    # Create the stripped matrices
    stripped_A = matrix_A[sorted(list(nonzero_rows_A)), :][:, keep_indices]
    stripped_B = matrix_B[:, sorted(list(nonzero_cols_B))][keep_indices, :]
    
    return stripped_A, stripped_B

def generate_vhdl_stimulus(matrix_type, matrix_to_print, N=8):
    """
    Generates VHDL 'constant' declarations for a given matrix stimulus.
    """
    active_rows, active_cols = matrix_to_print.shape if matrix_to_print.size > 0 else (0, 0)
    
    print(f"-- VHDL stimulus for {matrix_type} matrix")
    print(f"constant ACTIVE_ROWS_{matrix_type.upper()} : integer := {active_rows};")
    print(f"constant ACTIVE_COLS_{matrix_type.upper()} : integer := {active_cols};")
    
    vhdl = f"constant MATRIX_{matrix_type.upper()}_STIMULUS : systolic_array_matrix_input := (\n"

    for r in range(active_rows):
        row_elements = [f"u8({matrix_to_print[r, c]})" for c in range(active_cols)]
        padding = [f"u8(0)"] * (N - active_cols)
        vhdl += f"    ({', '.join(row_elements + padding)}),\n"
        
    vhdl += "    others => (others => u8(0))\n"
    vhdl += ");"
    
    print(vhdl)
    print("-" * 30)

# create a MAC calculator
def mac_calculator(data_mif_path, weight_mif_path, ROWS=8, COLS=8):
    # Load matrices from the .mif files
    A = mif_to_matrix(data_mif_path, ROWS, COLS)
    B = mif_to_matrix(weight_mif_path, ROWS, COLS)
    
    # Proceed only if both files were loaded successfully
    if A is not None and B is not None:
        # Perform the matrix multiplication
        C = np.matmul(A, B)
        
        print("--- Matrix A (Loaded from MIF) ---")
        print(A)
        print("\n--- Matrix B (Loaded from MIF) ---")
        print(B)
        print("\n--- Accumulated value of each PE (Result C) ---")
        print(C)
        
        
        
def simulate_systolic_array(matrix_A, matrix_B):
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
    latency = (rows_A - 1) + (cols_B - 1) + cols_A
    
    print("\n--- 4. Systolic Array Simulation ---")
    print(f"Input A shape: {matrix_A.shape}, Input B shape: {matrix_B.shape}")
    print("\nResult Matrix (C):")
    print(result_matrix)
    print("\n" + "="*40)
    print(f"Total Clock Cycles (Latency): {latency}\n")
    
def send_matrix_serial(serial_port, matrix):
    """Sends a matrix byte-by-byte over the serial port."""
    print(f"Sending {matrix.size} bytes...")
    
    # Flatten the matrix into a 1D array for easy iteration
    flat_matrix = matrix.flatten()
    
    # Send one byte at a time
    for byte_val in flat_matrix:
        serial_port.write(bytearray([byte_val]))
        # A small delay can help prevent buffer overflows on the receiver
        time.sleep(0.001)
        
    print("Matrix sent successfully.")    
    
def send_matrix_serial(serial_port, matrix):
    """Sends a matrix byte-by-byte over the serial port."""
    print(f"Sending {matrix.size} bytes...")
    
    # Flatten the matrix into a 1D array for easy iteration
    flat_matrix = matrix.flatten()
    
    # Send one byte at a time
    for byte_val in flat_matrix:
        serial_port.write(bytearray([byte_val]))
        # A small delay can help prevent buffer overflows on the receiver
        time.sleep(0.001)
        
    print("Matrix sent successfully.")
    
# --- Main Orchestration Function ---

def main():
    """
    Main function to orchestrate the model quantization, tensor extraction,
    .mif file generation, and final inference.
    """
    # 1. Load and Quantize the Model
    model_quantized = load_and_quantize_model()
    

    # 2. Load and Preprocess a Real Image
    input_tensor = preprocess_image(IMAGE_PATH)
    if input_tensor is None:
        return # Exit if image was not found

    # 3. Extract and process the relevant weight and activation tensors
    weights, activations = extract_and_process_tensors(model_quantized, input_tensor)

    # 4. Slice tensors, create tiles, and save as .mif files
    generate_and_save_mif_tiles(weights, activations, MIF_OUTPUT_DIR, LAYER_SIZE, TILE_SIZE)

    # 5. Run a full inference for reference and print the result
    labels = get_imagenet_labels()
    run_inference(model_quantized, input_tensor, labels)



    # Apply the stripping algorithm
    N_hardware = 8
    data_mif_path = 'C:/Users/iamkr/Documents/part-4-project/Final/mif/pipeline/activation_tile_4.mif'
    weight_mif_path = 'C:/Users/iamkr/Documents/part-4-project/Final/mif/pipeline/weight_tile_4.mif'
    # Load the matrices from the .mif files
    inputMatrix_data = mif_to_matrix(data_mif_path, N_hardware, N_hardware)
    inputMatrix_weight = mif_to_matrix(weight_mif_path, N_hardware, N_hardware)

    if inputMatrix_data is None or inputMatrix_weight is None:
        print("\nAborting due to file read error.")
    else:
        # --- CASE 1: WITH SPARSITY HANDLING (Coordinated Stripping) ---
        print("### VHDL FOR OPTIMIZED (SPARSITY) TEST ###\n")
        
        stripped_data, stripped_weight = strip_matrices(inputMatrix_data, inputMatrix_weight)

        generate_vhdl_stimulus("data", stripped_data, N=N_hardware)
        generate_vhdl_stimulus("weight", stripped_weight, N=N_hardware)

        # --- CASE 2: (Vanilla) ---
        print("\n### VHDL FOR UNOPTIMIZED (VANILLA) TEST ###\n")
        
        generate_vhdl_stimulus("data", inputMatrix_data, N=N_hardware)
        generate_vhdl_stimulus("weight", inputMatrix_weight, N=N_hardware)
        
    # simluate the systolic array operation 
    simulate_systolic_array(stripped_data, stripped_weight)
    
    
    # verify with mac calcuator
    mac_calculator(data_mif_path, weight_mif_path, 8,8);
    
    
    # 1. Open the serial port connection
    #    (Replace 'COM3' with the correct port for your DE1-SoC)
    try:
        ser = serial.Serial('COM3', 115200, timeout=1) # 115200 baud rate is common
        print(f"Opened serial port {ser.name}")
    except serial.SerialException as e:
        print(f"Error: Could not open serial port. {e}")
        return # Exit if the port can't be opened
    
    # 2. Send the matrices
    print("\n--- Sending Weight Matrix via UART ---")
    send_matrix_serial(ser, stripped_weight)

    print("\n--- Sending Activation Matrix via UART ---")
    send_matrix_serial(ser, stripped_data)

    # 4. Close the port
    ser.close()
    print("\nSerial port closed.")

if __name__ == '__main__':
    main()