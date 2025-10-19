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

# Wrapper class remains the same
class QuantizableAlexNet(torch.nn.Module):
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
    """
    Downloads and loads the ImageNet class labels.
    """
    labels_path = 'imagenet_class_index.json'
    labels_url = 'https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json'

    if not os.path.exists(labels_path):
        print(f"Downloading ImageNet labels from {labels_url}...")
        response = requests.get(labels_url)
        with open(labels_path, 'w') as f:
            json.dump(response.json(), f)
        print("Labels downloaded.")

    with open(labels_path) as f:
        labels = json.load(f)
    return labels

def save_matrix_as_mif(matrix, filename, data_width=8):
    """
    Saves a 2D NumPy matrix into a Quartus Memory Initialization File (.mif).
    """
    depth = matrix.size
    with open(filename, 'w') as f:
        # print the matrix values as an array for verification
        print("Matrix values:")
        print(matrix)
        # Write the MIF header
        f.write(f"WIDTH={data_width};\n")
        f.write(f"DEPTH={depth};\n")
        f.write("ADDRESS_RADIX=UNS;\n")
        f.write("DATA_RADIX=HEX;\n\n")
        f.write("CONTENT BEGIN\n")

        # Flatten the matrix and write the data
        # Data is stored in row-major order
        flat_matrix = matrix.flatten()
        for i, val in enumerate(flat_matrix):
            # Format the value as a two-character hex string (e.g., 0A, 1F)
            hex_val = f"{val:02X}"
            f.write(f"\t{i}\t:\t{hex_val};\n")
        
        f.write("END;\n")

def main():
    """
    Main function to load AlexNet, quantize it, extract matrices, and make an inference.
    """
    # 1. & 2. --- Load and Quantize the Model ---
    print("Loading and quantizing AlexNet model...")
    alexnet_fp32 = models.alexnet(weights=models.AlexNet_Weights.DEFAULT)
    quantizable_model = QuantizableAlexNet(model_fp32=alexnet_fp32)
    quantizable_model.eval()
    quantizable_model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    model_prepared = torch.quantization.prepare(quantizable_model, inplace=False)
    with torch.no_grad():
        for _ in range(10):
            model_prepared(torch.randn(1, 3, 224, 224))
    model_quantized = torch.quantization.convert(model_prepared, inplace=False)
    print("Model successfully quantized.")

    # 3. --- Load and Preprocess a Real Image ---
    image_path = 'C:/Users/iamkr/Documents/part-4-project/Python-ML/cat.jpg'
    try:
        img = Image.open(image_path).convert('RGB')
    except FileNotFoundError:
        print(f"ERROR: The image file '{image_path}' was not found. Please add a 'cat.jpg' to your directory.")
        return
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(img).unsqueeze(0)

    # 4. --- Extract RAW INT8 Matrices ---
    quantized_weight_tensor = model_quantized.model_fp32.classifier[1].weight()
    fc1_weights_int8 = quantized_weight_tensor.int_repr().numpy()
    
    activation = {}
    def get_quantized_activation(name):
        def hook(model, input, output):
            activation[name] = output
        return hook
    hook_handle = model_quantized.model_fp32.avgpool.register_forward_hook(get_quantized_activation('avgpool'))
    with torch.no_grad():
        model_quantized(input_tensor)
    hook_handle.remove()
    quantized_activation_tensor = activation['avgpool']
    activation_matrix_int8 = torch.flatten(quantized_activation_tensor.int_repr(), 1).numpy()
    
    # 5. --- Apply ReLU ---
    weights_after_relu = np.maximum(0, fc1_weights_int8)
    activations_after_relu = np.maximum(0, activation_matrix_int8)

    # 6. --- 8x8 TILING DEMO ---
    print("\n--- 8x8 TILING DEMO (First 8 Tiles of Weight Matrix) ---")
    tile_size = 8
    num_tiles_to_print = 8

    for i in range(num_tiles_to_print):
        start_col = i * tile_size
        end_col = start_col + tile_size
        weight_tile = weights_after_relu[0:tile_size, start_col:end_col]
        print(f"\n--- Tile {i+1} (Cols {start_col}-{end_col-1}) ---")
        print(weight_tile)
        # save as .mif file in this folder C:\Users\iamkr\Documents\part-4-project\Systolic Array (packing)\mif
        save_matrix_as_mif(weight_tile, f'C:/Users/iamkr/Documents/part-4-project/Systolic Array (packing)/mif/weight_tile_{i+1}.mif')
        # save_matrix_as_mif(weight_tile, f'weight_tile_{i+1}.mif')

    # --- 7. 8x8 Weight by 8x1 Activation Multiplication Demo ---
    print("\n--- 8x8 by 8x1 MATRIX MULTIPLICATION DEMO ---")
    #wsave the activation tile as a .mif file
    activation_tile = np.zeros((tile_size, tile_size), dtype=weights_after_relu.dtype)
    activation_tile[0, :] = activations_after_relu[:, 0:tile_size]
    # save the .mif file in this folder C:\Users\iamkr\Documents\part-4-project\Systolic Array (packing)\mif
    save_matrix_as_mif(activation_tile, 'C:/Users/iamkr/Documents/part-4-project/Systolic Array (packing)/mif/activation_tile.mif')
    # save_matrix_as_mif(activation_tile, 'activation_tile.mif')

    weight_matrix_8x8 = weights_after_relu[0:tile_size, 0:tile_size]
    activation_vector_8x1 = activations_after_relu[:, 0:tile_size].T

    print(f"Multiplying Weight Matrix of shape: {weight_matrix_8x8.shape}")
    print(f"with Activation Vector of shape: {activation_vector_8x1.shape}")

    start_ns = time.perf_counter_ns()
    result_vector = np.matmul(weight_matrix_8x8, activation_vector_8x1)
    end_ns = time.perf_counter_ns()
    execution_time_ns = end_ns - start_ns
    execution_time_seconds = execution_time_ns / 1_000_000_000

    print(f"\nResulting vector shape: {result_vector.shape}")
    print(f"Resulting vector (first 8 elements):\n{result_vector.flatten()}")
    print("\n--- Timing Analysis ---")
    print(f"Execution time in 'ticks' (nanoseconds): {execution_time_ns} ns")
    print(f"Converted to seconds: {execution_time_seconds:.8f} s")

    # --- 8. Full Model Inference ---
    print("\n--- FULL MODEL INFERENCE ---")
    
    # Load the class labels
    labels = get_imagenet_labels()
    
    # Get the raw output scores (logits) from the model
    output_logits = model_quantized(input_tensor)
    
    # Find the index of the highest score
    prediction_index = torch.argmax(output_logits, 1).item()
    
    # Convert logits to probabilities to get a confidence score
    probabilities = torch.nn.functional.softmax(output_logits, dim=1)
    confidence = probabilities[0, prediction_index].item() * 100
    
    # Look up the class name from the index
    predicted_class = labels[str(prediction_index)][1]
    
    print(f"\nPrediction: {predicted_class.replace('_', ' ').title()}")
    print(f"Confidence: {confidence:.2f}%")


if __name__ == '__main__':
    main()