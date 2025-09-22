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

def save_matrix_as_mif(matrix, filename, data_width=8):
    """
    Saves a 2D NumPy matrix into a Quartus Memory Initialization File (.mif).
    """
    depth = matrix.size
    with open(filename, 'w') as f:
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
    Main function to load AlexNet, quantize it, and generate .mif files for hardware.
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
    image_path = 'cat.jpg'
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

    # --- 6. Generate 8x8 Tiles and Save as .mif ---
    print("\n--- GENERATING 8x8 TILES AND SAVING AS .MIF ---")
    
    TILE_SIZE = 8
    
    # For this example, we'll just take the very first 8x8 tile
    weight_tile_0 = weights_after_relu[0:TILE_SIZE, 0:TILE_SIZE]
    
    # Create the corresponding 8x8 activation tile
    activation_vector_slice = activations_after_relu[:, 0:TILE_SIZE]
    activation_tile_0 = np.zeros((TILE_SIZE, TILE_SIZE), dtype=weights_after_relu.dtype)
    activation_tile_0[0, :] = activation_vector_slice

    # Save the tiles to .mif files
    save_matrix_as_mif(weight_tile_0, "weights.mif")
    print("\nSaved the first weight tile to 'weights.mif'")
    
    save_matrix_as_mif(activation_tile_0, "activations.mif")
    print("Saved the first activation tile to 'activations.mif'")
    print("\nYou can now use these files to initialize the BRAMs in your Quartus project.")


if __name__ == '__main__':
    main()