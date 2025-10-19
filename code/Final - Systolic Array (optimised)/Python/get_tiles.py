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

def main():
    """
    Main function to load AlexNet, quantize it, generate .mif files, and make an inference.
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
    image_path = 'C:/Users/iamkr/Documents/part-4-project/Final/Python/cat.jpg'
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

    # 4. --- Extract and Apply ReLU to Full Matrices ---
    quantized_weight_tensor = model_quantized.model_fp32.classifier[1].weight()
    fc1_weights_int8 = quantized_weight_tensor.int_repr().numpy()
    weights_after_relu = np.maximum(0, fc1_weights_int8)
    
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
    activations_after_relu = np.maximum(0, activation_matrix_int8)

    # --- 5. Slice Matrices to 64x64 for this Test ---
    print("\n--- SLICING MATRICES TO 64x64 ---")
    LAYER_SIZE = 64
    weights_64x64 = weights_after_relu[0:LAYER_SIZE, 0:LAYER_SIZE]
    activations_1x64 = activations_after_relu[:, 0:LAYER_SIZE]
    print(f"Created a {weights_64x64.shape} weight matrix.")
    print(f"Created a {activations_1x64.shape} activation matrix.")

    # --- 6. Generate 8x8 Tiles from 64x64 Slice and Save as .mif ---
    print("\n--- GENERATING 8x8 TILES AND SAVING AS .MIF ---")
    
    TILE_SIZE = 8
    mif_output_dir = "C:/Users/iamkr/Documents/part-4-project/Final/mif"
    
    if not os.path.exists(mif_output_dir):
        os.makedirs(mif_output_dir)
    
    tile_counter = 0
    # Iterate over the 64x64 matrix in 8x8 chunks
    for c in range(0, LAYER_SIZE, TILE_SIZE):
        # We only need to iterate over columns since the weight slice is 64 rows high
        w_tile = weights_64x64[0:TILE_SIZE, c:c+TILE_SIZE]
        
        a_vector_slice = activations_1x64[:, c:c+TILE_SIZE]
        a_tile = np.zeros((TILE_SIZE, TILE_SIZE), dtype=weights_64x64.dtype)
        a_tile[0, :] = a_vector_slice
        
        # Save the tiles to .mif files
        save_matrix_as_mif(w_tile, os.path.join(mif_output_dir, f"weight_tile_{tile_counter}.mif"))
        save_matrix_as_mif(a_tile, os.path.join(mif_output_dir, f"activation_tile_{tile_counter}.mif"))
        tile_counter += 1

    print(f"\nGenerated and saved {tile_counter} pairs of 8x8 tiles.")
    print(f"Files are saved in the '{mif_output_dir}' directory.")

    # --- 7. Full Model Inference ---
    print("\n--- FULL MODEL INFERENCE (FOR REFERENCE) ---")
    
    labels = get_imagenet_labels()
    
    output_logits = model_quantized(input_tensor)
    
    prediction_index = torch.argmax(output_logits, 1).item()
    
    probabilities = torch.nn.functional.softmax(output_logits, dim=1)
    confidence = probabilities[0, prediction_index].item() * 100
    
    predicted_class = labels[str(prediction_index)][1]
    
    print(f"\nPrediction: {predicted_class.replace('_', ' ').title()}")
    print(f"Confidence: {confidence:.2f}%")


if __name__ == '__main__':
    main()