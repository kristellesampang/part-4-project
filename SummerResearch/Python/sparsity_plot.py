import torch
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import numpy as np
import torch.nn.utils.prune as prune
import time
import json
import requests
import os
import matplotlib.pyplot as plt

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

def plot_matrix_heatmap(matrix, title):
    """
    Visualizes a matrix as a heatmap to show its sparsity pattern.
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    # Use a binary colormap: black for zero, white for non-zero
    ax.imshow(matrix != 0, cmap='gray', interpolation='nearest')
    ax.set_title(title, fontsize=16)
    plt.xlabel("Matrix Columns")
    plt.ylabel("Matrix Rows")
    plt.tight_layout()
    plt.show()

def main():
    """
    Main function to load AlexNet, and visualize unstructured vs. structured sparsity.
    """
    # 1. --- Load the Model ---
    print("Loading AlexNet model...")
    model = models.alexnet(weights=models.AlexNet_Weights.DEFAULT)
    model.eval()
    
    # --- 2. Visualize Unstructured Sparsity (Before Pruning) ---
    print("\n--- VISUALIZING UNSTRUCTURED SPARSITY ---")
    
    # Target the first fully connected layer
    fc1_layer = model.classifier[1]
    
    # Get the original weights and apply ReLU to create unstructured sparsity
    original_weights = fc1_layer.weight.detach().numpy()
    weights_after_relu = np.maximum(0, original_weights)
    
    # Calculate and print the sparsity
    non_zeros = np.count_nonzero(weights_after_relu)
    total_elements = weights_after_relu.size
    sparsity = 1 - (non_zeros / total_elements)
    plot_title_before = f'Unstructured Sparsity (After ReLU)\nSparsity: {sparsity:.2%}'
    
    print("Generating heatmap for weights with unstructured sparsity...")
    plot_matrix_heatmap(weights_after_relu, plot_title_before)
    
    # --- 3. Apply Structured Pruning ---
    print("\n--- APPLYING STRUCTURED PRUNING ---")
    
    # We will prune 40% of the rows (output neurons) from the layer
    prune.ln_structured(
        module=fc1_layer, 
        name='weight', 
        amount=0.4, # Prune 40%
        n=1, # L1 norm
        dim=0 # Pruning along dimension 0 removes entire rows (neurons)
    )
    
    # To make the pruning permanent, we remove the re-parameterization
    prune.remove(fc1_layer, 'weight')
    print("Pruning complete. 40% of rows have been removed.")

    # --- 4. Visualize Structured Sparsity (After Pruning) ---
    print("\n--- VISUALIZING STRUCTURED SPARSITY ---")

    # Get the new, pruned weights and apply ReLU
    pruned_weights = fc1_layer.weight.detach().numpy()
    pruned_weights_after_relu = np.maximum(0, pruned_weights)
    
    # Calculate and print the new sparsity
    non_zeros_pruned = np.count_nonzero(pruned_weights_after_relu)
    total_elements_pruned = pruned_weights_after_relu.size
    sparsity_pruned = 1 - (non_zeros_pruned / total_elements_pruned)
    plot_title_after = f'Structured Sparsity (After Pruning 40% of Rows)\nSparsity: {sparsity_pruned:.2%}'
    
    print("Generating heatmap for weights with structured sparsity...")
    plot_matrix_heatmap(pruned_weights_after_relu, plot_title_after)


if __name__ == '__main__':
    main()