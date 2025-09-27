import numpy as np
import re

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

# --- Main execution ---
if __name__ == '__main__':
    # Define the file paths and matrix dimensions
    # data_mif_path = 'data.mif'
    # weight_mif_path = 'weights.mif'
    data_mif_path = 'C:/Users/iamkr/Documents/part-4-project/Final/mif/activation_tile_0.mif'
    weight_mif_path = 'C:/Users/iamkr/Documents/part-4-project/Final/mif/weight_tile_0.mif'
    
    ROWS, COLS = 8, 8
    
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