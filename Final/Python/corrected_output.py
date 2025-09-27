import numpy as np
import re

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

def strip_and_get_indices(matrix_A, matrix_B):
    """
    Strips matrices and returns the stripped versions along with the indices
    of the non-zero rows/columns that were kept.
    """
    nonzero_rows_A = sorted(list({i for i in range(matrix_A.shape[0]) if np.any(matrix_A[i, :])}))
    nonzero_cols_B = sorted(list({j for j in range(matrix_B.shape[1]) if np.any(matrix_B[:, j])}))
    
    nonzero_cols_A = {j for j in range(matrix_A.shape[1]) if np.any(matrix_A[:, j])}
    nonzero_rows_B = {i for i in range(matrix_B.shape[0]) if np.any(matrix_B[i, :])}
    
    keep_indices = sorted(list(nonzero_cols_A.union(nonzero_rows_B)))
    
    stripped_A = matrix_A[nonzero_rows_A, :][:, keep_indices]
    stripped_B = matrix_B[:, nonzero_cols_B][keep_indices, :]
    
    return stripped_A, stripped_B, nonzero_rows_A, nonzero_cols_B

def reconstruct_matrix(stripped_result, kept_rows, kept_cols, original_shape):
    """
    Reconstructs the full output matrix from the stripped result and index lists.
    """
    # Create an all-zero matrix with the original, full dimensions
    full_matrix = np.zeros(original_shape, dtype=stripped_result.dtype)
    
    # Use the index lists to place the small result matrix into the correct locations
    full_matrix[np.ix_(kept_rows, kept_cols)] = stripped_result
    
    return full_matrix

# --- Main execution ---
if __name__ == '__main__':
    N_hardware = 8
    data_mif_path = 'C:/Users/iamkr/Documents/part-4-project/Final/mif/activation_tile_4.mif'
    weight_mif_path = 'C:/Users/iamkr/Documents/part-4-project/Final/mif/weight_tile_4.mif'

    original_A = mif_to_matrix(data_mif_path, N_hardware, N_hardware)
    original_B = mif_to_matrix(weight_mif_path, N_hardware, N_hardware)

    if original_A is None or original_B is None:
        print("\nAborting due to file read error.")
    else:
        print("--- OPTIMIZED (STRIPPED) MAC OPERATION & RECONSTRUCTION ---")
        
        # 1. Strip the matrices and get the indices of the kept rows/cols
        stripped_A, stripped_B, kept_rows_A, kept_cols_B = strip_and_get_indices(original_A, original_B)
        
        # 2. Perform the MAC operation on the smaller, stripped matrices
        stripped_C = np.matmul(stripped_A, stripped_B)
        
        # 3. Reconstruct the full output matrix
        reconstructed_C = reconstruct_matrix(stripped_C, kept_rows_A, kept_cols_B, original_A.shape)
        
        print("\nResult of Stripped MAC Operation (Stripped C):")
        print(stripped_C)
        print("\nReconstructed Full Output Matrix:")
        print(reconstructed_C)
        
        print("\n" + "="*50)
        
        # --- UNOPTIMIZED MAC OPERATION (FOR VERIFICATION) ---
        print("\n--- UNOPTIMIZED (ORIGINAL) MAC OPERATION ---")
        
        original_C = np.matmul(original_A, original_B)
        
        print("\nCorrect Full Output (Golden Reference):")
        print(original_C)