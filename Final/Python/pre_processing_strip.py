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

# --- Main execution ---

# Define paths to your .mif files
N_hardware = 8
data_mif_path = 'C:/Users/iamkr/Documents/part-4-project/Final/mif/activation_tile_4.mif'
weight_mif_path = 'C:/Users/iamkr/Documents/part-4-project/Final/mif/weight_tile_4.mif'

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