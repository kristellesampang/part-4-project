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

def RowRemoval(matrix):
    matrix = np.array(matrix)
    nonzeroRows = [i for i in range(matrix.shape[0]) if np.any(matrix[i, :] != 0)]
    nonzeroCols = [j for j in range(matrix.shape[1]) if np.any(matrix[:, j] != 0)]
    stripped = matrix[np.ix_(nonzeroRows, nonzeroCols)]
    return stripped, len(nonzeroRows), len(nonzeroCols)

def generate_vhdl_stimulus(matrix_type, matrix_to_print, active_rows, active_cols, N=8):
    """
    Generates VHDL 'constant' declarations for a given matrix stimulus.
    'matrix_type' should be "data" or "weight".
    """
    
    print(f"-- VHDL stimulus for {matrix_type} matrix")
    print(f"constant ACTIVE_ROWS_{matrix_type.upper()} : integer := {active_rows};")
    print(f"constant ACTIVE_COLS_{matrix_type.upper()} : integer := {active_cols};")
    
    vhdl = f"constant MATRIX_{matrix_type.upper()}_STIMULUS : systolic_array_matrix_input := (\n"
    
    # Handle cases where the matrix might be empty after stripping
    if matrix_to_print.size == 0:
        rows_to_print, cols_to_print = 0, 0
    else:
        rows_to_print, cols_to_print = matrix_to_print.shape

    for r in range(rows_to_print):
        row_elements = [f"u8({matrix_to_print[r, c]})" for c in range(cols_to_print)]
        padding = [f"u8(0)"] * (N - cols_to_print)
        vhdl += f"    ({', '.join(row_elements + padding)}),\n"
        
    vhdl += "    others => (others => u8(0))\n"
    vhdl += ");"
    
    print(vhdl)
    print("-" * 30)

# --- Main execution ---

# Define paths to your .mif files
N_hardware = 8
data_mif_path = 'C:/Users/iamkr/Documents/part-4-project/Final/mif/activation_tile_0.mif'
weight_mif_path = 'C:/Users/iamkr/Documents/part-4-project/Final/mif/weight_tile_0.mif'

# Load the matrices from the .mif files
inputMatrix_data = mif_to_matrix(data_mif_path, N_hardware, N_hardware)
inputMatrix_weight = mif_to_matrix(weight_mif_path, N_hardware, N_hardware)

if inputMatrix_data is None or inputMatrix_weight is None:
    print("\nAborting due to file read error.")
else:
    # --- CASE 1: WITH SPARSITY HANDLING (Row Stripping) ---
    print("### VHDL FOR OPTIMIZED (SPARSITY) TEST ###\n")
    dense_data, rows_data, cols_data = RowRemoval(inputMatrix_data)
    dense_weight, rows_weight, cols_weight = RowRemoval(inputMatrix_weight)

    # Generate the VHDL code using the DENSE matrices and their new, smaller dimensions
    generate_vhdl_stimulus("data", dense_data, rows_data, cols_data, N=N_hardware)
    generate_vhdl_stimulus("weight", dense_weight, rows_weight, cols_weight, N=N_hardware)

    # --- CASE 2: (Vanilla) ---
    print("\n### VHDL FOR UNOPTIMIZED (VANILLA) TEST ###\n")
    # Generate VHDL for the original, un-stripped matrices using the full hardware dimensions
    generate_vhdl_stimulus("data", inputMatrix_data, N_hardware, N_hardware, N=N_hardware)
    generate_vhdl_stimulus("weight", inputMatrix_weight, N_hardware, N_hardware, N=N_hardware)