import numpy as np

# --- Your original function ---
def RowRemoval(matrix):
    matrix = np.array(matrix)
    nonzeroRows = [i for i in range(matrix.shape[0]) if np.any(matrix[i, :] != 0)]
    nonzeroCols = [j for j in range(matrix.shape[1]) if np.any(matrix[:, j] != 0)]
    stripped = matrix[np.ix_(nonzeroRows, nonzeroCols)]
    return stripped, len(nonzeroRows), len(nonzeroCols)

# --- Your VHDL generation function (unchanged) ---
def generate_vhdl_stimulus(matrix_type, matrix_to_print, active_rows, active_cols, N=8):
    """
    Generates VHDL 'constant' declarations for a given matrix stimulus.
    'matrix_type' should be "data" or "weight".
    """
    
    print(f"-- VHDL stimulus for {matrix_type} matrix")
    # Use upper() to create constant names like ACTIVE_ROWS_DATA
    print(f"constant ACTIVE_ROWS_{matrix_type.upper()} : integer := {active_rows};")
    print(f"constant ACTIVE_COLS_{matrix_type.upper()} : integer := {active_cols};")
    
    vhdl = f"constant MATRIX_{matrix_type.upper()}_STIMULUS : systolic_array_matrix_input := (\n"
    
    rows_to_print, cols_to_print = matrix_to_print.shape

    for r in range(rows_to_print):
        row_elements = [f"u8({matrix_to_print[r, c]})" for c in range(cols_to_print)]
        padding = [f"u8(0)"] * (N - cols_to_print)
        vhdl += f"    ({', '.join(row_elements + padding)}),\n"
        
    vhdl += "    others => (others => u8(0))\n"
    vhdl += ");"
    
    print(vhdl)
    print("-" * 30)

# Define original sparse matrices in Python
N_hardware = 8
inputMatrix_data = np.array([[0, 0, 0, 0, 0, 0, 0, 0],
                             [3, 0, 5, 1, 0, 2, 0, 0], 
                             [0, 0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0, 0],
                             [8, 0, 0, 6, 0, 0, 0, 0],  
                             [0, 0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0, 0]])


inputMatrix_weight = np.array([[0, 9, 0, 0, 1, 0, 2, 0], 
                               [0, 0, 0, 0, 0, 0, 0, 0],
                               [0, 7, 0, 0, 0, 0, 0, 0], 
                               [0, 0, 0, 0, 4, 0, 0, 0], 
                               [0, 0, 0, 0, 0, 0, 0, 0],
                               [0, 1, 0, 0, 0, 0, 5, 0], 
                               [0, 0, 0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0, 0]])


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