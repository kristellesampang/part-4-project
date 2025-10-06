import numpy as np
def coordinated_row_removal(data_matrix, weight_matrix):
    """
    This function correctly implements your original goal. It finds all active
    rows and columns for each matrix but coordinates the inner dimension 'k'
    to ensure the multiplication is always valid.
    """
    data_matrix = np.array(data_matrix)
    weight_matrix = np.array(weight_matrix)

    # 1. Find the active rows for data (m) and active columns for weight (n).
    active_m_indices = np.where(np.any(data_matrix, axis=1))[0]
    active_n_indices = np.where(np.any(weight_matrix, axis=0))[0]

    # 2. Find the active inner dimension 'k' by taking the UNION of active
    #    data columns and active weight rows. This captures all contributing parts.
    data_k_indices = np.where(np.any(data_matrix, axis=0))[0]
    weight_k_indices = np.where(np.any(weight_matrix, axis=1))[0]
    # Using a set union ensures we have a sorted list of unique indices
    common_k_indices = sorted(list(set(data_k_indices) | set(weight_k_indices)))

    # 3. Create the new, dense matrices by stripping all zero-axes using these indices.
    compact_data = data_matrix[np.ix_(active_m_indices, common_k_indices)]
    compact_weight = weight_matrix[np.ix_(common_k_indices, active_n_indices)]

    # 4. Extract the final, correct dimensions.
    m_new = compact_data.shape[0]
    k_new = compact_data.shape[1]
    n_new = compact_weight.shape[1]

    return compact_data, compact_weight, m_new, k_new, n_new

def generate_vhdl_stimulus(compact_data, compact_weight, m, k, n, N=8):
    """
    Generates VHDL 'constant' declarations for the compacted matrices.
    The padding is a VHDL requirement to fit the smaller logical matrix
    into the fixed-size 8x8 physical type.
    """
    print(f"-- VHDL stimulus for compacted matrices")
    print(f"constant ACTIVE_ROWS : integer := {m};")
    print(f"constant ACTIVE_K : integer := {k};")
    print(f"constant ACTIVE_COLS : integer := {n};")

    # Generate VHDL for the Data Matrix
    vhdl_data = f"\nconstant MATRIX_DATA_STIMULUS : systolic_array_matrix_input := (\n"
    for r in range(m):
        row_elements = [f"u8({compact_data[r, c]})" for c in range(k)]
        padding = [f"u8(0)"] * (N - k)
        vhdl_data += f"    ({', '.join(row_elements + padding)}),\n"
    vhdl_data += "    others => (others => u8(0))\n);"
    print(vhdl_data)

    # Generate VHDL for the Weight Matrix
    vhdl_weight = f"\nconstant MATRIX_WEIGHT_STIMULUS : systolic_array_matrix_input := (\n"
    for r in range(k):
        row_elements = [f"u8({compact_weight[r, c]})" for c in range(n)]
        padding = [f"u8(0)"] * (N - n)
        vhdl_weight += f"    ({', '.join(row_elements + padding)}),\n"
    vhdl_weight += "    others => (others => u8(0))\n);"
    print(vhdl_weight)
    print("-" * 40)

# Define hardware size
N_hardware = 8

# Define your original sparse matrices
# inputMatrix_data = np.array([
#     [0, 0, 0, 0, 0, 0, 0, 0],
#     [3, 0, 5, 1, 0, 2, 0, 0],
#     [0, 0, 0, 0, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0, 0, 0, 0],
#     [8, 0, 0, 6, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0, 0, 0, 0]
# ])
inputMatrix_data = np.array([
    [1,2,3],
    [0,0,0],
    [5,6,7],
])
inputMatrix_weight = np.array([
    [1, 2, 0],
    [5, 6, 0],
    [7, 8, 0],
])

# inputMatrix_weight = np.array([
#     [0, 9, 0, 0, 1, 0, 2, 0],
#     [0, 0, 0, 0, 0, 0, 0, 0],
#     [0, 7, 0, 0, 0, 0, 0, 0],
#     [0, 0, 0, 0, 4, 0, 0, 0],
#     [0, 0, 0, 0, 0, 0, 0, 0],
#     [0, 1, 0, 0, 0, 0, 5, 0],
#     [0, 0, 0, 0, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0, 0, 0, 0]
# ])


# --- CASE 1: OPTIMIZED SPARSITY HANDLING ---
print("### VHDL FOR OPTIMIZED (SPARSITY) TEST ###\n")
cd1, cw1, m1, k1, n1 = coordinated_row_removal(inputMatrix_data, inputMatrix_weight)
generate_vhdl_stimulus(cd1, cw1, m1, k1, n1, N=N_hardware)


# --- CASE 2: UNOPTIMIZED (VANILLA) ---
print("\n### VHDL FOR UNOPTIMIZED TEST ###\n")
print("-- VHDL stimulus for original un-optimized matrices (8x8x8)")
print(f"constant ACTIVE_M : integer := {N_hardware};")
print(f"constant ACTIVE_K : integer := {N_hardware};")
print(f"constant ACTIVE_N : integer := {N_hardware};")
print("-- NOTE: For this test, you would load the full, original matrices into BRAMs.")
print("-" * 40)
