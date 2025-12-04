import numpy as np

# --- GLOBAL CONFIGURATION ---
N_HARDWARE = 32
N_LOGICAL = 32 # We test the full size of the original matrix before pruning.

def u16(x):
    """Helper function to format an integer as a VHDL 16-bit constant string."""
    # Use standard formatting for positive integers for display in VHDL
    return f"u16({int(x)})"

def coordinated_sparsity_removal(data_matrix, weight_matrix):
    """
    Finds active rows/cols/K indices across both matrices, strips zeros,
    and returns the compacted matrices and dimensions (M', K', N').
    """
    data_matrix = np.array(data_matrix)
    weight_matrix = np.array(weight_matrix)

    # 1. Find the active indices for the outer dimensions (M and N)
    active_m_indices = np.where(np.any(data_matrix, axis=1))[0]
    active_n_indices = np.where(np.any(weight_matrix, axis=0))[0]

    # 2. Find the active indices for the inner dimension (K)
    data_k_indices = np.where(np.any(data_matrix, axis=0))[0]
    weight_k_indices = np.where(np.any(weight_matrix, axis=1))[0]
    common_k_indices = sorted(list(set(data_k_indices) | set(weight_k_indices)))

    if not common_k_indices:
        return np.array([[]]), np.array([[]]), 0, 0, 0, 0

    # 3. Create the new, dense matrices by indexing with the coordinated indices.
    compact_data = data_matrix[np.ix_(active_m_indices, common_k_indices)]
    compact_weight = weight_matrix[np.ix_(common_k_indices, active_n_indices)]

    # 4. Extract the final, compact dimensions.
    m_new = compact_data.shape[0]
    k_new = compact_data.shape[1]
    n_new = compact_weight.shape[1]

    # Calculate the expected result matrix product
    expected_result = np.dot(compact_data, compact_weight)

    return compact_data, compact_weight, m_new, k_new, n_new, expected_result

def generate_vhdl_stimulus(compact_data, compact_weight, m, k, n, expected_result):
    """
    Generates VHDL 'constant' declarations for the compacted (dense) matrices.
    """
    # Pad the compacted matrices back up to the N_HARDWARE size (32x32) with zeros.
    # The NPU will only use the m x k and k x n parts.
    
    # 1. Data Matrix (M x K) Padding
    vhdl_data = f"\nconstant MATRIX_DATA_STIMULUS : systolic_array_matrix_input := (\n"
    for r in range(N_HARDWARE):
        row_elements = []
        if r < m:
            # Load M' rows and K' columns
            row_elements.extend([u16(compact_data[r, c]) for c in range(k)])
            # Pad the rest of the K dimension (N_HARDWARE - K')
            row_elements.extend([u16(0)] * (N_HARDWARE - k))
        else:
            # Pad the remaining rows (N_HARDWARE - M')
            row_elements.extend([u16(0)] * N_HARDWARE)
            
        vhdl_data += f"    ({', '.join(row_elements)})," if r < N_HARDWARE - 1 else f"    ({', '.join(row_elements)})"
        vhdl_data += "\n"
    vhdl_data += ");"
    
    # 2. Weight Matrix (K x N) Padding
    vhdl_weight = f"\nconstant MATRIX_WEIGHT_STIMULUS : systolic_array_matrix_input := (\n"
    for r in range(N_HARDWARE):
        row_elements = []
        if r < k:
            # Load K' rows and N' columns
            row_elements.extend([u16(compact_weight[r, c]) for c in range(n)])
            # Pad the rest of the N dimension (N_HARDWARE - N')
            row_elements.extend([u16(0)] * (N_HARDWARE - n))
        else:
            # Pad the remaining rows (N_HARDWARE - K')
            row_elements.extend([u16(0)] * N_HARDWARE)
            
        vhdl_weight += f"    ({', '.join(row_elements)})," if r < N_HARDWARE - 1 else f"    ({', '.join(row_elements)})"
        vhdl_weight += "\n"
    vhdl_weight += ");"
    
    print(f"### VHDL CONSTANTS FOR SPARSE TEST (M'={m}, K'={k}, N'={n}) ###\n")
    print(f"constant ACTIVE_ROWS : integer := {m};")
    print(f"constant ACTIVE_K : integer := {k};")
    print(f"constant ACTIVE_COLS : integer := {n};")
    print(f"constant EXPECTED_LATENCY : integer := {m+n+k-2};")
    
    print(vhdl_data)
    print(vhdl_weight)
    
    print("\n" + "="*70)
    print("### EXPECTED OUTPUT MATRIX (Compact M' x N') ###")
    print(f"Latency Reduction Achieved: {N_HARDWARE+N_HARDWARE+N_HARDWARE-2} -> {m+n+k-2} cycles.")
    print("Output Size (M'xN'):", expected_result.shape)
    print(expected_result)
    print("="*70)


# --- TEST CASE DEFINITION (A 32x32 matrix, approx 50% sparse) ---

# Define values for the dense regions (ensuring 16-bit safe values)
VAL_A = 10 
VAL_B = 5

# Create matrices initially filled with zeros
A_sparse = np.zeros((N_LOGICAL, N_LOGICAL), dtype=np.int64)
B_sparse = np.zeros((N_LOGICAL, N_LOGICAL), dtype=np.int64)

# Create sparse pattern (rows 0-15 active, columns 0-20 active)
# Active M' = 16 rows, Active N' = 21 cols, Active K' = 21 columns
A_sparse[0:16, 0:21] = VAL_A 
B_sparse[0:21, 0:21] = VAL_B 

# Execution
compact_data, compact_weight, m, k, n, expected_result = coordinated_sparsity_removal(A_sparse, B_sparse)
generate_vhdl_stimulus(compact_data, compact_weight, m, k, n, expected_result)