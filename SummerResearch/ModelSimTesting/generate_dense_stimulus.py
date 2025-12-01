import numpy as np

# --- GLOBAL CONFIGURATION ---
# The physical dimension N of the hardware array (from custom_types.vhd)
N_HARDWARE = 32

def u16(x):
    """Helper function to format an integer as a VHDL 16-bit constant string."""
    # VHDL must use to_signed(x, 16) for proper logic; this format provides the source data.
    return f"u16({int(x)})"

def generate_vhdl_dense_stimulus():
    """
    Generates VHDL 'constant' declarations for a full 32x32x32 dense multiplication.
    """
    # 1. Define Dense Matrices (32x32, 32x32)
    # Matrix A (Data/Activation): M x K, using a pattern of [1, 2, ..., 32] tiled across all rows.
    data_A = np.tile(np.arange(1, N_HARDWARE + 1, dtype=np.int16), (N_HARDWARE, 1))

    # Matrix B (Weight): K x N, using a pattern of [[1], [2], ..., [32]] tiled across all columns.
    weight_B = np.tile(np.arange(1, N_HARDWARE + 1, dtype=np.int16), (N_HARDWARE, 1)).T

    # 2. Calculate Expected Result C = A x B
    expected_result_C = np.dot(data_A, weight_B)
    
    # Expected constant value for every element: Sum of squares 1^2 to 32^2 = 11440
    EXPECTED_C_VALUE = expected_result_C[0][0]
    
    # 3. Generate VHDL Stimulus Constants
    
    print(f"--- VHDL CONSTANTS FOR DENSE TEST (Copy Below) ---")
    print(f"constant ACTIVE_ROWS : integer := {N_HARDWARE};")
    print(f"constant ACTIVE_K : integer := {N_HARDWARE};")
    print(f"constant ACTIVE_COLS : integer := {N_HARDWARE};")
    print(f"constant EXPECTED_LATENCY : integer := {N_HARDWARE + N_HARDWARE + N_HARDWARE - 2}; -- {N_HARDWARE + N_HARDWARE + N_HARDWARE - 2} Cycles\n")

    # --- Generate VHDL for the Data Matrix (A) ---
    print(f"constant MATRIX_DATA_STIMULUS : systolic_array_matrix_input := (")
    for r in range(N_HARDWARE):
        row_elements = [u16(data_A[r, c]) for c in range(N_HARDWARE)]
        print(f"    ({', '.join(row_elements)}),")
    print(f"    others => (others => u16(0))")
    print(f");")

    # --- Generate VHDL for the Weight Matrix (B) ---
    print(f"\nconstant MATRIX_WEIGHT_STIMULUS : systolic_array_matrix_input := (")
    for r in range(N_HARDWARE):
        row_elements = [u16(weight_B[r, c]) for c in range(N_HARDWARE)]
        print(f"    ({', '.join(row_elements)}),")
    print(f"    others => (others => u16(0))")
    print(f");")

    print("\n" + "="*70)
    print("### EXPECTED RESULT ###")
    print(f"All {N_HARDWARE}x{N_HARDWARE} output elements must equal: {EXPECTED_C_VALUE}")
    print(f"Latency must equal: {N_HARDWARE + N_HARDWARE + N_HARDWARE - 2} cycles.")
    print("="*70)

# --- EXECUTION ---
if __name__ == "__main__":
    generate_vhdl_dense_stimulus()