import numpy as np

# --- 1. Define Input Matrices (A and B) ---

# Matrix A (Data/Activation): 32x32. Each row is identical: [1, 2, 3, ..., 32]
# Data Type: Use 64-bit integers (int64) to ensure no Python overflow, mirroring the VHDL 64-bit accumulator.
data_A = np.tile(np.arange(1, 33, dtype=np.int64), (32, 1))

# Matrix B (Weight): 32x32. Each column is identical: [[1], [2], [3], ..., [32]]
weight_B = np.tile(np.arange(1, 33, dtype=np.int64), (32, 1)).T

# --- 2. Perform Matrix Multiplication (C = A @ B) ---

result_C = data_A @ weight_B

# --- 3. Output Verification and Full Matrix Print ---

expected_cell_value = result_C[0, 0] # All elements are identical due to input pattern: 1^2 + 2^2 + ... + 32^2 = 11440

print("="*80)
print("             MATHEMATICAL VERIFICATION: FULL 32x32 MATRIX PRODUCT")
print("="*80)

print("\n--- Input Matrix A (Data/Activation): First 3x3 Block ---")
print(data_A[:3, :3])

print("\n--- Input Matrix B (Weight): First 3x3 Block ---")
print(weight_B[:3, :3])

print("\n--- FULL OUTPUT MATRIX C (Expected VHDL Result) ---")
# Print the full 32x32 result matrix
with np.printoptions(linewidth=1000, threshold=np.inf, formatter={'all': lambda x: f'{x:6d}'}):
    print(result_C)

print(f"\nExpected Value for ALL cells (Sum of Squares 1^2 to 32^2): {expected_cell_value}")
print(f"Observed Waveform Value (C[0][0]): 9455 (INCORRECT)")
print(f"Difference (11440 - 9455): 1985")
print("="*80)