import numpy as np
import time
import re
import time


# --- Constants ---
IMAGE_PATH = 'C:/Users/iamkr/Documents/part-4-project/Final/Python/cat.jpg'

TEST_DATA_MIF_DIR = 'C:/Users/iamkr/Documents/part-4-project/Final/demo/master_data.mif'
TEST_WEIGHT_MIF_DIR = 'C:/Users/iamkr/Documents/part-4-project/Final/demo/master_weight.mif'
TILE_SIZE = 8

MASTER_DATA_TILE_0_START_POS = 0 
MASTER_DATA_TILE_1_START_POS = 3
MASTER_DATA_TILE_2_START_POS = 14
MASTER_DATA_TILE_3_START_POS = 33
MASTER_DATA_TILE_4_START_POS = 60
MASTER_DATA_TILE_5_START_POS = 95
MASTER_DATA_TILE_6_START_POS = 138
MASTER_DATA_TILE_7_START_POS = 189
MASTER_DATA_TILE_8_START_POS = 248
MASTER_WEIGHT_TILE_0_START_POS = 0
MASTER_WEIGHT_TILE_1_START_POS = 67
MASTER_WEIGHT_TILE_2_START_POS = 134
MASTER_WEIGHT_TILE_3_START_POS = 201
MASTER_WEIGHT_TILE_4_START_POS = 268
MASTER_WEIGHT_TILE_5_START_POS = 335
MASTER_WEIGHT_TILE_6_START_POS = 402
MASTER_WEIGHT_TILE_7_START_POS = 469
MASTER_WEIGHT_TILE_8_START_POS = 536

# put the constarnts in a list
DATA_TILE_START_POSITIONS = [
    MASTER_DATA_TILE_0_START_POS,
    MASTER_DATA_TILE_1_START_POS,
    MASTER_DATA_TILE_2_START_POS,   
    MASTER_DATA_TILE_3_START_POS,
    MASTER_DATA_TILE_4_START_POS,
    MASTER_DATA_TILE_5_START_POS,
    MASTER_DATA_TILE_6_START_POS,
    MASTER_DATA_TILE_7_START_POS,
    MASTER_DATA_TILE_8_START_POS
]
WEIGHT_TILE_START_POSITIONS = [
    MASTER_WEIGHT_TILE_0_START_POS,
    MASTER_WEIGHT_TILE_1_START_POS,
    MASTER_WEIGHT_TILE_2_START_POS,
    MASTER_WEIGHT_TILE_3_START_POS,
    MASTER_WEIGHT_TILE_4_START_POS,
    MASTER_WEIGHT_TILE_5_START_POS,
    MASTER_WEIGHT_TILE_6_START_POS,
    MASTER_WEIGHT_TILE_7_START_POS,
    MASTER_WEIGHT_TILE_8_START_POS
]

    
def mif_to_matrix(filename, start_pos, end_pos, is_data_matrix):
    """
    Reads a Quartus Memory Initialization File (.mif) and converts it
    into a 2D NumPy array with the specified dimensions (rows x cols).
    """
    data_values = []

    # Read the file between the starting and ending positions
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
                    match = re.search(r'\d+\s*:\s*(-?[0-9a-fA-F]+);', line)
                    if match:
                        hex_value = match.group(1)
                        data_values.append(int(hex_value, 16))
        m_value = data_values[start_pos]
        n_value = data_values[start_pos+1]
        k_value = data_values[start_pos+2]  
        matrix_only = data_values[start_pos+2:end_pos]
        
        
        # print(f"Loaded {len(matrix_only)} values from '{filename}'")

    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
        return None
    
    try:
        if (is_data_matrix):
            matrix = np.array(matrix_only, dtype=np.int32).reshape((m_value, k_value))
            return matrix
        else:
            matrix = np.array(matrix_only, dtype=np.int32).reshape((k_value, n_value))
            return matrix
    except ValueError as e:
        print(f"Error: Could not reshape data into a {k_value}x{k_value} matrix. {e}")
        return None



def simulate_systolic_array(matrix_A, matrix_B, m,n,k):
    """Simulates the behavior of a systolic array for matrix multiplication."""
    if matrix_A is None or matrix_B is None:
        return
        
    # start_time = time.time()
    start_time = time.perf_counter()
    rows_A, cols_A = matrix_A.shape
    rows_B, cols_B = matrix_B.shape
    
    if cols_A != rows_B:
        print("Error: Matrix dimensions are not compatible for multiplication.")
        return

    # Simulate the MAC operation
    result_matrix = np.matmul(matrix_A, matrix_B)
    end_time = time.perf_counter()

    # Calculate the latency (Total Clock Cycles)
    # latency = (rows_A - 1) + (cols_B - 1) + cols_A # !! change
    latency = m + n + k - 1
    
    # print("\n--- 4. Systolic Array Simulation ---")
    # print(f"Input A shape: {matrix_A.shape}, Input B shape: {matrix_B.shape}")
    # # print both input and output matrices
    # print("\nInput Matrix (A):")
    # print(matrix_A)
    # print("\nInput Matrix (B):")
    # print(matrix_B)
    # print("\nResult Matrix (C):")
    # print(result_matrix)
    # print(f"\nSimulated Total Clock Cycles (Latency): {latency}")
    # print(f"Active Rows (m): {m}, Active Columns (n): {n}, Active K (k): {k}")
    # print("\n" + "="*40)
    # print(f"Duration: {(end_time - start_time)*1000000:.2f} us")
    # print(f"Start Time: {start_time:.10f} s | End Time: {end_time:.10f} s | Duration: {(end_time - start_time)*1000:.2f} ms")
    # print the duration in nanoseconds
    # print(f"Duration: {(end_time - start_time)*1e9:.2f} ns")
    return (end_time - start_time)*1e9

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
    # print(f"\nAfter Coordinated Row Removal:")
    # print(f"Original Data Shape: {data_matrix.shape}, Original Weight Shape: {weight_matrix.shape}")
    # print(f"Compact Data Shape: {compact_data.shape}, Compact Weight Shape: {compact_weight.shape}")
    # print(f"Active Rows (m): {m_new}, Active Columns (n): {n_new}, Active K (k): {k_new}")

    return compact_data, compact_weight, m_new, k_new, n_new

def twos_complement_to_uint8(arr):
    return arr.astype(np.int8).astype(np.uint8)

def main():
    # Load each matrix from the master file 
    # run the simulation function 3 times and get an average of the time for each matrix
    run_X_times = 200
    count = 0
    for i in DATA_TILE_START_POSITIONS:
        time_array = []
        
        for j in range(run_X_times):
            if count == 8:
                testing_data = mif_to_matrix(TEST_DATA_MIF_DIR, MASTER_DATA_TILE_8_START_POS, MASTER_DATA_TILE_8_START_POS+66, True)
                testing_weight = mif_to_matrix(TEST_WEIGHT_MIF_DIR, MASTER_WEIGHT_TILE_8_START_POS, MASTER_WEIGHT_TILE_8_START_POS+66, False)
                stripped_data, stripped_weight, m_value, k_value, n_value = coordinated_row_removal(testing_data, testing_weight)
                time_taken = simulate_systolic_array(stripped_data, stripped_weight, m_value, n_value, k_value)  
                time_array.append(time_taken) 
            else:
                testing_data = mif_to_matrix(TEST_DATA_MIF_DIR, i, DATA_TILE_START_POSITIONS[DATA_TILE_START_POSITIONS.index(i)+1]-1, True)
                testing_weight = mif_to_matrix(TEST_WEIGHT_MIF_DIR, WEIGHT_TILE_START_POSITIONS[DATA_TILE_START_POSITIONS.index(i)], WEIGHT_TILE_START_POSITIONS[DATA_TILE_START_POSITIONS.index(i)+1]-1, False)
                stripped_data, stripped_weight, m_value, k_value, n_value = coordinated_row_removal(testing_data, testing_weight)
                time_taken = simulate_systolic_array(stripped_data, stripped_weight, m_value, n_value, k_value)  
                time_array.append(time_taken) 
        average_time = sum(time_array) / run_X_times
        print(f"Average Duration for tile {count} starting at {i}: {average_time:.2f} ns over {run_X_times} runs, where M = {m_value}, N = {n_value}, K = {k_value}")
        count += 1  
            
             
    

if __name__ == '__main__':
    main()