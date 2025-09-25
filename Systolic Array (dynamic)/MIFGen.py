import numpy as np
import os

def save_mif_32bit(filename, data, width=32):
    """
    Saves a NumPy array to a 32-bit wide MIF file, packing four 8-bit
    values into each 32-bit word.
    """
    # Flatten the 8x8 matrix into a 1D array of 64 bytes
    flat_data = data.flatten().astype(np.uint8)
    
    # Calculate the depth for a 32-bit memory (64 bytes / 4 bytes per word = 16 words)
    depth = len(flat_data) // 4

    with open(filename, "w") as f:
        # --- MIF Header ---
        f.write(f"WIDTH={width};\n")
        f.write(f"DEPTH={depth};\n")
        f.write("ADDRESS_RADIX=HEX;\n")
        f.write("DATA_RADIX=HEX;\n\n")
        f.write("CONTENT BEGIN\n")

        # --- MIF Content ---
        # Iterate through the flat data in chunks of 4 bytes
        for i in range(0, len(flat_data), 4):
            # Get a chunk of four 8-bit values
            chunk = flat_data[i:i+4]
            
            # Pack the four 8-bit bytes into a single 32-bit integer (big-endian)
            # Example: [9, 1, 0, 3] -> 0x09010003
            word = (chunk[0] << 24) | (chunk[1] << 16) | (chunk[2] << 8) | chunk[3]
            
            # The address for the 32-bit word
            addr = i // 4
            
            # Write the address and the 32-bit hex value to the file
            # :08X ensures the hex number is always padded to 8 digits (4 bytes)
            f.write(f"\t{addr:02X} : {word:08X};\n")
            
        f.write("END;\n")

# === Main script execution ===

# Create a directory for the MIF files if it doesn't exist
output_dir = "C:/Users/OEM\Documents/part-4-project/Systolic Array (dynamic)"
os.makedirs(output_dir, exist_ok=True)

# Generate two 8×8 matrices for testing
matrix_data = np.arange(1, 65, dtype=np.uint8).reshape(8, 8)      # [1, 2, ..., 64]
matrix_weight = np.flip(matrix_data, axis=1)                      # Horizontally flipped version

print("--- Generating 32-bit Wide MIF Files ---")
print("Matrix Data (first 4 values):", matrix_data.flatten()[:4])
print("Matrix Weight (first 4 values):", matrix_weight.flatten()[:4])

# Define the output file paths
data_mif_path = os.path.join(output_dir, "matrix_data_32bit.mif")
weight_mif_path = os.path.join(output_dir, "matrix_weight_32bit.mif")

# Save the matrices to the new 32-bit wide MIF files
save_mif_32bit(data_mif_path, matrix_data)
save_mif_32bit(weight_mif_path, matrix_weight)

print(f"\n✅ 32-bit MIF files generated in '{output_dir}' folder.")
print("You can now use these files to initialize your BRAMs in Platform Designer.")
