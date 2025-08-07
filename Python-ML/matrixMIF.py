import numpy as np

def save_mif(filename, data, width=8, depth=2048):
    flat_data = data.flatten().astype(np.uint8)
    if len(flat_data) < depth:
        flat_data = np.concatenate([flat_data, np.zeros(depth - len(flat_data), dtype=np.uint8)])
    else:
        flat_data = flat_data[:depth]
    
    with open(filename, "w") as f:
        f.write(f"WIDTH={width};\n")
        f.write(f"DEPTH={depth};\n")
        f.write("ADDRESS_RADIX=HEX;\n")
        f.write("DATA_RADIX=HEX;\n\n")
        f.write("CONTENT BEGIN\n")
        for addr, val in enumerate(flat_data):
            f.write(f"{addr:04X} : {val:02X};\n")
        f.write("END;\n")

# === STEP 1: Generate two 8×8 matrices ===
matrix_data = np.arange(1, 65, dtype=np.uint8).reshape(8, 8)         # [1, 2, ..., 64]
matrix_weight = np.flip(matrix_data, axis=1)                         # Horizontal flip for contrast

print("Matrix A (Data):")
print(matrix_data)
print("\nMatrix B (Weight):")
print(matrix_weight)

# === STEP 2: Save them to MIF files ===
save_mif("C:/Users/OEM/Documents/part-4-project/MIF_files/matrix_data.mif", matrix_data)
save_mif("C:/Users/OEM/Documents/part-4-project/MIF_files/matrix_weight.mif", matrix_weight)

print("\n✅ MIF files generated: 'matrix_data.mif' and 'matrix_weight.mif'")
