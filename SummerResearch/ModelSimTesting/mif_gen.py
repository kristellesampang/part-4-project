import os
from typing import List

# --- Configuration ---
BUFFER_DEPTH = 1024
ADDRESS_RADIX = "HEX"
DATA_RADIX = "HEX"
# Directory where the MIF files will be saved
OUTPUT_DIR = "C:/Users/OEM/Documents/part-4-project/SummerResearch/Memory/Buffers/MIFs" 

def generate_mif_content(width: int, data_value: int) -> str:
    """Generates the full MIF file content (header + data block)."""
    
    # Calculate hex string length (4 for 16-bit, 16 for 64-bit)
    hex_width = width // 4
    data_hex = f"{data_value:0{hex_width}X}"
    
    content = [
        f"WIDTH={width};",
        f"DEPTH={BUFFER_DEPTH};",
        "",
        f"ADDRESS_RADIX={ADDRESS_RADIX};",
        f"DATA_RADIX={DATA_RADIX};",
        "",
        "CONTENT BEGIN",
        f"\t[000..{BUFFER_DEPTH - 1:X}] : {data_hex};  -- Fill entire buffer with value {data_hex}",
        "END;"
    ]
    return "\n".join(content)

def write_mif_file(filename: str, content: str):
    """Writes the content string to a file in the specified output directory."""
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    filepath = os.path.join(OUTPUT_DIR, filename)
    with open(filepath, 'w') as f:
        f.write(content)
    print(f"Successfully created: {filepath}")


# ----------------------------------------------------------------------------
# --- GENERATOR FUNCTIONS ---
# ----------------------------------------------------------------------------

def generate_input_mifs():
    """Generates the four 16-bit input/weight buffers."""
    
    # 1. PING Buffers (Initial Tile: Data=1, Weight=3)
    write_mif_file(
        "DataBuffer1.mif", 
        generate_mif_content(width=16, data_value=1) # Value X"0001"
    )
    write_mif_file(
        "WeightBuffer1.mif", 
        generate_mif_content(width=16, data_value=3) # Value X"0003"
    )
    
    # 2. PONG Buffers (Next Tile: Data=2, Weight=4)
    write_mif_file(
        "DataBuffer2.mif", 
        generate_mif_content(width=16, data_value=2) # Value X"0002"
    )
    write_mif_file(
        "WeightBuffer2.mif", 
        generate_mif_content(width=16, data_value=4) # Value X"0004"
    )

def generate_output_mif():
    """Generates the single 64-bit output result buffer (initializes to zero)."""
    
    # Output buffer must be 64-bit wide and initialized to zero
    write_mif_file(
        "OutputBuffer.mif", 
        generate_mif_content(width=64, data_value=0) # Value X"00...00"
    )

def generate_all_mif_files():
    """Main function to run all generators."""
    generate_input_mifs()
    generate_output_mif()
    
# ----------------------------------------------------------------------------
# --- MAIN EXECUTION BLOCK ---
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    generate_all_mif_files()