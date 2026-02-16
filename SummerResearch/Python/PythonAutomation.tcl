# --- PASTE INTO SYSTEM CONSOLE ---
set bin_path "C:/Users/pchh520/Documents/GitHub/part-4-project/SummerResearch/Python/tile.bin"
set res_path "C:/Users/pchh520/Documents/GitHub/part-4-project/SummerResearch/Python/result.bin"

set master_path [lindex [get_service_paths master] 0]
if {[is_service_open master $master_path] == 0} { open_service master $master_path }

puts "WATCHER ACTIVE: Running in BLIND-FIRE mode (Ignoring Zeros)..."

while {1} {
    if {[file exists $bin_path]} {
        # 1. Read dimensions from file
        set fp [open $bin_path r]
        fconfigure $fp -translation binary
        set header_data [read $fp 4]
        binary scan $header_data cccc m n k padding
        close $fp

        # 2. Write Data & Weights
        master_write_from_file $master_path $bin_path 0x1000
        
        # 3. Write Dimensions (WE DO NOT READ THEM BACK)
        master_write_32 $master_path 0x3004 $m
        master_write_32 $master_path 0x3008 $n
        master_write_32 $master_path 0x300C $k
        master_write_32 $master_path 0x3000 1 

        # 4. HARD WAIT (Matching your successful manual test)
        # Since we can't 'see' the status register, we just wait.
        puts "NPU Triggered. Waiting 1000ms for calculation..."
        after 1000 

        # 5. Read Result
        master_read_to_file $master_path $res_path 0x0000 [expr $m * $n * 2]
        
        puts "Tile (M=$m, N=$n) Processed Successfully."
    }
    after 200
}