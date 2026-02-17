# --- PASTE INTO SYSTEM CONSOLE ---
set bin_path "C:/Users/pchh520/Documents/GitHub/part-4-project/SummerResearch/Python/tile.bin"
set res_path "C:/Users/pchh520/Documents/GitHub/part-4-project/SummerResearch/Python/result.bin"

set master_path [lindex [get_service_paths master] 0]
if {[is_service_open master $master_path] == 0} { open_service master $master_path }

puts "WATCHER ACTIVE: Running in BLIND-FIRE mode (Ignoring Zeros)..."

while {1} {
    if {[file exists $bin_path]} {
        # 1. Get header from file
        set fp [open $bin_path r]
        fconfigure $fp -translation binary
        set header_data [read $fp 4]
        binary scan $header_data cccc m n k padding
        close $fp

        # 2. Write everything (MNK and Start Bit)
        master_write_from_file $master_path $bin_path 0x1000 
        master_write_32 $master_path 0x3004 $m
        master_write_32 $master_path 0x3008 $n
        master_write_32 $master_path 0x300C $k
        master_write_32 $master_path 0x3000 1 

        # 3. THE KEY: DO NOT READ THE STATUS. JUST WAIT.
        # This mimics your manual 'after 1000'
        puts "NPU Running... waiting for hardware latency..."
        after 2000

        set data [master_read_32 $master_path 0x2000 16]
        set idx 0
        foreach d $data {
            puts "Data Index $idx: $d"
            incr idx
        }
        
        set weight [master_read_32 $master_path 0x1000 16]
        set idx 0
        foreach w $weight {
            puts "Weight Index $idx: $w"
            incr idx
        }

        set k_check [master_read_32 $master_path 0x300C 1]
        puts "Hardware K-Register Check: $k_check"

        # 4. Grab results and delete the trigger file
        master_read_to_file $master_path $res_path 0x0000 [expr $m * $n * 2]
        set results [master_read_32 $master_path 0x0000 16]
        set idx 0
        foreach res $results {
            puts "Index $idx: $res"
            incr idx
        }
        file delete $bin_path
        puts "Tile Done."
    }
    after 200
}