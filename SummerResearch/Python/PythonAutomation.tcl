# --- PASTE INTO SYSTEM CONSOLE ---
set bin_path "C:/Users/pchh520/Documents/GitHub/part-4-project/SummerResearch/Python/tile.bin"
set res_path "C:/Users/pchh520/Documents/GitHub/part-4-project/SummerResearch/Python/result.bin"

set master_path [lindex [get_service_paths master] 0]
if {[is_service_open master $master_path] == 0} { open_service master $master_path }

puts "WATCHER ACTIVE..."

while {1} {
    if {[file exists $bin_path]} {
        # 1. READ HEADER FIRST
        set fp [open $bin_path r]
        fconfigure $fp -translation binary
        set header_data [read $fp 4]
        binary scan $header_data cccc m n k padding
        close $fp

        # 2. THE FIX: BLOCK UNTIL FULL FILE ARRIVES
        # Logic: M*N*2 (data) + K*N*2 (weights) + 4 (header)
        set expected [expr ($m * $k * 2) + ($k * $n * 2) + 4]
        while {[file size $bin_path] < $expected} { after 10 }

        # 3. WRITE DIMENSIONS
        master_write_32 $master_path 0x3004 $m
        master_write_32 $master_path 0x3008 $n
        master_write_32 $master_path 0x300C $k
        
        # 4. DATA STREAM
        master_write_from_file $master_path $bin_path 0x1000 
        
        # 5. TRIGGER & POLL
        master_write_32 $master_path 0x3000 1
        set status 1
        while {$status != 0} {
            set status [master_read_32 $master_path 0x3000 1]
            after 10
        }

        # 6. READ RESULT
        master_read_to_file $master_path $res_path 0x0000 [expr $m * $n * 2]
        
        file delete $bin_path
        puts "Tile (M=$m, N=$n) Processed."
    }
    after 200
}