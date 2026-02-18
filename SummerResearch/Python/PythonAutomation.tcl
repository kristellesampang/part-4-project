set bin_path "C:/Users/pchh520/Documents/GitHub/part-4-project/SummerResearch/Python/tile.bin"
set res_path "C:/Users/pchh520/Documents/GitHub/part-4-project/SummerResearch/Python/result.bin"

set master_path [lindex [get_service_paths master] 0]
if {[is_service_open master $master_path] == 0} { open_service master $master_path }

puts "WATCHER ACTIVE: Fixed Handshake Mode..."

while {1} {
    if {[file exists $bin_path]} {
        after 50 ;# Tiny delay to ensure file is closed by OS
        set fp [open $bin_path r]; fconfigure $fp -translation binary
        set raw_content [read $fp]
        close $fp

        # FIXED: Explicitly scan the first 4 bytes for M, N, K
        binary scan $raw_content cccc m n k padding
        # Convert signed char to unsigned integer for the master_write
        set m [expr {$m & 0xFF}]; set n [expr {$n & 0xFF}]; set k [expr {$k & 0xFF}]

        set d_start 4
        set d_len [expr $m * $k * 2]
        set w_start [expr $d_start + $d_len]
        set w_len [expr $k * $n * 2]

        # Scan 16-bit values and prepare for 32-bit "Sparse" writes
        binary scan [string range $raw_content $d_start [expr $d_start + $d_len - 1]] s* d_values
        binary scan [string range $raw_content $w_start [expr $w_start + $w_len - 1]] s* w_values

        set d_final_list {}; foreach val $d_values { lappend d_final_list [expr $val & 0xFFFF] }
        set w_final_list {}; foreach val $w_values { lappend w_final_list [expr $val & 0xFFFF] }

        # --- FPGA EXECUTION ---
        master_write_32 $master_path 0x2000 $d_final_list
        master_write_32 $master_path 0x1000 $w_final_list

        # CONFIGURE REGISTERS
        master_write_32 $master_path 0x3004 $m
        master_write_32 $master_path 0x3008 $n
        master_write_32 $master_path 0x300C $k
        master_write_32 $master_path 0x3000 1 

        puts "Running NPU: M=$m, N=$n, K=$k"
        after 1500

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

        # READ RESULTS: Each result is 4 bytes (32-bit)
        master_read_to_file $master_path $res_path 0x0000 [expr $m * $n * 4]
        set results [master_read_32 $master_path 0x0000 16]
        set idx 0
        foreach res $results {
            puts "Index $idx: $res"
            incr idx
        }
        
        file delete $bin_path
        puts "Tile Done."
    }
    after 100
}