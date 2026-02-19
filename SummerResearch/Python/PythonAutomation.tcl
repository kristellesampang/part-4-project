set bin_path "C:/Users/pchh520/Documents/GitHub/part-4-project/SummerResearch/Python/tile.bin"
set res_path "C:/Users/pchh520/Documents/GitHub/part-4-project/SummerResearch/Python/result.bin"

set master_path [lindex [get_service_paths master] 0]
if {[is_service_open master $master_path] == 0} { open_service master $master_path }

puts "WATCHER ACTIVE: Full Active Tile Mode..."

while {1} {
    if {[file exists $bin_path]} {
        after 100
        set fp [open $bin_path r]; fconfigure $fp -translation binary
        set raw_content [read $fp]
        close $fp

        binary scan $raw_content cccc m n k padding
        set m [expr {$m & 0xFF}]; set n [expr {$n & 0xFF}]; set k [expr {$k & 0xFF}]

        # GRID SYNC: 1024 words per matrix
        set grid_size 1024
        set byte_len [expr $grid_size * 2]

        binary scan [string range $raw_content 4 [expr 4 + $byte_len - 1]] s* d_vals
        binary scan [string range $raw_content [expr 4 + $byte_len] [expr 4 + (2 * $byte_len) - 1]] s* w_vals

        set d_final {}; foreach v $d_vals { lappend d_final [expr $v & 0xFFFF] }
        set w_final {}; foreach v $w_vals { lappend w_final [expr $v & 0xFFFF] }

        master_write_32 $master_path 0x2000 $d_final
        master_write_32 $master_path 0x1000 $w_final

        master_write_32 $master_path 0x3004 $m
        master_write_32 $master_path 0x3008 $n
        master_write_32 $master_path 0x300C $k
        master_write_32 $master_path 0x3000 1 

        puts "NPU Running: M=$m, N=$n, K=$k"
        after 1500

        # READ FULL 32x32 RESULT (4096 BYTES) FOR PYTHON
        master_read_to_file $master_path $res_path 0x0000 4096
        
        # --- DYNAMIC TCL PRINTING: Printing the FULL Active Tile ---
        set total_active [expr $m * $n]
        set results [master_read_32 $master_path 0x0000 $total_active]
        
        puts "--- FULL ACTIVE HARDWARE TILE ($m x $n) ---"
        set idx 0
        foreach res $results {
            # Print in rows for readability
            puts -nonewline [format "%8d " $res]
            incr idx
            if {[expr $idx % $n] == 0} { puts "" }
        }
        
        file delete $bin_path
        puts "Tile Complete.\n"
    }
    after 100
}