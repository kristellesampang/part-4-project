# --- Full Corrected TCL: Dense Tile Handshake with Debug Printing ---
set bin_path "C:/Users/pchh520/Documents/GitHub/part-4-project/SummerResearch/Python/tile.bin"
set res_path "C:/Users/pchh520/Documents/GitHub/part-4-project/SummerResearch/Python/result.bin"

set master_path [lindex [get_service_paths master] 0]
if {[is_service_open master $master_path] == 0} { open_service master $master_path }

puts "WATCHER ACTIVE: Waiting for Python tile.bin..."

while {1} {
    if {[file exists $bin_path]} {
        after 100
        set fp [open $bin_path r]; fconfigure $fp -translation binary
        set raw_content [read $fp]
        close $fp

        binary scan $raw_content cccc m n k padding
        set m [expr {$m & 0xFF}]; set n [expr {$n & 0xFF}]; set k [expr {$k & 0xFF}]

        # Extract Dense Input
        set total_active [expr {$m * $k}]
        binary scan [string range $raw_content 4 [expr 4 + $total_active*2 - 1]] s* d_vals
        set total_weight [expr {$k * $n}]
        binary scan [string range $raw_content [expr 4 + $total_active*2] [expr 4 + ($total_active + $total_weight)*2 - 1]] s* w_vals

        # Write to NPU
        set d_final {}; foreach v $d_vals { lappend d_final [expr {$v & 0xFFFF}] }
        set w_final {}; foreach v $w_vals { lappend w_final [expr {$v & 0xFFFF}] }
        master_write_32 $master_path 0x2000 $d_final
        master_write_32 $master_path 0x1000 $w_final

        master_write_32 $master_path 0x3004 $m
        master_write_32 $master_path 0x3008 $n
        master_write_32 $master_path 0x300C $k
        master_write_32 $master_path 0x3000 1 

        puts "NPU Running: M=$m, N=$n, K=$k"
        after 1500

        # Read back results
        set total_result [expr {$m * $n}]
        set results [master_read_32 $master_path 0x0000 $total_result]

        # PRINT TO CONSOLE FOR VERIFICATION
        puts "--- HARDWARE DENSE TILE ($m x $n) ---"
        set idx 0
        foreach res $results {
            # Handle 32-bit signedness for display
            if {$res > 0x7FFFFFFF} { set res [expr {$res - 0x100000000}] }
            puts -nonewline [format "%8d " $res]
            incr idx
            if {[expr $idx % $n] == 0} { puts "" }
        }

        # Write result.bin for Python
        set fw [open $res_path w]; fconfigure $fw -translation binary
        puts -nonewline $fw [binary format i* $results]
        close $fw
        
        file delete $bin_path
        puts "Tile Complete. Result written to disk.\n"
    }
    after 100
}