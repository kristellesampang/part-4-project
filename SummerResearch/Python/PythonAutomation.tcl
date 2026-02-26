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

        set total_active [expr {$m * $k}]
        binary scan [string range $raw_content 4 [expr 4 + $total_active*2 - 1]] s* d_vals
        set total_weight [expr {$k * $n}]
        binary scan [string range $raw_content [expr 4 + $total_active*2] [expr 4 + ($total_active + $total_weight)*2 - 1]] s* w_vals

        set d_final {}; foreach v $d_vals { lappend d_final $v }
        set w_final {}; foreach v $w_vals { lappend w_final $v }

        master_write_32 $master_path 0x31000 $d_final
        master_write_32 $master_path 0x32000 $w_final
        after 500
        
        # Verify writes
        set verify_data [master_read_32 $master_path 0x31000 16]
        set verify_weight [master_read_32 $master_path 0x32000 16]
        puts "DATA_MEM first 16: $verify_data"
        puts "WEIGHT_MEM first 16: $verify_weight"
        set verify_data_end [master_read_32 $master_path [expr {0x31000 + ($m*$k - 16)*4}] 16]
        set verify_weight_end [master_read_32 $master_path [expr {0x32000 + ($k*$n - 16)*4}] 16]
        puts "DATA_MEM last 16: $verify_data_end"
        puts "WEIGHT_MEM last 16: $verify_weight_end"

        master_write_32 $master_path 0x30004 $m
        master_write_32 $master_path 0x30008 $n
        master_write_32 $master_path 0x3000C $k
        master_write_32 $master_path 0x30000 1

        puts "NPU Running: M=$m, N=$n, K=$k"
        after 1500

        set total_result [expr {$m * $n}]
        set results [master_read_32 $master_path 0x33000 $total_result]

        puts "--- HARDWARE DENSE TILE ($m x $n) ---"
        set idx 0
        foreach res $results {
            if {$res > 0x7FFFFFFF} { set res [expr {$res - 0x100000000}] }
            puts -nonewline [format "%8d " $res]
            incr idx
            if {[expr $idx % $n] == 0} { puts "" }
        }

        set fw [open $res_path w]; fconfigure $fw -translation binary
        puts -nonewline $fw [binary format i* $results]
        close $fw

        file delete $bin_path
        puts "Tile Complete. Result written to disk.\n"
    }
    after 100
}