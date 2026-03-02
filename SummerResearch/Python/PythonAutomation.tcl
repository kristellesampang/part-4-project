set NPU_CTRL   0x30000
set DATA_MEM   0x31000
set WEIGHT_MEM 0x32000
set OUT_MEM    0x33000

set bin_path "C:/Users/pchh520/Documents/GitHub/part-4-project/SummerResearch/Python/tile.bin"
set res_path "C:/Users/pchh520/Documents/GitHub/part-4-project/SummerResearch/Python/result.bin"

set master_path [lindex [get_service_paths master] 0]
if {[is_service_open master $master_path] == 0} { open_service master $master_path }

puts "SYSTEM ALIGNED: Monitoring for tile.bin..."

while {1} {
    if {[file exists $bin_path]} {
        after 200
        set fp [open $bin_path r]; fconfigure $fp -translation binary
        set raw_content [read $fp]; close $fp

        binary scan $raw_content cccc m n k padding
        set m [expr {$m & 0xFF}]; set n [expr {$n & 0xFF}]; set k [expr {$k & 0xFF}]

        set total_act [expr {$m * $k}]
        set total_wgt [expr {$k * $n}]

        binary scan [string range $raw_content 4 [expr {4 + $total_act*2 - 1}]] s* d_vals_raw
        binary scan [string range $raw_content [expr {4 + $total_act*2}] [expr {4 + ($total_act + $total_wgt)*2 - 1}]] s* w_vals_raw

        set d_final {}; foreach v $d_vals_raw { lappend d_final [expr {$v & 0xFFFF}] }
        set w_final {}; foreach v $w_vals_raw { lappend w_final [expr {$v & 0xFFFF}] }

        master_write_32 $master_path $NPU_CTRL 0
        after 100

        master_write_32 $master_path $DATA_MEM $d_final
        master_write_32 $master_path $WEIGHT_MEM $w_final

        master_write_32 $master_path [expr {$NPU_CTRL + 0x4}] $m
        master_write_32 $master_path [expr {$NPU_CTRL + 0x8}] $n
        master_write_32 $master_path [expr {$NPU_CTRL + 0xC}] $k

        set check_val [master_read_32 $master_path $DATA_MEM 1]
        puts "DEBUG: JTAG verified Data at $DATA_MEM is: $check_val"

        after 500

        master_write_32 $master_path $NPU_CTRL 1
        after 50
        master_write_32 $master_path $NPU_CTRL 0
        puts "NPU Pulse Sent to $NPU_CTRL (M=$m, N=$n, K=$k)"

        after 2000

        set total_result [expr {$m * $n}]
        set results [master_read_32 $master_path $OUT_MEM $total_result]

        puts "--- HARDWARE RESULT ($m x $n) ---"
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
        puts "Cycle Complete.\n"
    }
    after 100
}