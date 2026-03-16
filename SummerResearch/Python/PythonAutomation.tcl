set NPU_CTRL   0x30000
set DATA_MEM   0x31000
set WEIGHT_MEM 0x32000
set OUT_MEM    0x33000
set CFG_REG    0x30010
set DONE_REG   0x30014
set CC_REG     0x3001C

set bin_path "C:/Users/pchh520/Documents/GitHub/part-4-project/SummerResearch/Python/tile.bin"
set res_path "C:/Users/pchh520/Documents/GitHub/part-4-project/SummerResearch/Python/result.bin"

set master_path [lindex [get_service_paths master] 0]
if {[is_service_open master $master_path] == 0} {
    open_service master $master_path
}

puts "SYSTEM ALIGNED: Monitoring for tile.bin..."

while {1} {
    if {[file exists $bin_path]} {
        set fp [open $bin_path r]; fconfigure $fp -translation binary
        set raw_content [read $fp]; close $fp

        # Parse header: M, N, K, config
        binary scan $raw_content cccc m n k config
        set m      [expr {$m      & 0xFF}]
        set n      [expr {$n      & 0xFF}]
        set k      [expr {$k      & 0xFF}]
        set config [expr {$config & 0xFF}]

        puts "Parsed: M=$m N=$n K=$k Config=$config"

        set total_act [expr {$m * $k}]
        set total_wgt [expr {$k * $n}]

        if {$config == 1} {
            set bytes_per_val 1
            set fmt "c*"
        } else {
            set bytes_per_val 2
            set fmt "s*"
        }

        binary scan [string range $raw_content 4 [expr {4 + $total_act*$bytes_per_val - 1}]] $fmt d_vals_raw
        binary scan [string range $raw_content [expr {4 + $total_act*$bytes_per_val}] [expr {4 + ($total_act + $total_wgt)*$bytes_per_val - 1}]] $fmt w_vals_raw

        set mask [expr {$config == 1 ? 0xFF : 0xFFFF}]
        set d_final {}
        foreach v $d_vals_raw { lappend d_final [expr {$v & $mask}] }
        set w_final {}
        foreach v $w_vals_raw { lappend w_final [expr {$v & $mask}] }

        # Reset
        master_write_32 $master_path $NPU_CTRL 0

        # Write config, memories, dimensions
        master_write_32 $master_path $CFG_REG $config
        master_write_32 $master_path $DATA_MEM $d_final
        master_write_32 $master_path $WEIGHT_MEM $w_final
        master_write_32 $master_path [expr {$NPU_CTRL + 0x4}] $m
        master_write_32 $master_path [expr {$NPU_CTRL + 0x8}] $n
        master_write_32 $master_path [expr {$NPU_CTRL + 0xC}] $k

        # Trigger
        master_write_32 $master_path $NPU_CTRL 1
        puts "NPU triggered (M=$m, N=$n, K=$k, Config=$config)"

       
        #Poll n_done_mux until high
        set timeout 500
        set elapsed 0
        while {1} {
            set done [master_read_32 $master_path $DONE_REG 1]
            if {$done & 1} {
                puts "Done detected after ${elapsed}ms"
                break
            }
            if {$elapsed >= $timeout} {
                puts "ERROR: Timeout waiting for done"
                break
            }
            after 10
            incr elapsed 10
        }

        set cycle_count [master_read_32 $master_path $CC_REG 1]

        # Read results
        set total_result [expr {$m * $n}]
        set results [master_read_32 $master_path $OUT_MEM $total_result]

        # Save result.bin — result words followed by cycle count as final word
        set fw [open $res_path w]; fconfigure $fw -translation binary
        puts -nonewline $fw [binary format i* $results]
        puts -nonewline $fw [binary format i $cycle_count]
        close $fw

        file delete $bin_path
        puts "Cycle Complete.\n"
    }

    after 100
}