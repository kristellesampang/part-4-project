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
        after 200
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

        set d_final {}
        foreach v $d_vals_raw { lappend d_final [expr {$v & 0xFFFF}] }
        set w_final {}
        foreach v $w_vals_raw { lappend w_final [expr {$v & 0xFFFF}] }

        # Reset
        master_write_32 $master_path $NPU_CTRL 0
        after 100

        # Write config, memories, dimensions
        master_write_32 $master_path $CFG_REG $config
        master_write_32 $master_path $DATA_MEM $d_final
        master_write_32 $master_path $WEIGHT_MEM $w_final
        master_write_32 $master_path [expr {$NPU_CTRL + 0x4}] $m
        master_write_32 $master_path [expr {$NPU_CTRL + 0x8}] $n
        master_write_32 $master_path [expr {$NPU_CTRL + 0xC}] $k

        set check_val [master_read_32 $master_path $DATA_MEM 1]
        puts "DEBUG: JTAG verified Data at $DATA_MEM is: $check_val"

       # set d0 [master_read_32 $master_path $DATA_MEM 20]
        #set w0 [master_read_32 $master_path $WEIGHT_MEM 4]
        #puts "DEBUG: First 20 Data values: $d0"
        #puts "DEBUG: First 4 Weight values: $w0"

        # Trigger
        master_write_32 $master_path $NPU_CTRL 1
        puts "NPU triggered (M=$m, N=$n, K=$k, Config=$config)"


        # Poll n_done_mux until high
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

        set cc [master_read_32 $master_path $CC_REG 1]
        puts "Cycle Count: $cc"

        # Read results
        set total_result [expr {$m * $n}]
        set results [master_read_32 $master_path $OUT_MEM $total_result]

        # puts "--- HARDWARE RESULT ($m x $n) ---"
        # set idx 0
        # foreach res $results {
        #     if {$res > 0x7FFFFFFF} { set res [expr {$res - 0x100000000}] }
        #     puts -nonewline [format "%8d " $res]
        #     incr idx
        #     if {[expr $idx % $n] == 0} { puts "" }
        # }
         set cc [master_read_32 $master_path $CC_REG 1]
        puts "Cycle Count: $cc"
        
        # Save result.bin
        set fw [open $res_path w]; fconfigure $fw -translation binary
        puts -nonewline $fw [binary format i* $results]
        close $fw

        file delete $bin_path
        puts "Cycle Complete.\n"
    }
    after 100
}