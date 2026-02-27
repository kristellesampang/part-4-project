# --- Addresses Aligned to master_0.master View (Confirmed by Discovery) ---
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
        # Ensure Python has finished writing the file
        after 200
        set fp [open $bin_path r]; fconfigure $fp -translation binary
        set raw_content [read $fp]; close $fp

        # 1. Parse Header (M, N, K)
        binary scan $raw_content cccc m n k padding
        set m [expr {$m & 0xFF}]; set n [expr {$n & 0xFF}]; set k [expr {$k & 0xFF}]
        
        # 2. Extract 32x32 padded blocks (2048 bytes each) for the Stride-32 logic
        set data_payload [string range $raw_content 4 [expr {4 + 2048 - 1}]]
        binary scan $data_payload s1024 d_vals_raw
        
        set weight_payload [string range $raw_content [expr {4 + 2048}] [expr {4 + 4096 - 1}]]
        binary scan $weight_payload s1024 w_vals_raw

        # 3. Mask for 32-bit Avalon Bus writes
        set d_final {}; foreach v $d_vals_raw { lappend d_final [expr {$v & 0xFFFF}] }
        set w_final {}; foreach v $w_vals_raw { lappend w_final [expr {$v & 0xFFFF}] }

        # 4. RESET: Clear Start register to guarantee a clean rising edge
        master_write_32 $master_path $NPU_CTRL 0
        after 100

        # 5. LOAD: Writing to Master's 0x3xxxx view (mapped to NPU's 0x2xxxx side)
        master_write_32 $master_path $DATA_MEM $d_final
        master_write_32 $master_path $WEIGHT_MEM $w_final
        
        # Load Parameters into Control Registers
        master_write_32 $master_path [expr {$NPU_CTRL + 0x4}] $m
        master_write_32 $master_path [expr {$NPU_CTRL + 0x8}] $n
        master_write_32 $master_path [expr {$NPU_CTRL + 0xC}] $k
        
        # 6. VERIFY: Confirm the Master can see the data it just wrote
        set check_val [master_read_32 $master_path $DATA_MEM 1]
        puts "DEBUG: JTAG verified Data at $DATA_MEM is: $check_val"
        
        # 7. SETTLE: Give the Avalon Interconnect time to commit the burst
        after 500

        # 8. EXECUTE: Pulse Start (Write 1 then 0)
        master_write_32 $master_path $NPU_CTRL 1
        after 50
        master_write_32 $master_path $NPU_CTRL 0
        puts "NPU Pulse Sent to $NPU_CTRL (M=$m, N=$n, K=$k)"

        # 9. WAIT: Hardware execution window (2 seconds)
        after 2000

        # 10. READBACK: Retrieve results from Master's 0x33000 view
        set total_result [expr {$m * $n}]
        set results [master_read_32 $master_path $OUT_MEM $total_result]

        # 11. FORMAT OUTPUT
        puts "--- HARDWARE RESULT ($m x $n) ---"
        set idx 0
        foreach res $results {
            if {$res > 0x7FFFFFFF} { set res [expr {$res - 0x100000000}] }
            puts -nonewline [format "%8d " $res]
            incr idx
            if {[expr $idx % $n] == 0} { puts "" }
        }

        # Save result.bin for Python verification
        set fw [open $res_path w]; fconfigure $fw -translation binary
        puts -nonewline $fw [binary format i* $results]
        close $fw

        file delete $bin_path
        puts "Cycle Complete.\n"
    }
    after 100
}