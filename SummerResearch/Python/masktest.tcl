set master_path [lindex [get_service_paths master] 0]
open_service master $master_path

set data16   {1 2 3 4 5 6 7 8 1 1 1 1 2 2 2 2}
set weight16 {1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1}

#============ TEST 1: INT16 FIRST RUN ============
puts "--- INT16 TEST 1 ---"
master_write_32 $master_path 0x30000 0
after 100
master_write_32 $master_path 0x31000 $data16
master_write_32 $master_path 0x32000 $weight16
master_write_32 $master_path 0x30010 0
master_write_32 $master_path 0x30004 4
master_write_32 $master_path 0x30008 4
master_write_32 $master_path 0x3000C 4
after 100
master_write_32 $master_path 0x30000 1
after 2000
set results [master_read_32 $master_path 0x33000 16]
set idx 0
foreach res $results {
    if {$res > 0x7FFFFFFF} { set res [expr {$res - 0x100000000}] }
    puts -nonewline [format "%6d " $res]
    incr idx
    if {[expr $idx % 4] == 0} { puts "" }
}

#============ TEST 2: INT16 SECOND RUN (same data) ============
puts "\n--- INT16 TEST 2 (expect same as test 1) ---"
master_write_32 $master_path 0x30000 0
after 100
master_write_32 $master_path 0x31000 $data16
master_write_32 $master_path 0x32000 $weight16
master_write_32 $master_path 0x30010 0
master_write_32 $master_path 0x30004 4
master_write_32 $master_path 0x30008 4
master_write_32 $master_path 0x3000C 4
after 100
master_write_32 $master_path 0x30000 1
after 2000
set results [master_read_32 $master_path 0x33000 16]
set idx 0
foreach res $results {
    if {$res > 0x7FFFFFFF} { set res [expr {$res - 0x100000000}] }
    puts -nonewline [format "%6d " $res]
    incr idx
    if {[expr $idx % 4] == 0} { puts "" }
}