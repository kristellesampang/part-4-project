# PythonAutomation.tcl
set m [lindex $argv 0]
set n [lindex $argv 1]
set k [lindex $argv 2]

# Read data from the file Python just made
# Use the full path so it doesn't matter where Java is launched from
set fp [open "C:/Users/pchh520/Documents/GitHub/part-4-project/SummerResearch/Python/tile_data.txt" r]
set data_list [gets $fp]
set weight_list [gets $fp]
close $fp

refresh_connections
set master_path [lindex [get_service_paths master] 0]
open_service master $master_path

# Write to FPGA
master_write_32 $master_path 0x2000 $data_list
master_write_32 $master_path 0x1000 $weight_list
master_write_32 $master_path 0x3004 $m
master_write_32 $master_path 0x3008 $n
master_write_32 $master_path 0x300C $k
master_write_32 $master_path 0x3000 1

after 200
set results [master_read_32 $master_path 0x0000 [expr $m * $n]]
puts "FPGA_DATA: $results"

close_service master $master_path
exit