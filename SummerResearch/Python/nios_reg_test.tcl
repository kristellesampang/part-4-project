set master_path [lindex [get_service_paths master] 0]
if {[is_service_open master $master_path] == 0} {
    open_service master $master_path
}

# Read NPU control registers
set m [master_read_32 $master_path 0x30004 1]
set n [master_read_32 $master_path 0x30008 1]
set k [master_read_32 $master_path 0x3000C 1]
set ready [master_read_32 $master_path 0x30000 1]

puts "M=$m N=$n K=$k READY=$ready"

# Read first 16 words of data_mem
set data [master_read_32 $master_path 0x31000 16]
puts "DATA_MEM: $data"

# Read first 16 words of weight_mem
set weights [master_read_32 $master_path 0x32000 16]
puts "WEIGHT_MEM: $weights"

# Read first 16 words of out_mem
set results [master_read_32 $master_path 0x33000 16]
puts "OUT_MEM: $results"