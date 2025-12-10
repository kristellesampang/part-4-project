# --- FIX_MODELSIM.TCL ---

# 1. *** YOU MUST EDIT THIS PATH ***
# Replace the path below with the root directory of your Intel Quartus installation.
# Since you have Quartus 25.3, it should be something like:
# /opt/intel/intelFPGA_lite/25.3/quartus/
# or wherever your system placed the 25.3 root folder.
set QUARTUS_PATH "/opt/intelFPGA/20.1/modelsim_ase/altera/vhdl/src/altera_mf/altera_mf.vhd"

# 2. Create the target library for Altera primitives
vlib altera_mf
vmap altera_mf ./altera_mf

# 3. Compile the core Altera VHDL primitives into 'altera_mf'
# This resolves the 'ram_2port_2041' error.
puts "Compiling Altera VHDL Primitives..."
vcom -work altera_mf $QUARTUS_PATH/eda/sim_lib/altera_mf.vhd
vcom -work altera_mf $QUARTUS_PATH/eda/sim_lib/220model.vhd
# Note: altera_primitives.vhd may also be required depending on vers
# vcom -work altera_mf $QUARTUS_PATH/eda/sim_lib/altera_primitives.vhd

puts "Compilation of Altera Primitives Complete."

# End of script