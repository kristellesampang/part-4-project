# --- fix_qsim_compile.tcl ---

# 1. *** EDIT THIS PATH ***
# Path to the VHDL source files for altera_mf
set SIM_LIB_PATH "/home/pratham/altera_pro/25.3/questa_fse/intel/vhdl/src/altera_mf"

# 2. Create the compilation target directory
# We create a new, empty directory named 'compiled_altera_mf'
file mkdir compiled_altera_mf
vlib compiled_altera_mf 

# 3. Map the newly created, empty directory as the compilation target
vmap altera_mf ./compiled_altera_mf

# 4. Compile the source files into the mapped library
# NOTE: We are compiling the source files *from* the SIM_LIB_PATH 
#       *into* the new 'altera_mf' library.
# --- Edit the compilation block in fix_qsim_compile.tcl ---

# 4. Compile the source files into the mapped library

# NEW STEP 1: Compile the foundation (LPM) components first!
puts "Compiling foundation dependency: lpm_components.vhd..."
vcom -work altera_mf /home/pratham/altera_pro/25.3/questa_fse/intel/vhdl/src/lpm_components.vhd

# NEW STEP: Compile the dependency file first!
puts "Compiling dependency: altera_mf_components.vhd..."
vcom -work altera_mf /home/pratham/altera_pro/25.3/questa_fse/intel/vhdl/src/altera_mf/altera_mf_components.vhd 

# Original Step (Now the second step): Compile the main file
puts "Compiling main library: altera_mf.vhd..."
vcom -work altera_mf /home/pratham/altera_pro/25.3/questa_fse/intel/vhdl/src/altera_mf/altera_mf.vhd

# (Optional) Compile 220model.vhd next
puts "Compiling standard components: 220model.vhd..."
vcom -work altera_mf /home/pratham/altera_pro/25.3/questa_fse/intel/vhdl/src/220model/220model.vhd
puts "Altera library compilation complete. Ready for project files."

# --- END OF SCRIPT ---