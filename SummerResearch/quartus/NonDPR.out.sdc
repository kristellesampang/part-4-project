# 1. Define the physical 50MHz clock entering the FPGA
# 20ns period = 50MHz. [get_ports {CLK_50}] must match your VHDL port name.
create_clock -name {CLK_50} -period 20.000 [get_ports {CLK_50}]

# 2. Derive internal PLL clocks 
# Even if you aren't using a PLL yet, this prevents Quartus from guessing.
derive_pll_clocks

# 3. Derive Clock Uncertainty
# This is mandatory for Arria 10 to account for jitter and prevent bus hangs.
derive_clock_uncertainty

# 4. Handle the JTAG-to-Avalon Bridge
# This tells the timing analyzer to ignore the paths between the 
# JTAG clock (USB) and your System clock (CLK_50). 
# This prevents the "Channel Closed" error caused by timing violations.
set_clock_groups -asynchronous -group [get_clocks {CLK_50}] -group [get_clocks {altera_reserved_tck}]