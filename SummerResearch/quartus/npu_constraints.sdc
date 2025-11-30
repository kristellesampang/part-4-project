# Constrain the main clock to 100 MHz
# Period is 10.0 ns (1 / 100 MHz)
create_clock -name {clk} -period 10.0 -waveform {0.0 5.0} [get_ports {clk}]