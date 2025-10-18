# Constrain the main clock to 100 MHz
# Period is 10.0 ns (1 / 100 MHz)
create_clock -name {CLOCK_50} -period 10.0 -waveform {0.0 5.0} [get_ports {CLOCK_50}]
# Period is 13.333 ns (1 / 75 MHz)
# Constrain the main clock to 200 MHz
# Period is 5.0 ns (1 / 200 MHz)
#create_clock -name {clk} -period 5.0 -waveform {0.0 2.5} [get_ports {clk}]
# Constrain the main clock to 50 MHz
# Period is 20.0 ns (1 / 50 MHz)
#create_clock -name {CLOCK_50} -period 20.0 -waveform {0.0 10.0} [get_ports {CLOCK_50}]