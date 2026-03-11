## Generated SDC file "NonDPR.out.sdc"

## Copyright (C) 2024  Intel Corporation. All rights reserved.
## Your use of Intel Corporation's design tools, logic functions 
## and other software and tools, and any partner logic 
## functions, and any output files from any of the foregoing 
## (including device programming or simulation files), and any 
## associated documentation or information are expressly subject 
## to the terms and conditions of the Intel Program License 
## Subscription Agreement, the Intel Quartus Prime License Agreement,
## the Intel FPGA IP License Agreement, or other applicable license
## agreement, including, without limitation, that your use is for
## the sole purpose of programming logic devices manufactured by
## Intel and sold by Intel or its authorized distributors.  Please
## refer to the Intel FPGA Software License Subscription Agreements 
## on the Quartus Prime software download page.


## VENDOR  "Intel Corporation"
## PROGRAM "Quartus Prime"
## VERSION "Version 24.1.0 Build 115 03/21/2024 SC Pro Edition"

## DATE    "Wed Mar 11 14:12:15 2026"

##
## DEVICE  "10AS066N3F40E2SG"
##


#**************************************************************
# Time Information
#**************************************************************

set_time_format -unit ns -decimal_places 3



#**************************************************************
# Create Clock
#**************************************************************

create_clock -name {CLK_50} -period 20.000 -waveform { 0.000 10.000 } [get_ports {CLK_50}]


#**************************************************************
# Create Generated Clock
#**************************************************************



#**************************************************************
# Set Clock Latency
#**************************************************************



#**************************************************************
# Set Clock Uncertainty
#**************************************************************

set_clock_uncertainty -rise_from [get_clocks {CLK_50}] -rise_to [get_clocks {CLK_50}]  0.030  
set_clock_uncertainty -rise_from [get_clocks {CLK_50}] -fall_to [get_clocks {CLK_50}]  0.030  
set_clock_uncertainty -fall_from [get_clocks {CLK_50}] -rise_to [get_clocks {CLK_50}]  0.030  
set_clock_uncertainty -fall_from [get_clocks {CLK_50}] -fall_to [get_clocks {CLK_50}]  0.030  


#**************************************************************
# Set Input Delay
#**************************************************************



#**************************************************************
# Set Output Delay
#**************************************************************



#**************************************************************
# Set Clock Groups
#**************************************************************

set_clock_groups -asynchronous -group [get_clocks {CLK_50}] -group [get_clocks {altera_reserved_tck}] 


#**************************************************************
# Set False Path
#**************************************************************

set_false_path -from [get_registers {*|alt_jtag_atlantic:*|jupdate}] -to [get_registers {*|alt_jtag_atlantic:*|jupdate1*}]
set_false_path -from [get_registers {*|alt_jtag_atlantic:*|rdata[*]}] -to [get_registers {*|alt_jtag_atlantic*|td_shift[*]}]
set_false_path -from [get_registers {*|alt_jtag_atlantic:*|read}] -to [get_registers {*|alt_jtag_atlantic:*|read1*}]
set_false_path -from [get_registers {*|alt_jtag_atlantic:*|read_req}] 
set_false_path -from [get_registers {*|alt_jtag_atlantic:*|rvalid}] -to [get_registers {*|alt_jtag_atlantic*|td_shift[*]}]
set_false_path -from [get_registers {*|t_dav}] -to [get_registers {*|alt_jtag_atlantic:*|tck_t_dav}]
set_false_path -from [get_registers {*|alt_jtag_atlantic:*|user_saw_rvalid}] -to [get_registers {*|alt_jtag_atlantic:*|rvalid0*}]
set_false_path -from [get_registers {*|alt_jtag_atlantic:*|wdata[*]}] -to [get_registers *]
set_false_path -from [get_registers {*|alt_jtag_atlantic:*|write}] -to [get_registers {*|alt_jtag_atlantic:*|write1*}]
set_false_path -from [get_registers {*|alt_jtag_atlantic:*|write_stalled}] -to [get_registers {*|alt_jtag_atlantic:*|t_ena*}]
set_false_path -from [get_registers {*|alt_jtag_atlantic:*|write_stalled}] -to [get_registers {*|alt_jtag_atlantic:*|t_pause*}]
set_false_path -from [get_registers {*|alt_jtag_atlantic:*|write_valid}] 
set_false_path -to [get_keepers {*altera_std_synchronizer:*|din_s1}]
set_false_path -from [get_registers {*altera_jtag_src_crosser:*|sink_data_buffer*}] -to [get_registers {*altera_jtag_src_crosser:*|src_data*}]
set_false_path -to [get_pins -nocase -compatibility_mode {System|master_0|master_0|rst_controller|*alt_rst_sync_uq1|altera_reset_synchronizer_int_chain*|clrn}]
set_false_path -to [get_pins -nocase -compatibility_mode {System|rst_controller|*alt_rst_sync_uq1|altera_reset_synchronizer_int_chain*|clrn}]
set_false_path -to [get_pins -nocase -compatibility_mode {*|alt_rst_req_sync_in_rst|altera_reset_synchronizer_int_chain*|clrn}]
set_false_path -to [get_pins -nocase -compatibility_mode {*|alt_rst_req_sync_out_rst|altera_reset_synchronizer_int_chain*|clrn}]


#**************************************************************
# Set Multicycle Path
#**************************************************************

set_multicycle_path -setup -end -from [get_registers {*w_row_idx_reg*}] -to [get_registers {*out_data_reg*}] 2
set_multicycle_path -hold -end -from [get_registers {*w_row_idx_reg*}] -to [get_registers {*out_data_reg*}] 1


#**************************************************************
# Set Maximum Delay
#**************************************************************

set_max_delay -from [get_registers {*altera_avalon_st_clock_crosser:*|in_data_buffer*}] -to [get_registers {*altera_avalon_st_clock_crosser:*|out_data_buffer*}] 100.000 
set_max_delay -from [get_registers {*altera_avalon_st_clock_crosser:*}] -to [get_registers {*altera_avalon_st_clock_crosser:*|altera_std_synchronizer_nocut:*|din_s1}] 100.000 


#**************************************************************
# Set Minimum Delay
#**************************************************************

set_min_delay -from [get_registers {*altera_avalon_st_clock_crosser:*|in_data_buffer*}] -to [get_registers {*altera_avalon_st_clock_crosser:*|out_data_buffer*}] -100.000
set_min_delay -from [get_registers {*altera_avalon_st_clock_crosser:*}] -to [get_registers {*altera_avalon_st_clock_crosser:*|altera_std_synchronizer_nocut:*|din_s1}] -100.000


#**************************************************************
# Set Input Transition
#**************************************************************



#**************************************************************
# Set Net Delay
#**************************************************************

set_net_delay -max -value_multiplier 0.800 -get_value_from_clock_period dst_clock_period -from [get_registers {System|master_0|master_0|jtag_phy_embedded_in_jtag_master|normal.jtag_dc_streaming|sink_crosser|in_data_buffer*}] -to [get_registers {System|master_0|master_0|jtag_phy_embedded_in_jtag_master|normal.jtag_dc_streaming|sink_crosser|out_data_buffer*}]
set_net_delay -max -value_multiplier 0.800 -get_value_from_clock_period dst_clock_period -from [get_registers {System|intel_niosv_g_0|intel_niosv_g_0|dbg_mod|dtm_inst|rsp_clk_xer|clock_xer|in_data_buffer*}] -to [get_registers {System|intel_niosv_g_0|intel_niosv_g_0|dbg_mod|dtm_inst|rsp_clk_xer|clock_xer|out_data_buffer*}]
set_net_delay -max -value_multiplier 0.800 -get_value_from_clock_period dst_clock_period -from [get_registers {System|intel_niosv_g_0|intel_niosv_g_0|dbg_mod|dtm_inst|cmd_clk_xer|clock_xer|in_data_buffer*}] -to [get_registers {System|intel_niosv_g_0|intel_niosv_g_0|dbg_mod|dtm_inst|cmd_clk_xer|clock_xer|out_data_buffer*}]
set_net_delay -max -value_multiplier 0.800 -get_value_from_clock_period dst_clock_period -from [get_registers {System|master_0|master_0|jtag_phy_embedded_in_jtag_master|normal.jtag_dc_streaming|sink_crosser|in_data_toggle}] -to [get_registers {System|master_0|master_0|jtag_phy_embedded_in_jtag_master|normal.jtag_dc_streaming|sink_crosser|in_to_out_synchronizer|din_s1}]
set_net_delay -max -value_multiplier 0.800 -get_value_from_clock_period dst_clock_period -from [get_registers {System|intel_niosv_g_0|intel_niosv_g_0|dbg_mod|dtm_inst|rsp_clk_xer|clock_xer|in_data_toggle}] -to [get_registers {System|intel_niosv_g_0|intel_niosv_g_0|dbg_mod|dtm_inst|rsp_clk_xer|clock_xer|in_to_out_synchronizer|din_s1}]
set_net_delay -max -value_multiplier 0.800 -get_value_from_clock_period dst_clock_period -from [get_registers {System|intel_niosv_g_0|intel_niosv_g_0|dbg_mod|dtm_inst|cmd_clk_xer|clock_xer|in_data_toggle}] -to [get_registers {System|intel_niosv_g_0|intel_niosv_g_0|dbg_mod|dtm_inst|cmd_clk_xer|clock_xer|in_to_out_synchronizer|din_s1}]
set_net_delay -max -value_multiplier 0.800 -get_value_from_clock_period dst_clock_period -from [get_registers {System|intel_niosv_g_0|intel_niosv_g_0|dbg_mod|dtm_inst|rsp_clk_xer|clock_xer|out_data_toggle_flopped_n}] -to [get_registers {System|intel_niosv_g_0|intel_niosv_g_0|dbg_mod|dtm_inst|rsp_clk_xer|clock_xer|out_to_in_synchronizer|din_s1}]
set_net_delay -max -value_multiplier 0.800 -get_value_from_clock_period dst_clock_period -from [get_registers {System|master_0|master_0|jtag_phy_embedded_in_jtag_master|normal.jtag_dc_streaming|sink_crosser|out_data_toggle_flopped_n}] -to [get_registers {System|master_0|master_0|jtag_phy_embedded_in_jtag_master|normal.jtag_dc_streaming|sink_crosser|out_to_in_synchronizer|din_s1}]
set_net_delay -max -value_multiplier 0.800 -get_value_from_clock_period dst_clock_period -from [get_registers {System|intel_niosv_g_0|intel_niosv_g_0|dbg_mod|dtm_inst|cmd_clk_xer|clock_xer|out_data_toggle_flopped_n}] -to [get_registers {System|intel_niosv_g_0|intel_niosv_g_0|dbg_mod|dtm_inst|cmd_clk_xer|clock_xer|out_to_in_synchronizer|din_s1}]


#**************************************************************
# Set Max Skew
#**************************************************************

set_max_skew -from [get_registers {System|master_0|master_0|jtag_phy_embedded_in_jtag_master|normal.jtag_dc_streaming|sink_crosser|in_data_buffer*}] -to [get_registers {System|master_0|master_0|jtag_phy_embedded_in_jtag_master|normal.jtag_dc_streaming|sink_crosser|out_data_buffer*}] -get_skew_value_from_clock_period src_clock_period -skew_value_multiplier 0.800 -nowarn
set_max_skew -from [get_registers {System|intel_niosv_g_0|intel_niosv_g_0|dbg_mod|dtm_inst|rsp_clk_xer|clock_xer|in_data_buffer*}] -to [get_registers {System|intel_niosv_g_0|intel_niosv_g_0|dbg_mod|dtm_inst|rsp_clk_xer|clock_xer|out_data_buffer*}] -get_skew_value_from_clock_period src_clock_period -skew_value_multiplier 0.800 -nowarn
set_max_skew -from [get_registers {System|intel_niosv_g_0|intel_niosv_g_0|dbg_mod|dtm_inst|cmd_clk_xer|clock_xer|in_data_buffer*}] -to [get_registers {System|intel_niosv_g_0|intel_niosv_g_0|dbg_mod|dtm_inst|cmd_clk_xer|clock_xer|out_data_buffer*}] -get_skew_value_from_clock_period src_clock_period -skew_value_multiplier 0.800 -nowarn
set_max_skew -from [get_registers {System|master_0|master_0|jtag_phy_embedded_in_jtag_master|normal.jtag_dc_streaming|sink_crosser|in_data_toggle}] -to [get_registers {System|master_0|master_0|jtag_phy_embedded_in_jtag_master|normal.jtag_dc_streaming|sink_crosser|in_to_out_synchronizer|din_s1}] -get_skew_value_from_clock_period src_clock_period -skew_value_multiplier 0.800 -nowarn
set_max_skew -from [get_registers {System|intel_niosv_g_0|intel_niosv_g_0|dbg_mod|dtm_inst|rsp_clk_xer|clock_xer|in_data_toggle}] -to [get_registers {System|intel_niosv_g_0|intel_niosv_g_0|dbg_mod|dtm_inst|rsp_clk_xer|clock_xer|in_to_out_synchronizer|din_s1}] -get_skew_value_from_clock_period src_clock_period -skew_value_multiplier 0.800 -nowarn
set_max_skew -from [get_registers {System|intel_niosv_g_0|intel_niosv_g_0|dbg_mod|dtm_inst|cmd_clk_xer|clock_xer|in_data_toggle}] -to [get_registers {System|intel_niosv_g_0|intel_niosv_g_0|dbg_mod|dtm_inst|cmd_clk_xer|clock_xer|in_to_out_synchronizer|din_s1}] -get_skew_value_from_clock_period src_clock_period -skew_value_multiplier 0.800 -nowarn
set_max_skew -from [get_registers {System|intel_niosv_g_0|intel_niosv_g_0|dbg_mod|dtm_inst|rsp_clk_xer|clock_xer|out_data_toggle_flopped_n}] -to [get_registers {System|intel_niosv_g_0|intel_niosv_g_0|dbg_mod|dtm_inst|rsp_clk_xer|clock_xer|out_to_in_synchronizer|din_s1}] -get_skew_value_from_clock_period src_clock_period -skew_value_multiplier 0.800 -nowarn
set_max_skew -from [get_registers {System|master_0|master_0|jtag_phy_embedded_in_jtag_master|normal.jtag_dc_streaming|sink_crosser|out_data_toggle_flopped_n}] -to [get_registers {System|master_0|master_0|jtag_phy_embedded_in_jtag_master|normal.jtag_dc_streaming|sink_crosser|out_to_in_synchronizer|din_s1}] -get_skew_value_from_clock_period src_clock_period -skew_value_multiplier 0.800 -nowarn
set_max_skew -from [get_registers {System|intel_niosv_g_0|intel_niosv_g_0|dbg_mod|dtm_inst|cmd_clk_xer|clock_xer|out_data_toggle_flopped_n}] -to [get_registers {System|intel_niosv_g_0|intel_niosv_g_0|dbg_mod|dtm_inst|cmd_clk_xer|clock_xer|out_to_in_synchronizer|din_s1}] -get_skew_value_from_clock_period src_clock_period -skew_value_multiplier 0.800 -nowarn
