source [file join [file dirname [info script]] ./../../../ip/PR_Test_Top/PR_Test_Top_jtag_uart_0/sim/common/modelsim_files.tcl]
source [file join [file dirname [info script]] ./../../../ip/PR_Test_Top/PR_Test_Top_intel_niosv_g_0/sim/common/modelsim_files.tcl]
source [file join [file dirname [info script]] ./../../../ip/PR_Test_Top/PR_Test_Top_intel_onchip_memory_0/sim/common/modelsim_files.tcl]
source [file join [file dirname [info script]] ./../../../ip/PR_Test_Top/PR_Test_Top_reset_in/sim/common/modelsim_files.tcl]
source [file join [file dirname [info script]] ./../../../ip/PR_Test_Top/PR_Test_Top_clock_in/sim/common/modelsim_files.tcl]
source [file join [file dirname [info script]] ./../../../ip/PR_Test_Top/SDRAM/sim/common/modelsim_files.tcl]

namespace eval PR_Test_Top {
  proc get_design_libraries {} {
    set libraries [dict create]
    set libraries [dict merge $libraries [PR_Test_Top_jtag_uart_0::get_design_libraries]]
    set libraries [dict merge $libraries [PR_Test_Top_intel_niosv_g_0::get_design_libraries]]
    set libraries [dict merge $libraries [PR_Test_Top_intel_onchip_memory_0::get_design_libraries]]
    set libraries [dict merge $libraries [PR_Test_Top_reset_in::get_design_libraries]]
    set libraries [dict merge $libraries [PR_Test_Top_clock_in::get_design_libraries]]
    set libraries [dict merge $libraries [SDRAM::get_design_libraries]]
    dict set libraries altera_merlin_axi_translator_1950    1
    dict set libraries altera_merlin_slave_translator_191   1
    dict set libraries altera_merlin_axi_master_ni_1980     1
    dict set libraries altera_merlin_slave_agent_1921       1
    dict set libraries altera_avalon_sc_fifo_1932           1
    dict set libraries altera_merlin_router_1921            1
    dict set libraries altera_merlin_traffic_limiter_1921   1
    dict set libraries altera_avalon_st_pipeline_stage_1930 1
    dict set libraries altera_merlin_burst_adapter_1932     1
    dict set libraries altera_merlin_demultiplexer_1921     1
    dict set libraries altera_merlin_multiplexer_1922       1
    dict set libraries altera_mm_interconnect_1920          1
    dict set libraries altera_irq_mapper_2001               1
    dict set libraries altera_reset_controller_1922         1
    dict set libraries PR_Test_Top                          1
    return $libraries
  }
  
  proc get_memory_files {QSYS_SIMDIR} {
    set memory_files [list]
    set memory_files [concat $memory_files [PR_Test_Top_jtag_uart_0::get_memory_files "$QSYS_SIMDIR/../../ip/PR_Test_Top/PR_Test_Top_jtag_uart_0/sim/"]]
    set memory_files [concat $memory_files [PR_Test_Top_intel_niosv_g_0::get_memory_files "$QSYS_SIMDIR/../../ip/PR_Test_Top/PR_Test_Top_intel_niosv_g_0/sim/"]]
    set memory_files [concat $memory_files [PR_Test_Top_intel_onchip_memory_0::get_memory_files "$QSYS_SIMDIR/../../ip/PR_Test_Top/PR_Test_Top_intel_onchip_memory_0/sim/"]]
    set memory_files [concat $memory_files [PR_Test_Top_reset_in::get_memory_files "$QSYS_SIMDIR/../../ip/PR_Test_Top/PR_Test_Top_reset_in/sim/"]]
    set memory_files [concat $memory_files [PR_Test_Top_clock_in::get_memory_files "$QSYS_SIMDIR/../../ip/PR_Test_Top/PR_Test_Top_clock_in/sim/"]]
    set memory_files [concat $memory_files [SDRAM::get_memory_files "$QSYS_SIMDIR/../../ip/PR_Test_Top/SDRAM/sim/"]]
    return $memory_files
  }
  
  proc get_common_design_files {USER_DEFINED_COMPILE_OPTIONS USER_DEFINED_VERILOG_COMPILE_OPTIONS USER_DEFINED_VHDL_COMPILE_OPTIONS QSYS_SIMDIR} {
    set design_files [dict create]
    set design_files [dict merge $design_files [PR_Test_Top_jtag_uart_0::get_common_design_files $USER_DEFINED_COMPILE_OPTIONS $USER_DEFINED_VERILOG_COMPILE_OPTIONS $USER_DEFINED_VHDL_COMPILE_OPTIONS "$QSYS_SIMDIR/../../ip/PR_Test_Top/PR_Test_Top_jtag_uart_0/sim/"]]
    set design_files [dict merge $design_files [PR_Test_Top_intel_niosv_g_0::get_common_design_files $USER_DEFINED_COMPILE_OPTIONS $USER_DEFINED_VERILOG_COMPILE_OPTIONS $USER_DEFINED_VHDL_COMPILE_OPTIONS "$QSYS_SIMDIR/../../ip/PR_Test_Top/PR_Test_Top_intel_niosv_g_0/sim/"]]
    set design_files [dict merge $design_files [PR_Test_Top_intel_onchip_memory_0::get_common_design_files $USER_DEFINED_COMPILE_OPTIONS $USER_DEFINED_VERILOG_COMPILE_OPTIONS $USER_DEFINED_VHDL_COMPILE_OPTIONS "$QSYS_SIMDIR/../../ip/PR_Test_Top/PR_Test_Top_intel_onchip_memory_0/sim/"]]
    set design_files [dict merge $design_files [PR_Test_Top_reset_in::get_common_design_files $USER_DEFINED_COMPILE_OPTIONS $USER_DEFINED_VERILOG_COMPILE_OPTIONS $USER_DEFINED_VHDL_COMPILE_OPTIONS "$QSYS_SIMDIR/../../ip/PR_Test_Top/PR_Test_Top_reset_in/sim/"]]
    set design_files [dict merge $design_files [PR_Test_Top_clock_in::get_common_design_files $USER_DEFINED_COMPILE_OPTIONS $USER_DEFINED_VERILOG_COMPILE_OPTIONS $USER_DEFINED_VHDL_COMPILE_OPTIONS "$QSYS_SIMDIR/../../ip/PR_Test_Top/PR_Test_Top_clock_in/sim/"]]
    set design_files [dict merge $design_files [SDRAM::get_common_design_files $USER_DEFINED_COMPILE_OPTIONS $USER_DEFINED_VERILOG_COMPILE_OPTIONS $USER_DEFINED_VHDL_COMPILE_OPTIONS "$QSYS_SIMDIR/../../ip/PR_Test_Top/SDRAM/sim/"]]
    return $design_files
  }
  
  proc get_design_files {USER_DEFINED_COMPILE_OPTIONS USER_DEFINED_VERILOG_COMPILE_OPTIONS USER_DEFINED_VHDL_COMPILE_OPTIONS QSYS_SIMDIR} {
    set design_files [list]
    set design_files [concat $design_files [PR_Test_Top_jtag_uart_0::get_design_files $USER_DEFINED_COMPILE_OPTIONS $USER_DEFINED_VERILOG_COMPILE_OPTIONS $USER_DEFINED_VHDL_COMPILE_OPTIONS "$QSYS_SIMDIR/../../ip/PR_Test_Top/PR_Test_Top_jtag_uart_0/sim/"]]
    set design_files [concat $design_files [PR_Test_Top_intel_niosv_g_0::get_design_files $USER_DEFINED_COMPILE_OPTIONS $USER_DEFINED_VERILOG_COMPILE_OPTIONS $USER_DEFINED_VHDL_COMPILE_OPTIONS "$QSYS_SIMDIR/../../ip/PR_Test_Top/PR_Test_Top_intel_niosv_g_0/sim/"]]
    set design_files [concat $design_files [PR_Test_Top_intel_onchip_memory_0::get_design_files $USER_DEFINED_COMPILE_OPTIONS $USER_DEFINED_VERILOG_COMPILE_OPTIONS $USER_DEFINED_VHDL_COMPILE_OPTIONS "$QSYS_SIMDIR/../../ip/PR_Test_Top/PR_Test_Top_intel_onchip_memory_0/sim/"]]
    set design_files [concat $design_files [PR_Test_Top_reset_in::get_design_files $USER_DEFINED_COMPILE_OPTIONS $USER_DEFINED_VERILOG_COMPILE_OPTIONS $USER_DEFINED_VHDL_COMPILE_OPTIONS "$QSYS_SIMDIR/../../ip/PR_Test_Top/PR_Test_Top_reset_in/sim/"]]
    set design_files [concat $design_files [PR_Test_Top_clock_in::get_design_files $USER_DEFINED_COMPILE_OPTIONS $USER_DEFINED_VERILOG_COMPILE_OPTIONS $USER_DEFINED_VHDL_COMPILE_OPTIONS "$QSYS_SIMDIR/../../ip/PR_Test_Top/PR_Test_Top_clock_in/sim/"]]
    set design_files [concat $design_files [SDRAM::get_design_files $USER_DEFINED_COMPILE_OPTIONS $USER_DEFINED_VERILOG_COMPILE_OPTIONS $USER_DEFINED_VHDL_COMPILE_OPTIONS "$QSYS_SIMDIR/../../ip/PR_Test_Top/SDRAM/sim/"]]
    lappend design_files "vlog -sv $USER_DEFINED_VERILOG_COMPILE_OPTIONS $USER_DEFINED_COMPILE_OPTIONS  \"[normalize_path "$QSYS_SIMDIR/../altera_merlin_axi_translator_1950/sim/PR_Test_Top_altera_merlin_axi_translator_1950_sjnedva.sv"]\"  -work altera_merlin_axi_translator_1950"                       
    lappend design_files "vlog -sv $USER_DEFINED_VERILOG_COMPILE_OPTIONS $USER_DEFINED_COMPILE_OPTIONS  \"[normalize_path "$QSYS_SIMDIR/../altera_merlin_slave_translator_191/sim/PR_Test_Top_altera_merlin_slave_translator_191_x56fcki.sv"]\"  -work altera_merlin_slave_translator_191"                    
    lappend design_files "vlog -sv $USER_DEFINED_VERILOG_COMPILE_OPTIONS $USER_DEFINED_COMPILE_OPTIONS  \"[normalize_path "$QSYS_SIMDIR/../altera_merlin_axi_master_ni_1980/sim/altera_merlin_address_alignment.sv"]\"  -work altera_merlin_axi_master_ni_1980"                                               
    lappend design_files "vlog -sv $USER_DEFINED_VERILOG_COMPILE_OPTIONS $USER_DEFINED_COMPILE_OPTIONS  \"[normalize_path "$QSYS_SIMDIR/../altera_merlin_axi_master_ni_1980/sim/PR_Test_Top_altera_merlin_axi_master_ni_1980_4qd7sla.sv"]\"  -work altera_merlin_axi_master_ni_1980"                          
    lappend design_files "vlog -sv $USER_DEFINED_VERILOG_COMPILE_OPTIONS $USER_DEFINED_COMPILE_OPTIONS  \"[normalize_path "$QSYS_SIMDIR/../altera_merlin_slave_agent_1921/sim/PR_Test_Top_altera_merlin_slave_agent_1921_b6r3djy.sv"]\"  -work altera_merlin_slave_agent_1921"                                
    lappend design_files "vlog -sv $USER_DEFINED_VERILOG_COMPILE_OPTIONS $USER_DEFINED_COMPILE_OPTIONS  \"[normalize_path "$QSYS_SIMDIR/../altera_merlin_slave_agent_1921/sim/altera_merlin_burst_uncompressor.sv"]\"  -work altera_merlin_slave_agent_1921"                                                  
    lappend design_files "vlog $USER_DEFINED_VERILOG_COMPILE_OPTIONS $USER_DEFINED_COMPILE_OPTIONS  \"[normalize_path "$QSYS_SIMDIR/../altera_avalon_sc_fifo_1932/sim/PR_Test_Top_altera_avalon_sc_fifo_1932_5j7ufsq.v"]\"  -work altera_avalon_sc_fifo_1932"                                                 
    lappend design_files "vlog -sv $USER_DEFINED_VERILOG_COMPILE_OPTIONS $USER_DEFINED_COMPILE_OPTIONS  \"[normalize_path "$QSYS_SIMDIR/../altera_merlin_router_1921/sim/PR_Test_Top_altera_merlin_router_1921_6n6pnsq.sv"]\"  -work altera_merlin_router_1921"                                               
    lappend design_files "vlog -sv $USER_DEFINED_VERILOG_COMPILE_OPTIONS $USER_DEFINED_COMPILE_OPTIONS  \"[normalize_path "$QSYS_SIMDIR/../altera_merlin_router_1921/sim/PR_Test_Top_altera_merlin_router_1921_zojnl5i.sv"]\"  -work altera_merlin_router_1921"                                               
    lappend design_files "vlog -sv $USER_DEFINED_VERILOG_COMPILE_OPTIONS $USER_DEFINED_COMPILE_OPTIONS  \"[normalize_path "$QSYS_SIMDIR/../altera_merlin_router_1921/sim/PR_Test_Top_altera_merlin_router_1921_bziz2oq.sv"]\"  -work altera_merlin_router_1921"                                               
    lappend design_files "vlog -sv $USER_DEFINED_VERILOG_COMPILE_OPTIONS $USER_DEFINED_COMPILE_OPTIONS  \"[normalize_path "$QSYS_SIMDIR/../altera_merlin_router_1921/sim/PR_Test_Top_altera_merlin_router_1921_ugpjvea.sv"]\"  -work altera_merlin_router_1921"                                               
    lappend design_files "vcom $USER_DEFINED_VHDL_COMPILE_OPTIONS $USER_DEFINED_COMPILE_OPTIONS  \"[normalize_path "$QSYS_SIMDIR/../altera_merlin_traffic_limiter_1921/sim/PR_Test_Top_altera_merlin_traffic_limiter_altera_avalon_sc_fifo_1921_xbo52nq.vhd"]\"  -work altera_merlin_traffic_limiter_1921"    
    lappend design_files "vlog -sv $USER_DEFINED_VERILOG_COMPILE_OPTIONS $USER_DEFINED_COMPILE_OPTIONS  \"[normalize_path "$QSYS_SIMDIR/../altera_merlin_traffic_limiter_1921/sim/altera_merlin_reorder_memory.sv"]\"  -work altera_merlin_traffic_limiter_1921"                                              
    lappend design_files "vlog -sv $USER_DEFINED_VERILOG_COMPILE_OPTIONS $USER_DEFINED_COMPILE_OPTIONS  \"[normalize_path "$QSYS_SIMDIR/../altera_merlin_traffic_limiter_1921/sim/altera_avalon_st_pipeline_base.v"]\"  -work altera_merlin_traffic_limiter_1921"                                             
    lappend design_files "vlog -sv $USER_DEFINED_VERILOG_COMPILE_OPTIONS $USER_DEFINED_COMPILE_OPTIONS  \"[normalize_path "$QSYS_SIMDIR/../altera_merlin_traffic_limiter_1921/sim/PR_Test_Top_altera_merlin_traffic_limiter_1921_7c7jj4q.sv"]\"  -work altera_merlin_traffic_limiter_1921"                    
    lappend design_files "vlog -sv $USER_DEFINED_VERILOG_COMPILE_OPTIONS $USER_DEFINED_COMPILE_OPTIONS  \"[normalize_path "$QSYS_SIMDIR/../altera_avalon_st_pipeline_stage_1930/sim/PR_Test_Top_altera_avalon_st_pipeline_stage_1930_bv2ucky.sv"]\"  -work altera_avalon_st_pipeline_stage_1930"              
    lappend design_files "vlog -sv $USER_DEFINED_VERILOG_COMPILE_OPTIONS $USER_DEFINED_COMPILE_OPTIONS  \"[normalize_path "$QSYS_SIMDIR/../altera_avalon_st_pipeline_stage_1930/sim/altera_avalon_st_pipeline_base.v"]\"  -work altera_avalon_st_pipeline_stage_1930"                                         
    lappend design_files "vcom $USER_DEFINED_VHDL_COMPILE_OPTIONS $USER_DEFINED_COMPILE_OPTIONS  \"[normalize_path "$QSYS_SIMDIR/../altera_merlin_burst_adapter_1932/sim/PR_Test_Top_altera_merlin_burst_adapter_altera_avalon_st_pipeline_stage_1932_urolo3y.vhd"]\"  -work altera_merlin_burst_adapter_1932"
    lappend design_files "vlog -sv $USER_DEFINED_VERILOG_COMPILE_OPTIONS $USER_DEFINED_COMPILE_OPTIONS  \"[normalize_path "$QSYS_SIMDIR/../altera_merlin_burst_adapter_1932/sim/PR_Test_Top_altera_merlin_burst_adapter_1932_whxnvmi.sv"]\"  -work altera_merlin_burst_adapter_1932"                          
    lappend design_files "vlog -sv $USER_DEFINED_VERILOG_COMPILE_OPTIONS $USER_DEFINED_COMPILE_OPTIONS  \"[normalize_path "$QSYS_SIMDIR/../altera_merlin_burst_adapter_1932/sim/altera_merlin_burst_adapter_uncmpr.sv"]\"  -work altera_merlin_burst_adapter_1932"                                            
    lappend design_files "vlog -sv $USER_DEFINED_VERILOG_COMPILE_OPTIONS $USER_DEFINED_COMPILE_OPTIONS  \"[normalize_path "$QSYS_SIMDIR/../altera_merlin_burst_adapter_1932/sim/altera_merlin_burst_adapter_13_1.sv"]\"  -work altera_merlin_burst_adapter_1932"                                              
    lappend design_files "vlog -sv $USER_DEFINED_VERILOG_COMPILE_OPTIONS $USER_DEFINED_COMPILE_OPTIONS  \"[normalize_path "$QSYS_SIMDIR/../altera_merlin_burst_adapter_1932/sim/altera_merlin_burst_adapter_new.sv"]\"  -work altera_merlin_burst_adapter_1932"                                               
    lappend design_files "vlog -sv $USER_DEFINED_VERILOG_COMPILE_OPTIONS $USER_DEFINED_COMPILE_OPTIONS  \"[normalize_path "$QSYS_SIMDIR/../altera_merlin_burst_adapter_1932/sim/altera_incr_burst_converter.sv"]\"  -work altera_merlin_burst_adapter_1932"                                                   
    lappend design_files "vlog -sv $USER_DEFINED_VERILOG_COMPILE_OPTIONS $USER_DEFINED_COMPILE_OPTIONS  \"[normalize_path "$QSYS_SIMDIR/../altera_merlin_burst_adapter_1932/sim/altera_wrap_burst_converter.sv"]\"  -work altera_merlin_burst_adapter_1932"                                                   
    lappend design_files "vlog -sv $USER_DEFINED_VERILOG_COMPILE_OPTIONS $USER_DEFINED_COMPILE_OPTIONS  \"[normalize_path "$QSYS_SIMDIR/../altera_merlin_burst_adapter_1932/sim/altera_default_burst_converter.sv"]\"  -work altera_merlin_burst_adapter_1932"                                                
    lappend design_files "vlog -sv $USER_DEFINED_VERILOG_COMPILE_OPTIONS $USER_DEFINED_COMPILE_OPTIONS  \"[normalize_path "$QSYS_SIMDIR/../altera_merlin_burst_adapter_1932/sim/altera_merlin_address_alignment.sv"]\"  -work altera_merlin_burst_adapter_1932"                                               
    lappend design_files "vlog -sv $USER_DEFINED_VERILOG_COMPILE_OPTIONS $USER_DEFINED_COMPILE_OPTIONS  \"[normalize_path "$QSYS_SIMDIR/../altera_merlin_demultiplexer_1921/sim/PR_Test_Top_altera_merlin_demultiplexer_1921_37cgyry.sv"]\"  -work altera_merlin_demultiplexer_1921"                          
    lappend design_files "vlog -sv $USER_DEFINED_VERILOG_COMPILE_OPTIONS $USER_DEFINED_COMPILE_OPTIONS  \"[normalize_path "$QSYS_SIMDIR/../altera_merlin_demultiplexer_1921/sim/PR_Test_Top_altera_merlin_demultiplexer_1921_ge4sdza.sv"]\"  -work altera_merlin_demultiplexer_1921"                          
    lappend design_files "vlog -sv $USER_DEFINED_VERILOG_COMPILE_OPTIONS $USER_DEFINED_COMPILE_OPTIONS  \"[normalize_path "$QSYS_SIMDIR/../altera_merlin_multiplexer_1922/sim/PR_Test_Top_altera_merlin_multiplexer_1922_npoxlmq.sv"]\"  -work altera_merlin_multiplexer_1922"                                
    lappend design_files "vlog -sv $USER_DEFINED_VERILOG_COMPILE_OPTIONS $USER_DEFINED_COMPILE_OPTIONS  \"[normalize_path "$QSYS_SIMDIR/../altera_merlin_multiplexer_1922/sim/altera_merlin_arbitrator.sv"]\"  -work altera_merlin_multiplexer_1922"                                                          
    lappend design_files "vlog -sv $USER_DEFINED_VERILOG_COMPILE_OPTIONS $USER_DEFINED_COMPILE_OPTIONS  \"[normalize_path "$QSYS_SIMDIR/../altera_merlin_multiplexer_1922/sim/PR_Test_Top_altera_merlin_multiplexer_1922_so763ra.sv"]\"  -work altera_merlin_multiplexer_1922"                                
    lappend design_files "vlog -sv $USER_DEFINED_VERILOG_COMPILE_OPTIONS $USER_DEFINED_COMPILE_OPTIONS  \"[normalize_path "$QSYS_SIMDIR/../altera_merlin_multiplexer_1922/sim/altera_merlin_arbitrator.sv"]\"  -work altera_merlin_multiplexer_1922"                                                          
    lappend design_files "vlog -sv $USER_DEFINED_VERILOG_COMPILE_OPTIONS $USER_DEFINED_COMPILE_OPTIONS  \"[normalize_path "$QSYS_SIMDIR/../altera_merlin_demultiplexer_1921/sim/PR_Test_Top_altera_merlin_demultiplexer_1921_l2pgmgq.sv"]\"  -work altera_merlin_demultiplexer_1921"                          
    lappend design_files "vlog -sv $USER_DEFINED_VERILOG_COMPILE_OPTIONS $USER_DEFINED_COMPILE_OPTIONS  \"[normalize_path "$QSYS_SIMDIR/../altera_merlin_multiplexer_1922/sim/PR_Test_Top_altera_merlin_multiplexer_1922_nkkbquq.sv"]\"  -work altera_merlin_multiplexer_1922"                                
    lappend design_files "vlog -sv $USER_DEFINED_VERILOG_COMPILE_OPTIONS $USER_DEFINED_COMPILE_OPTIONS  \"[normalize_path "$QSYS_SIMDIR/../altera_merlin_multiplexer_1922/sim/altera_merlin_arbitrator.sv"]\"  -work altera_merlin_multiplexer_1922"                                                          
    lappend design_files "vlog -sv $USER_DEFINED_VERILOG_COMPILE_OPTIONS $USER_DEFINED_COMPILE_OPTIONS  \"[normalize_path "$QSYS_SIMDIR/../altera_merlin_multiplexer_1922/sim/PR_Test_Top_altera_merlin_multiplexer_1922_aaxrxoq.sv"]\"  -work altera_merlin_multiplexer_1922"                                
    lappend design_files "vlog -sv $USER_DEFINED_VERILOG_COMPILE_OPTIONS $USER_DEFINED_COMPILE_OPTIONS  \"[normalize_path "$QSYS_SIMDIR/../altera_merlin_multiplexer_1922/sim/altera_merlin_arbitrator.sv"]\"  -work altera_merlin_multiplexer_1922"                                                          
    lappend design_files "vcom $USER_DEFINED_VHDL_COMPILE_OPTIONS $USER_DEFINED_COMPILE_OPTIONS  \"[normalize_path "$QSYS_SIMDIR/../altera_mm_interconnect_1920/sim/pr_test_top_altera_mm_interconnect_1920_ho7sqxa_intel_niosv_g_0_data_manager_translator.vhd"]\"  -work altera_mm_interconnect_1920"       
    lappend design_files "vcom $USER_DEFINED_VHDL_COMPILE_OPTIONS $USER_DEFINED_COMPILE_OPTIONS  \"[normalize_path "$QSYS_SIMDIR/../altera_mm_interconnect_1920/sim/pr_test_top_altera_mm_interconnect_1920_ho7sqxa_intel_niosv_g_0_instruction_manager_translator.vhd"]\"  -work altera_mm_interconnect_1920"
    lappend design_files "vcom $USER_DEFINED_VHDL_COMPILE_OPTIONS $USER_DEFINED_COMPILE_OPTIONS  \"[normalize_path "$QSYS_SIMDIR/../altera_mm_interconnect_1920/sim/pr_test_top_altera_mm_interconnect_1920_ho7sqxa_jtag_uart_0_avalon_jtag_slave_agent_rsp_fifo.vhd"]\"  -work altera_mm_interconnect_1920"  
    lappend design_files "vcom $USER_DEFINED_VHDL_COMPILE_OPTIONS $USER_DEFINED_COMPILE_OPTIONS  \"[normalize_path "$QSYS_SIMDIR/../altera_mm_interconnect_1920/sim/pr_test_top_altera_mm_interconnect_1920_ho7sqxa_jtag_uart_0_avalon_jtag_slave_agent_rdata_fifo.vhd"]\"  -work altera_mm_interconnect_1920"
    lappend design_files "vcom $USER_DEFINED_VHDL_COMPILE_OPTIONS $USER_DEFINED_COMPILE_OPTIONS  \"[normalize_path "$QSYS_SIMDIR/../altera_mm_interconnect_1920/sim/pr_test_top_altera_mm_interconnect_1920_ho7sqxa_jtag_uart_0_avalon_jtag_slave_translator.vhd"]\"  -work altera_mm_interconnect_1920"      
    lappend design_files "vcom $USER_DEFINED_VHDL_COMPILE_OPTIONS $USER_DEFINED_COMPILE_OPTIONS  \"[normalize_path "$QSYS_SIMDIR/../altera_mm_interconnect_1920/sim/pr_test_top_altera_mm_interconnect_1920_ho7sqxa_intel_niosv_g_0_dm_agent_translator.vhd"]\"  -work altera_mm_interconnect_1920"           
    lappend design_files "vcom $USER_DEFINED_VHDL_COMPILE_OPTIONS $USER_DEFINED_COMPILE_OPTIONS  \"[normalize_path "$QSYS_SIMDIR/../altera_mm_interconnect_1920/sim/pr_test_top_altera_mm_interconnect_1920_ho7sqxa_intel_onchip_memory_0_s1_translator.vhd"]\"  -work altera_mm_interconnect_1920"           
    lappend design_files "vcom $USER_DEFINED_VHDL_COMPILE_OPTIONS $USER_DEFINED_COMPILE_OPTIONS  \"[normalize_path "$QSYS_SIMDIR/../altera_mm_interconnect_1920/sim/pr_test_top_altera_mm_interconnect_1920_ho7sqxa_intel_niosv_g_0_timer_sw_agent_translator.vhd"]\"  -work altera_mm_interconnect_1920"     
    lappend design_files "vcom $USER_DEFINED_VHDL_COMPILE_OPTIONS $USER_DEFINED_COMPILE_OPTIONS  \"[normalize_path "$QSYS_SIMDIR/../altera_mm_interconnect_1920/sim/PR_Test_Top_altera_mm_interconnect_1920_ho7sqxa.vhd"]\"  -work altera_mm_interconnect_1920"                                               
    lappend design_files "vlog -sv $USER_DEFINED_VERILOG_COMPILE_OPTIONS $USER_DEFINED_COMPILE_OPTIONS  \"[normalize_path "$QSYS_SIMDIR/../altera_irq_mapper_2001/sim/PR_Test_Top_altera_irq_mapper_2001_ghcid5i.sv"]\"  -work altera_irq_mapper_2001"                                                        
    lappend design_files "vlog $USER_DEFINED_VERILOG_COMPILE_OPTIONS $USER_DEFINED_COMPILE_OPTIONS  \"[normalize_path "$QSYS_SIMDIR/../altera_reset_controller_1922/sim/mentor/altera_reset_controller.v"]\"  -work altera_reset_controller_1922"                                                             
    lappend design_files "vlog $USER_DEFINED_VERILOG_COMPILE_OPTIONS $USER_DEFINED_COMPILE_OPTIONS  \"[normalize_path "$QSYS_SIMDIR/../altera_reset_controller_1922/sim/mentor/altera_reset_synchronizer.v"]\"  -work altera_reset_controller_1922"                                                           
    lappend design_files "vcom $USER_DEFINED_VHDL_COMPILE_OPTIONS $USER_DEFINED_COMPILE_OPTIONS  \"[normalize_path "$QSYS_SIMDIR/PR_Test_Top.vhd"]\"  -work PR_Test_Top"                                                                                                                                      
    return $design_files
  }
  
  proc get_non_duplicate_elab_option {ELAB_OPTIONS NEW_ELAB_OPTION} {
    set IS_DUPLICATE [string first $NEW_ELAB_OPTION $ELAB_OPTIONS]
    if {$IS_DUPLICATE == -1} {
      return $NEW_ELAB_OPTION
    } else {
      return ""
    }
  }
  
  
  proc get_elab_options {SIMULATOR_TOOL_BITNESS} {
    set ELAB_OPTIONS ""
    append ELAB_OPTIONS [get_non_duplicate_elab_option $ELAB_OPTIONS [PR_Test_Top_jtag_uart_0::get_elab_options $SIMULATOR_TOOL_BITNESS]]
    append ELAB_OPTIONS [get_non_duplicate_elab_option $ELAB_OPTIONS [PR_Test_Top_intel_niosv_g_0::get_elab_options $SIMULATOR_TOOL_BITNESS]]
    append ELAB_OPTIONS [get_non_duplicate_elab_option $ELAB_OPTIONS [PR_Test_Top_intel_onchip_memory_0::get_elab_options $SIMULATOR_TOOL_BITNESS]]
    append ELAB_OPTIONS [get_non_duplicate_elab_option $ELAB_OPTIONS [PR_Test_Top_reset_in::get_elab_options $SIMULATOR_TOOL_BITNESS]]
    append ELAB_OPTIONS [get_non_duplicate_elab_option $ELAB_OPTIONS [PR_Test_Top_clock_in::get_elab_options $SIMULATOR_TOOL_BITNESS]]
    append ELAB_OPTIONS [get_non_duplicate_elab_option $ELAB_OPTIONS [SDRAM::get_elab_options $SIMULATOR_TOOL_BITNESS]]
    if ![ string match "bit_64" $SIMULATOR_TOOL_BITNESS ] {
    } else {
    }
    append ELAB_OPTIONS { -t fs}
    return $ELAB_OPTIONS
  }
  
  
  proc get_sim_options {SIMULATOR_TOOL_BITNESS} {
    set SIM_OPTIONS ""
    append SIM_OPTIONS [PR_Test_Top_jtag_uart_0::get_sim_options $SIMULATOR_TOOL_BITNESS]
    append SIM_OPTIONS [PR_Test_Top_intel_niosv_g_0::get_sim_options $SIMULATOR_TOOL_BITNESS]
    append SIM_OPTIONS [PR_Test_Top_intel_onchip_memory_0::get_sim_options $SIMULATOR_TOOL_BITNESS]
    append SIM_OPTIONS [PR_Test_Top_reset_in::get_sim_options $SIMULATOR_TOOL_BITNESS]
    append SIM_OPTIONS [PR_Test_Top_clock_in::get_sim_options $SIMULATOR_TOOL_BITNESS]
    append SIM_OPTIONS [SDRAM::get_sim_options $SIMULATOR_TOOL_BITNESS]
    if ![ string match "bit_64" $SIMULATOR_TOOL_BITNESS ] {
    } else {
    }
    return $SIM_OPTIONS
  }
  
  
  proc get_env_variables {SIMULATOR_TOOL_BITNESS} {
    set ENV_VARIABLES [dict create]
    set LD_LIBRARY_PATH [dict create]
    set LD_LIBRARY_PATH [dict merge $LD_LIBRARY_PATH [dict get [PR_Test_Top_jtag_uart_0::get_env_variables $SIMULATOR_TOOL_BITNESS] "LD_LIBRARY_PATH"]]
    set LD_LIBRARY_PATH [dict merge $LD_LIBRARY_PATH [dict get [PR_Test_Top_intel_niosv_g_0::get_env_variables $SIMULATOR_TOOL_BITNESS] "LD_LIBRARY_PATH"]]
    set LD_LIBRARY_PATH [dict merge $LD_LIBRARY_PATH [dict get [PR_Test_Top_intel_onchip_memory_0::get_env_variables $SIMULATOR_TOOL_BITNESS] "LD_LIBRARY_PATH"]]
    set LD_LIBRARY_PATH [dict merge $LD_LIBRARY_PATH [dict get [PR_Test_Top_reset_in::get_env_variables $SIMULATOR_TOOL_BITNESS] "LD_LIBRARY_PATH"]]
    set LD_LIBRARY_PATH [dict merge $LD_LIBRARY_PATH [dict get [PR_Test_Top_clock_in::get_env_variables $SIMULATOR_TOOL_BITNESS] "LD_LIBRARY_PATH"]]
    set LD_LIBRARY_PATH [dict merge $LD_LIBRARY_PATH [dict get [SDRAM::get_env_variables $SIMULATOR_TOOL_BITNESS] "LD_LIBRARY_PATH"]]
    dict set ENV_VARIABLES "LD_LIBRARY_PATH" $LD_LIBRARY_PATH
    if ![ string match "bit_64" $SIMULATOR_TOOL_BITNESS ] {
    } else {
    }
    return $ENV_VARIABLES
  }
  
  
  proc normalize_path {FILEPATH} {
      if {[catch { package require fileutil } err]} { 
          return $FILEPATH 
      } 
      set path [fileutil::lexnormalize [file join [pwd] $FILEPATH]]  
      if {[file pathtype $FILEPATH] eq "relative"} { 
          set path [fileutil::relative [pwd] $path] 
      } 
      return $path 
  } 
  proc get_dpi_libraries {QSYS_SIMDIR} {
    set libraries [dict create]
    set libraries [dict merge $libraries [PR_Test_Top_jtag_uart_0::get_dpi_libraries "$QSYS_SIMDIR/../../ip/PR_Test_Top/PR_Test_Top_jtag_uart_0/sim/"]]
    set libraries [dict merge $libraries [PR_Test_Top_intel_niosv_g_0::get_dpi_libraries "$QSYS_SIMDIR/../../ip/PR_Test_Top/PR_Test_Top_intel_niosv_g_0/sim/"]]
    set libraries [dict merge $libraries [PR_Test_Top_intel_onchip_memory_0::get_dpi_libraries "$QSYS_SIMDIR/../../ip/PR_Test_Top/PR_Test_Top_intel_onchip_memory_0/sim/"]]
    set libraries [dict merge $libraries [PR_Test_Top_reset_in::get_dpi_libraries "$QSYS_SIMDIR/../../ip/PR_Test_Top/PR_Test_Top_reset_in/sim/"]]
    set libraries [dict merge $libraries [PR_Test_Top_clock_in::get_dpi_libraries "$QSYS_SIMDIR/../../ip/PR_Test_Top/PR_Test_Top_clock_in/sim/"]]
    set libraries [dict merge $libraries [SDRAM::get_dpi_libraries "$QSYS_SIMDIR/../../ip/PR_Test_Top/SDRAM/sim/"]]
    
    return $libraries
  }
  
}
