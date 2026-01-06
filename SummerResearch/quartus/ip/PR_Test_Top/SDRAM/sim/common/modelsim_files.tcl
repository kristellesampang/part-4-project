
namespace eval SDRAM {
  proc get_design_libraries {} {
    set libraries [dict create]
    dict set libraries altera_emif_arch_nf_191                1
    dict set libraries altera_avalon_mm_bridge_2010           1
    dict set libraries altera_jtag_dc_streaming_191           1
    dict set libraries timing_adapter_1940                    1
    dict set libraries altera_avalon_sc_fifo_1932             1
    dict set libraries altera_avalon_st_bytes_to_packets_1922 1
    dict set libraries altera_avalon_st_packets_to_bytes_1922 1
    dict set libraries altera_avalon_packets_to_master_1922   1
    dict set libraries channel_adapter_1921                   1
    dict set libraries altera_reset_controller_1922           1
    dict set libraries alt_mem_if_jtag_master_191             1
    dict set libraries altera_merlin_master_translator_192    1
    dict set libraries altera_merlin_slave_translator_191     1
    dict set libraries altera_mm_interconnect_1920            1
    dict set libraries altera_ip_col_if_1911                  1
    dict set libraries altera_avalon_onchip_memory2_1939      1
    dict set libraries altera_emif_cal_slave_nf_191           1
    dict set libraries altera_emif_1923                       1
    dict set libraries SDRAM                                  1
    return $libraries
  }
  
  proc get_memory_files {QSYS_SIMDIR} {
    set memory_files [list]
    lappend memory_files "[normalize_path "$QSYS_SIMDIR/../altera_emif_arch_nf_191/sim/SDRAM_altera_emif_arch_nf_191_65stwui_seq_params_sim.hex"]"
    lappend memory_files "[normalize_path "$QSYS_SIMDIR/../altera_emif_arch_nf_191/sim/SDRAM_altera_emif_arch_nf_191_65stwui_seq_params_synth.hex"]"
    lappend memory_files "[normalize_path "$QSYS_SIMDIR/../altera_emif_arch_nf_191/sim/SDRAM_altera_emif_arch_nf_191_65stwui_seq_cal.hex"]"
    lappend memory_files "[normalize_path "$QSYS_SIMDIR/../altera_avalon_onchip_memory2_1939/sim/seq_cal_soft_m20k.hex"]"
    return $memory_files
  }
  
  proc get_common_design_files {USER_DEFINED_COMPILE_OPTIONS USER_DEFINED_VERILOG_COMPILE_OPTIONS USER_DEFINED_VHDL_COMPILE_OPTIONS QSYS_SIMDIR} {
    set design_files [dict create]
    return $design_files
  }
  
  proc get_design_files {USER_DEFINED_COMPILE_OPTIONS USER_DEFINED_VERILOG_COMPILE_OPTIONS USER_DEFINED_VHDL_COMPILE_OPTIONS QSYS_SIMDIR} {
    set design_files [list]
    lappend design_files "vlog -sv $USER_DEFINED_VERILOG_COMPILE_OPTIONS $USER_DEFINED_COMPILE_OPTIONS  \"[normalize_path "$QSYS_SIMDIR/../altera_emif_arch_nf_191/sim/SDRAM_altera_emif_arch_nf_191_65stwui_top.sv"]\"  -work altera_emif_arch_nf_191"                                
    lappend design_files "vlog -sv $USER_DEFINED_VERILOG_COMPILE_OPTIONS $USER_DEFINED_COMPILE_OPTIONS  \"[normalize_path "$QSYS_SIMDIR/../altera_emif_arch_nf_191/sim/SDRAM_altera_emif_arch_nf_191_65stwui_io_aux.sv"]\"  -work altera_emif_arch_nf_191"                             
    lappend design_files "vlog -sv $USER_DEFINED_VERILOG_COMPILE_OPTIONS $USER_DEFINED_COMPILE_OPTIONS  \"[normalize_path "$QSYS_SIMDIR/../altera_emif_arch_nf_191/sim/altera_emif_arch_nf_oct.sv"]\"  -work altera_emif_arch_nf_191"                                                  
    lappend design_files "vlog -sv $USER_DEFINED_VERILOG_COMPILE_OPTIONS $USER_DEFINED_COMPILE_OPTIONS  \"[normalize_path "$QSYS_SIMDIR/../altera_emif_arch_nf_191/sim/altera_emif_arch_nf_buf_udir_df_o.sv"]\"  -work altera_emif_arch_nf_191"                                        
    lappend design_files "vlog -sv $USER_DEFINED_VERILOG_COMPILE_OPTIONS $USER_DEFINED_COMPILE_OPTIONS  \"[normalize_path "$QSYS_SIMDIR/../altera_emif_arch_nf_191/sim/altera_emif_arch_nf_buf_bdir_df.sv"]\"  -work altera_emif_arch_nf_191"                                          
    lappend design_files "vlog -sv $USER_DEFINED_VERILOG_COMPILE_OPTIONS $USER_DEFINED_COMPILE_OPTIONS  \"[normalize_path "$QSYS_SIMDIR/../altera_emif_arch_nf_191/sim/altera_emif_arch_nf_buf_bdir_se.sv"]\"  -work altera_emif_arch_nf_191"                                          
    lappend design_files "vlog -sv $USER_DEFINED_VERILOG_COMPILE_OPTIONS $USER_DEFINED_COMPILE_OPTIONS  \"[normalize_path "$QSYS_SIMDIR/../altera_emif_arch_nf_191/sim/altera_emif_arch_nf_buf_udir_cp_i.sv"]\"  -work altera_emif_arch_nf_191"                                        
    lappend design_files "vlog -sv $USER_DEFINED_VERILOG_COMPILE_OPTIONS $USER_DEFINED_COMPILE_OPTIONS  \"[normalize_path "$QSYS_SIMDIR/../altera_emif_arch_nf_191/sim/altera_emif_arch_nf_buf_udir_df_i.sv"]\"  -work altera_emif_arch_nf_191"                                        
    lappend design_files "vlog -sv $USER_DEFINED_VERILOG_COMPILE_OPTIONS $USER_DEFINED_COMPILE_OPTIONS  \"[normalize_path "$QSYS_SIMDIR/../altera_emif_arch_nf_191/sim/altera_emif_arch_nf_buf_udir_se_i.sv"]\"  -work altera_emif_arch_nf_191"                                        
    lappend design_files "vlog -sv $USER_DEFINED_VERILOG_COMPILE_OPTIONS $USER_DEFINED_COMPILE_OPTIONS  \"[normalize_path "$QSYS_SIMDIR/../altera_emif_arch_nf_191/sim/altera_emif_arch_nf_buf_udir_se_o.sv"]\"  -work altera_emif_arch_nf_191"                                        
    lappend design_files "vlog -sv $USER_DEFINED_VERILOG_COMPILE_OPTIONS $USER_DEFINED_COMPILE_OPTIONS  \"[normalize_path "$QSYS_SIMDIR/../altera_emif_arch_nf_191/sim/altera_emif_arch_nf_core_clks_rsts.sv"]\"  -work altera_emif_arch_nf_191"                                       
    lappend design_files "vlog -sv $USER_DEFINED_VERILOG_COMPILE_OPTIONS $USER_DEFINED_COMPILE_OPTIONS  \"[normalize_path "$QSYS_SIMDIR/../altera_emif_arch_nf_191/sim/altera_emif_arch_nf_io_tiles.sv"]\"  -work altera_emif_arch_nf_191"                                             
    lappend design_files "vlog -sv $USER_DEFINED_VERILOG_COMPILE_OPTIONS $USER_DEFINED_COMPILE_OPTIONS  \"[normalize_path "$QSYS_SIMDIR/../altera_emif_arch_nf_191/sim/altera_emif_arch_nf_io_tiles_abphy.sv"]\"  -work altera_emif_arch_nf_191"                                       
    lappend design_files "vlog -sv $USER_DEFINED_VERILOG_COMPILE_OPTIONS $USER_DEFINED_COMPILE_OPTIONS  \"[normalize_path "$QSYS_SIMDIR/../altera_emif_arch_nf_191/sim/altera_emif_arch_nf_pll.sv"]\"  -work altera_emif_arch_nf_191"                                                  
    lappend design_files "vlog -sv $USER_DEFINED_VERILOG_COMPILE_OPTIONS $USER_DEFINED_COMPILE_OPTIONS  \"[normalize_path "$QSYS_SIMDIR/../altera_emif_arch_nf_191/sim/twentynm_io_12_lane_abphy.sv"]\"  -work altera_emif_arch_nf_191"                                                
    lappend design_files "vlog -sv $USER_DEFINED_VERILOG_COMPILE_OPTIONS $USER_DEFINED_COMPILE_OPTIONS  \"[normalize_path "$QSYS_SIMDIR/../altera_emif_arch_nf_191/sim/twentynm_io_12_lane_encrypted_abphy.sv"]\"  -work altera_emif_arch_nf_191"                                      
    lappend design_files "vlog -sv $USER_DEFINED_VERILOG_COMPILE_OPTIONS $USER_DEFINED_COMPILE_OPTIONS  \"[normalize_path "$QSYS_SIMDIR/../altera_emif_arch_nf_191/sim/twentynm_io_12_lane_nf5es_encrypted_abphy.sv"]\"  -work altera_emif_arch_nf_191"                                
    lappend design_files "vlog -sv $USER_DEFINED_VERILOG_COMPILE_OPTIONS $USER_DEFINED_COMPILE_OPTIONS  \"[normalize_path "$QSYS_SIMDIR/../altera_emif_arch_nf_191/sim/altera_emif_arch_nf_bufs.sv"]\"  -work altera_emif_arch_nf_191"                                                 
    lappend design_files "vlog -sv $USER_DEFINED_VERILOG_COMPILE_OPTIONS $USER_DEFINED_COMPILE_OPTIONS  \"[normalize_path "$QSYS_SIMDIR/../altera_emif_arch_nf_191/sim/altera_emif_arch_nf_buf_unused.sv"]\"  -work altera_emif_arch_nf_191"                                           
    lappend design_files "vlog -sv $USER_DEFINED_VERILOG_COMPILE_OPTIONS $USER_DEFINED_COMPILE_OPTIONS  \"[normalize_path "$QSYS_SIMDIR/../altera_emif_arch_nf_191/sim/altera_emif_arch_nf_cal_counter.sv"]\"  -work altera_emif_arch_nf_191"                                          
    lappend design_files "vlog -sv $USER_DEFINED_VERILOG_COMPILE_OPTIONS $USER_DEFINED_COMPILE_OPTIONS  \"[normalize_path "$QSYS_SIMDIR/../altera_emif_arch_nf_191/sim/altera_emif_arch_nf_pll_fast_sim.sv"]\"  -work altera_emif_arch_nf_191"                                         
    lappend design_files "vlog -sv $USER_DEFINED_VERILOG_COMPILE_OPTIONS $USER_DEFINED_COMPILE_OPTIONS  \"[normalize_path "$QSYS_SIMDIR/../altera_emif_arch_nf_191/sim/altera_emif_arch_nf_pll_extra_clks.sv"]\"  -work altera_emif_arch_nf_191"                                       
    lappend design_files "vlog -sv $USER_DEFINED_VERILOG_COMPILE_OPTIONS $USER_DEFINED_COMPILE_OPTIONS  \"[normalize_path "$QSYS_SIMDIR/../altera_emif_arch_nf_191/sim/altera_emif_arch_nf_hps_clks_rsts.sv"]\"  -work altera_emif_arch_nf_191"                                        
    lappend design_files "vlog -sv $USER_DEFINED_VERILOG_COMPILE_OPTIONS $USER_DEFINED_COMPILE_OPTIONS  \"[normalize_path "$QSYS_SIMDIR/../altera_emif_arch_nf_191/sim/altera_emif_arch_nf_io_tiles_wrap.sv"]\"  -work altera_emif_arch_nf_191"                                        
    lappend design_files "vlog -sv $USER_DEFINED_VERILOG_COMPILE_OPTIONS $USER_DEFINED_COMPILE_OPTIONS  \"[normalize_path "$QSYS_SIMDIR/../altera_emif_arch_nf_191/sim/altera_emif_arch_nf_abphy_mux.sv"]\"  -work altera_emif_arch_nf_191"                                            
    lappend design_files "vlog -sv $USER_DEFINED_VERILOG_COMPILE_OPTIONS $USER_DEFINED_COMPILE_OPTIONS  \"[normalize_path "$QSYS_SIMDIR/../altera_emif_arch_nf_191/sim/altera_emif_arch_nf_hmc_avl_if.sv"]\"  -work altera_emif_arch_nf_191"                                           
    lappend design_files "vlog -sv $USER_DEFINED_VERILOG_COMPILE_OPTIONS $USER_DEFINED_COMPILE_OPTIONS  \"[normalize_path "$QSYS_SIMDIR/../altera_emif_arch_nf_191/sim/altera_emif_arch_nf_hmc_sideband_if.sv"]\"  -work altera_emif_arch_nf_191"                                      
    lappend design_files "vlog -sv $USER_DEFINED_VERILOG_COMPILE_OPTIONS $USER_DEFINED_COMPILE_OPTIONS  \"[normalize_path "$QSYS_SIMDIR/../altera_emif_arch_nf_191/sim/altera_emif_arch_nf_hmc_mmr_if.sv"]\"  -work altera_emif_arch_nf_191"                                           
    lappend design_files "vlog -sv $USER_DEFINED_VERILOG_COMPILE_OPTIONS $USER_DEFINED_COMPILE_OPTIONS  \"[normalize_path "$QSYS_SIMDIR/../altera_emif_arch_nf_191/sim/altera_emif_arch_nf_hmc_amm_data_if.sv"]\"  -work altera_emif_arch_nf_191"                                      
    lappend design_files "vlog -sv $USER_DEFINED_VERILOG_COMPILE_OPTIONS $USER_DEFINED_COMPILE_OPTIONS  \"[normalize_path "$QSYS_SIMDIR/../altera_emif_arch_nf_191/sim/altera_emif_arch_nf_hmc_ast_data_if.sv"]\"  -work altera_emif_arch_nf_191"                                      
    lappend design_files "vlog -sv $USER_DEFINED_VERILOG_COMPILE_OPTIONS $USER_DEFINED_COMPILE_OPTIONS  \"[normalize_path "$QSYS_SIMDIR/../altera_emif_arch_nf_191/sim/altera_emif_arch_nf_afi_if.sv"]\"  -work altera_emif_arch_nf_191"                                               
    lappend design_files "vlog -sv $USER_DEFINED_VERILOG_COMPILE_OPTIONS $USER_DEFINED_COMPILE_OPTIONS  \"[normalize_path "$QSYS_SIMDIR/../altera_emif_arch_nf_191/sim/altera_emif_arch_nf_seq_if.sv"]\"  -work altera_emif_arch_nf_191"                                               
    lappend design_files "vlog -sv $USER_DEFINED_VERILOG_COMPILE_OPTIONS $USER_DEFINED_COMPILE_OPTIONS  \"[normalize_path "$QSYS_SIMDIR/../altera_emif_arch_nf_191/sim/altera_emif_arch_nf_regs.sv"]\"  -work altera_emif_arch_nf_191"                                                 
    lappend design_files "vlog -sv $USER_DEFINED_VERILOG_COMPILE_OPTIONS $USER_DEFINED_COMPILE_OPTIONS  \"[normalize_path "$QSYS_SIMDIR/../altera_emif_arch_nf_191/sim/altera_oct.sv"]\"  -work altera_emif_arch_nf_191"                                                               
    lappend design_files "vlog -sv $USER_DEFINED_VERILOG_COMPILE_OPTIONS $USER_DEFINED_COMPILE_OPTIONS  \"[normalize_path "$QSYS_SIMDIR/../altera_emif_arch_nf_191/sim/altera_oct_um_fsm.sv"]\"  -work altera_emif_arch_nf_191"                                                        
    lappend design_files "vlog $USER_DEFINED_VERILOG_COMPILE_OPTIONS $USER_DEFINED_COMPILE_OPTIONS  \"[normalize_path "$QSYS_SIMDIR/../altera_emif_arch_nf_191/sim/altera_std_synchronizer_nocut.v"]\"  -work altera_emif_arch_nf_191"                                                 
    lappend design_files "vlog -sv $USER_DEFINED_VERILOG_COMPILE_OPTIONS $USER_DEFINED_COMPILE_OPTIONS  \"[normalize_path "$QSYS_SIMDIR/../altera_emif_arch_nf_191/sim/mem_array_abphy.sv"]\"  -work altera_emif_arch_nf_191"                                                          
    lappend design_files "vlog -sv $USER_DEFINED_VERILOG_COMPILE_OPTIONS $USER_DEFINED_COMPILE_OPTIONS  \"[normalize_path "$QSYS_SIMDIR/../altera_emif_arch_nf_191/sim/io_12_lane_bcm__nf5es_abphy.sv"]\"  -work altera_emif_arch_nf_191"                                              
    lappend design_files "vlog -sv $USER_DEFINED_VERILOG_COMPILE_OPTIONS $USER_DEFINED_COMPILE_OPTIONS  \"[normalize_path "$QSYS_SIMDIR/../altera_emif_arch_nf_191/sim/io_12_lane__nf5es_abphy.sv"]\"  -work altera_emif_arch_nf_191"                                                  
    lappend design_files "vcom $USER_DEFINED_VHDL_COMPILE_OPTIONS $USER_DEFINED_COMPILE_OPTIONS  \"[normalize_path "$QSYS_SIMDIR/../altera_emif_arch_nf_191/sim/SDRAM_altera_emif_arch_nf_191_65stwui.vhd"]\"  -work altera_emif_arch_nf_191"                                          
    lappend design_files "vlog $USER_DEFINED_VERILOG_COMPILE_OPTIONS $USER_DEFINED_COMPILE_OPTIONS  \"[normalize_path "$QSYS_SIMDIR/../altera_avalon_mm_bridge_2010/sim/SDRAM_altera_avalon_mm_bridge_2010_xnk2xbi.v"]\"  -work altera_avalon_mm_bridge_2010"                          
    lappend design_files "vlog $USER_DEFINED_VERILOG_COMPILE_OPTIONS $USER_DEFINED_COMPILE_OPTIONS  \"[normalize_path "$QSYS_SIMDIR/../altera_avalon_mm_bridge_2010/sim/altera_merlin_waitrequest_adapter.v"]\"  -work altera_avalon_mm_bridge_2010"                                   
    lappend design_files "vlog $USER_DEFINED_VERILOG_COMPILE_OPTIONS $USER_DEFINED_COMPILE_OPTIONS  \"[normalize_path "$QSYS_SIMDIR/../altera_avalon_mm_bridge_2010/sim/altera_avalon_sc_fifo.v"]\"  -work altera_avalon_mm_bridge_2010"                                               
    lappend design_files "vlog $USER_DEFINED_VERILOG_COMPILE_OPTIONS $USER_DEFINED_COMPILE_OPTIONS  \"[normalize_path "$QSYS_SIMDIR/../altera_jtag_dc_streaming_191/sim/altera_avalon_st_jtag_interface.v"]\"  -work altera_jtag_dc_streaming_191"                                     
    lappend design_files "vlog $USER_DEFINED_VERILOG_COMPILE_OPTIONS $USER_DEFINED_COMPILE_OPTIONS  \"[normalize_path "$QSYS_SIMDIR/../altera_jtag_dc_streaming_191/sim/altera_jtag_dc_streaming.v"]\"  -work altera_jtag_dc_streaming_191"                                            
    lappend design_files "vlog $USER_DEFINED_VERILOG_COMPILE_OPTIONS $USER_DEFINED_COMPILE_OPTIONS  \"[normalize_path "$QSYS_SIMDIR/../altera_jtag_dc_streaming_191/sim/altera_jtag_sld_node.v"]\"  -work altera_jtag_dc_streaming_191"                                                
    lappend design_files "vlog $USER_DEFINED_VERILOG_COMPILE_OPTIONS $USER_DEFINED_COMPILE_OPTIONS  \"[normalize_path "$QSYS_SIMDIR/../altera_jtag_dc_streaming_191/sim/altera_jtag_streaming.v"]\"  -work altera_jtag_dc_streaming_191"                                               
    lappend design_files "vlog $USER_DEFINED_VERILOG_COMPILE_OPTIONS $USER_DEFINED_COMPILE_OPTIONS  \"[normalize_path "$QSYS_SIMDIR/../altera_jtag_dc_streaming_191/sim/altera_avalon_st_clock_crosser.v"]\"  -work altera_jtag_dc_streaming_191"                                      
    lappend design_files "vlog $USER_DEFINED_VERILOG_COMPILE_OPTIONS $USER_DEFINED_COMPILE_OPTIONS  \"[normalize_path "$QSYS_SIMDIR/../altera_jtag_dc_streaming_191/sim/altera_reset_synchronizer.v"]\"  -work altera_jtag_dc_streaming_191"                                           
    lappend design_files "vlog $USER_DEFINED_VERILOG_COMPILE_OPTIONS $USER_DEFINED_COMPILE_OPTIONS  \"[normalize_path "$QSYS_SIMDIR/../altera_jtag_dc_streaming_191/sim/altera_std_synchronizer_nocut.v"]\"  -work altera_jtag_dc_streaming_191"                                       
    lappend design_files "vlog $USER_DEFINED_VERILOG_COMPILE_OPTIONS $USER_DEFINED_COMPILE_OPTIONS  \"[normalize_path "$QSYS_SIMDIR/../altera_jtag_dc_streaming_191/sim/altera_avalon_st_pipeline_base.v"]\"  -work altera_jtag_dc_streaming_191"                                      
    lappend design_files "vlog $USER_DEFINED_VERILOG_COMPILE_OPTIONS $USER_DEFINED_COMPILE_OPTIONS  \"[normalize_path "$QSYS_SIMDIR/../altera_jtag_dc_streaming_191/sim/altera_avalon_st_idle_remover.v"]\"  -work altera_jtag_dc_streaming_191"                                       
    lappend design_files "vlog $USER_DEFINED_VERILOG_COMPILE_OPTIONS $USER_DEFINED_COMPILE_OPTIONS  \"[normalize_path "$QSYS_SIMDIR/../altera_jtag_dc_streaming_191/sim/altera_avalon_st_idle_inserter.v"]\"  -work altera_jtag_dc_streaming_191"                                      
    lappend design_files "vlog -sv $USER_DEFINED_VERILOG_COMPILE_OPTIONS $USER_DEFINED_COMPILE_OPTIONS  \"[normalize_path "$QSYS_SIMDIR/../altera_jtag_dc_streaming_191/sim/altera_avalon_st_pipeline_stage.sv"]\"  -work altera_jtag_dc_streaming_191"                                
    lappend design_files "vlog -sv $USER_DEFINED_VERILOG_COMPILE_OPTIONS $USER_DEFINED_COMPILE_OPTIONS  \"[normalize_path "$QSYS_SIMDIR/../timing_adapter_1940/sim/SDRAM_timing_adapter_1940_5ju4ddy.sv"]\"  -work timing_adapter_1940"                                                
    lappend design_files "vlog $USER_DEFINED_VERILOG_COMPILE_OPTIONS $USER_DEFINED_COMPILE_OPTIONS  \"[normalize_path "$QSYS_SIMDIR/../altera_avalon_sc_fifo_1932/sim/SDRAM_altera_avalon_sc_fifo_1932_5j7ufsq.v"]\"  -work altera_avalon_sc_fifo_1932"                                
    lappend design_files "vlog $USER_DEFINED_VERILOG_COMPILE_OPTIONS $USER_DEFINED_COMPILE_OPTIONS  \"[normalize_path "$QSYS_SIMDIR/../altera_avalon_st_bytes_to_packets_1922/sim/altera_avalon_st_bytes_to_packets.v"]\"  -work altera_avalon_st_bytes_to_packets_1922"               
    lappend design_files "vlog $USER_DEFINED_VERILOG_COMPILE_OPTIONS $USER_DEFINED_COMPILE_OPTIONS  \"[normalize_path "$QSYS_SIMDIR/../altera_avalon_st_packets_to_bytes_1922/sim/altera_avalon_st_packets_to_bytes.v"]\"  -work altera_avalon_st_packets_to_bytes_1922"               
    lappend design_files "vlog $USER_DEFINED_VERILOG_COMPILE_OPTIONS $USER_DEFINED_COMPILE_OPTIONS  \"[normalize_path "$QSYS_SIMDIR/../altera_avalon_packets_to_master_1922/sim/altera_avalon_packets_to_master.v"]\"  -work altera_avalon_packets_to_master_1922"                     
    lappend design_files "vlog -sv $USER_DEFINED_VERILOG_COMPILE_OPTIONS $USER_DEFINED_COMPILE_OPTIONS  \"[normalize_path "$QSYS_SIMDIR/../channel_adapter_1921/sim/SDRAM_channel_adapter_1921_5wnzrci.sv"]\"  -work channel_adapter_1921"                                             
    lappend design_files "vlog -sv $USER_DEFINED_VERILOG_COMPILE_OPTIONS $USER_DEFINED_COMPILE_OPTIONS  \"[normalize_path "$QSYS_SIMDIR/../channel_adapter_1921/sim/SDRAM_channel_adapter_1921_fkajlia.sv"]\"  -work channel_adapter_1921"                                             
    lappend design_files "vlog $USER_DEFINED_VERILOG_COMPILE_OPTIONS $USER_DEFINED_COMPILE_OPTIONS  \"[normalize_path "$QSYS_SIMDIR/../altera_reset_controller_1922/sim/mentor/altera_reset_controller.v"]\"  -work altera_reset_controller_1922"                                      
    lappend design_files "vlog $USER_DEFINED_VERILOG_COMPILE_OPTIONS $USER_DEFINED_COMPILE_OPTIONS  \"[normalize_path "$QSYS_SIMDIR/../altera_reset_controller_1922/sim/mentor/altera_reset_synchronizer.v"]\"  -work altera_reset_controller_1922"                                    
    lappend design_files "vcom $USER_DEFINED_VHDL_COMPILE_OPTIONS $USER_DEFINED_COMPILE_OPTIONS  \"[normalize_path "$QSYS_SIMDIR/../alt_mem_if_jtag_master_191/sim/SDRAM_alt_mem_if_jtag_master_191_rksoe3i.vhd"]\"  -work alt_mem_if_jtag_master_191"                                 
    lappend design_files "vlog -sv $USER_DEFINED_VERILOG_COMPILE_OPTIONS $USER_DEFINED_COMPILE_OPTIONS  \"[normalize_path "$QSYS_SIMDIR/../altera_merlin_master_translator_192/sim/SDRAM_altera_merlin_master_translator_192_lykd4la.sv"]\"  -work altera_merlin_master_translator_192"
    lappend design_files "vlog -sv $USER_DEFINED_VERILOG_COMPILE_OPTIONS $USER_DEFINED_COMPILE_OPTIONS  \"[normalize_path "$QSYS_SIMDIR/../altera_merlin_slave_translator_191/sim/SDRAM_altera_merlin_slave_translator_191_x56fcki.sv"]\"  -work altera_merlin_slave_translator_191"   
    lappend design_files "vcom $USER_DEFINED_VHDL_COMPILE_OPTIONS $USER_DEFINED_COMPILE_OPTIONS  \"[normalize_path "$QSYS_SIMDIR/../altera_mm_interconnect_1920/sim/SDRAM_altera_mm_interconnect_1920_mlnx7ry.vhd"]\"  -work altera_mm_interconnect_1920"                              
    lappend design_files "vcom $USER_DEFINED_VHDL_COMPILE_OPTIONS $USER_DEFINED_COMPILE_OPTIONS  \"[normalize_path "$QSYS_SIMDIR/../altera_ip_col_if_1911/sim/SDRAM_altera_ip_col_if_1911_rqtxiaa.vhd"]\"  -work altera_ip_col_if_1911"                                                
    lappend design_files "vcom $USER_DEFINED_VHDL_COMPILE_OPTIONS $USER_DEFINED_COMPILE_OPTIONS  \"[normalize_path "$QSYS_SIMDIR/../altera_avalon_onchip_memory2_1939/sim/SDRAM_altera_avalon_onchip_memory2_1939_fptqw4y.vhd"]\"  -work altera_avalon_onchip_memory2_1939"            
    lappend design_files "vcom $USER_DEFINED_VHDL_COMPILE_OPTIONS $USER_DEFINED_COMPILE_OPTIONS  \"[normalize_path "$QSYS_SIMDIR/../altera_mm_interconnect_1920/sim/SDRAM_altera_mm_interconnect_1920_ymgjipa.vhd"]\"  -work altera_mm_interconnect_1920"                              
    lappend design_files "vcom $USER_DEFINED_VHDL_COMPILE_OPTIONS $USER_DEFINED_COMPILE_OPTIONS  \"[normalize_path "$QSYS_SIMDIR/../altera_emif_cal_slave_nf_191/sim/SDRAM_altera_emif_cal_slave_nf_191_rmzieji.vhd"]\"  -work altera_emif_cal_slave_nf_191"                           
    lappend design_files "vcom $USER_DEFINED_VHDL_COMPILE_OPTIONS $USER_DEFINED_COMPILE_OPTIONS  \"[normalize_path "$QSYS_SIMDIR/../altera_mm_interconnect_1920/sim/SDRAM_altera_mm_interconnect_1920_mnl7iba.vhd"]\"  -work altera_mm_interconnect_1920"                              
    lappend design_files "vcom $USER_DEFINED_VHDL_COMPILE_OPTIONS $USER_DEFINED_COMPILE_OPTIONS  \"[normalize_path "$QSYS_SIMDIR/../altera_emif_1923/sim/SDRAM_altera_emif_1923_wh6slty.vhd"]\"  -work altera_emif_1923"                                                               
    lappend design_files "vcom $USER_DEFINED_VHDL_COMPILE_OPTIONS $USER_DEFINED_COMPILE_OPTIONS  \"[normalize_path "$QSYS_SIMDIR/SDRAM.vhd"]\"  -work SDRAM"                                                                                                                           
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
    if ![ string match "bit_64" $SIMULATOR_TOOL_BITNESS ] {
    } else {
    }
    append ELAB_OPTIONS { -t fs}
    return $ELAB_OPTIONS
  }
  
  
  proc get_sim_options {SIMULATOR_TOOL_BITNESS} {
    set SIM_OPTIONS ""
    if ![ string match "bit_64" $SIMULATOR_TOOL_BITNESS ] {
    } else {
    }
    return $SIM_OPTIONS
  }
  
  
  proc get_env_variables {SIMULATOR_TOOL_BITNESS} {
    set ENV_VARIABLES [dict create]
    set LD_LIBRARY_PATH [dict create]
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
    
    return $libraries
  }
  
}
