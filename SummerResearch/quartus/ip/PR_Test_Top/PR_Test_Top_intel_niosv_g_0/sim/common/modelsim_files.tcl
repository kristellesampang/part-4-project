
namespace eval PR_Test_Top_intel_niosv_g_0 {
  proc get_design_libraries {} {
    set libraries [dict create]
    dict set libraries altera_reset_controller_1922 1
    dict set libraries intel_niosv_g_unit_220       1
    dict set libraries intel_niosv_timer_msip_120   1
    dict set libraries intel_niosv_dbg_mod_210      1
    dict set libraries altera_irq_mapper_2001       1
    dict set libraries intel_niosv_g_220            1
    dict set libraries PR_Test_Top_intel_niosv_g_0  1
    return $libraries
  }
  
  proc get_memory_files {QSYS_SIMDIR} {
    set memory_files [list]
    lappend memory_files "[normalize_path "$QSYS_SIMDIR/../intel_niosv_g_unit_220/sim/fp32_sqrt_memoryC0_uid62_sqrtTables_lutmem.hex"]"
    lappend memory_files "[normalize_path "$QSYS_SIMDIR/../intel_niosv_g_unit_220/sim/fp32_sqrt_memoryC1_uid65_sqrtTables_lutmem.hex"]"
    lappend memory_files "[normalize_path "$QSYS_SIMDIR/../intel_niosv_g_unit_220/sim/fp32_sqrt_memoryC2_uid68_sqrtTables_lutmem.hex"]"
    lappend memory_files "[normalize_path "$QSYS_SIMDIR/../intel_niosv_dbg_mod_210/sim/csr_mlab.mif"]"
    lappend memory_files "[normalize_path "$QSYS_SIMDIR/../intel_niosv_dbg_mod_210/sim/debug_rom.mif"]"
    return $memory_files
  }
  
  proc get_common_design_files {USER_DEFINED_COMPILE_OPTIONS USER_DEFINED_VERILOG_COMPILE_OPTIONS USER_DEFINED_VHDL_COMPILE_OPTIONS QSYS_SIMDIR} {
    set design_files [dict create]
    return $design_files
  }
  
  proc get_design_files {USER_DEFINED_COMPILE_OPTIONS USER_DEFINED_VERILOG_COMPILE_OPTIONS USER_DEFINED_VHDL_COMPILE_OPTIONS QSYS_SIMDIR} {
    set design_files [list]
    lappend design_files "vlog $USER_DEFINED_VERILOG_COMPILE_OPTIONS $USER_DEFINED_COMPILE_OPTIONS  \"[normalize_path "$QSYS_SIMDIR/../altera_reset_controller_1922/sim/mentor/altera_reset_controller.v"]\"  -work altera_reset_controller_1922"                                          
    lappend design_files "vlog $USER_DEFINED_VERILOG_COMPILE_OPTIONS $USER_DEFINED_COMPILE_OPTIONS  \"[normalize_path "$QSYS_SIMDIR/../altera_reset_controller_1922/sim/mentor/altera_reset_synchronizer.v"]\"  -work altera_reset_controller_1922"                                        
    lappend design_files "vcom $USER_DEFINED_VHDL_COMPILE_OPTIONS $USER_DEFINED_COMPILE_OPTIONS  \"[normalize_path "$QSYS_SIMDIR/../intel_niosv_g_unit_220/sim/niosv_reset_controller.vhd"]\"  -work intel_niosv_g_unit_220"                                                               
    lappend design_files "vlog -sv $USER_DEFINED_VERILOG_COMPILE_OPTIONS $USER_DEFINED_COMPILE_OPTIONS  \"[normalize_path "$QSYS_SIMDIR/../intel_niosv_g_unit_220/sim/mentor/niosv_opcode_def.sv"]\"  -work intel_niosv_g_unit_220"                                                        
    lappend design_files "vlog -sv $USER_DEFINED_VERILOG_COMPILE_OPTIONS $USER_DEFINED_COMPILE_OPTIONS  \"[normalize_path "$QSYS_SIMDIR/../intel_niosv_g_unit_220/sim/mentor/niosv_mem_op_state.sv"]\"  -work intel_niosv_g_unit_220"                                                      
    lappend design_files "vlog -sv $USER_DEFINED_VERILOG_COMPILE_OPTIONS $USER_DEFINED_COMPILE_OPTIONS  \"[normalize_path "$QSYS_SIMDIR/../intel_niosv_g_unit_220/sim/mentor/niosv_ram.sv"]\"  -work intel_niosv_g_unit_220"                                                               
    lappend design_files "vlog -sv $USER_DEFINED_VERILOG_COMPILE_OPTIONS $USER_DEFINED_COMPILE_OPTIONS  \"[normalize_path "$QSYS_SIMDIR/../intel_niosv_g_unit_220/sim/mentor/niosv_altecc_32enc.v"]\"  -work intel_niosv_g_unit_220"                                                       
    lappend design_files "vlog -sv $USER_DEFINED_VERILOG_COMPILE_OPTIONS $USER_DEFINED_COMPILE_OPTIONS  \"[normalize_path "$QSYS_SIMDIR/../intel_niosv_g_unit_220/sim/mentor/niosv_altecc_32dec.v"]\"  -work intel_niosv_g_unit_220"                                                       
    lappend design_files "vlog -sv $USER_DEFINED_VERILOG_COMPILE_OPTIONS $USER_DEFINED_COMPILE_OPTIONS  \"[normalize_path "$QSYS_SIMDIR/../intel_niosv_g_unit_220/sim/mentor/niosv_tcm_ram.sv"]\"  -work intel_niosv_g_unit_220"                                                           
    lappend design_files "vlog -sv $USER_DEFINED_VERILOG_COMPILE_OPTIONS $USER_DEFINED_COMPILE_OPTIONS  \"[normalize_path "$QSYS_SIMDIR/../intel_niosv_g_unit_220/sim/mentor/niosv_reg_file.sv"]\"  -work intel_niosv_g_unit_220"                                                          
    lappend design_files "vlog -sv $USER_DEFINED_VERILOG_COMPILE_OPTIONS $USER_DEFINED_COMPILE_OPTIONS  \"[normalize_path "$QSYS_SIMDIR/../intel_niosv_g_unit_220/sim/mentor/niosv_alu.sv"]\"  -work intel_niosv_g_unit_220"                                                               
    lappend design_files "vlog -sv $USER_DEFINED_VERILOG_COMPILE_OPTIONS $USER_DEFINED_COMPILE_OPTIONS  \"[normalize_path "$QSYS_SIMDIR/../intel_niosv_g_unit_220/sim/mentor/niosv_csr.sv"]\"  -work intel_niosv_g_unit_220"                                                               
    lappend design_files "vlog -sv $USER_DEFINED_VERILOG_COMPILE_OPTIONS $USER_DEFINED_COMPILE_OPTIONS  \"[normalize_path "$QSYS_SIMDIR/../intel_niosv_g_unit_220/sim/mentor/niosv_interrupt_handler.sv"]\"  -work intel_niosv_g_unit_220"                                                 
    lappend design_files "vlog -sv $USER_DEFINED_VERILOG_COMPILE_OPTIONS $USER_DEFINED_COMPILE_OPTIONS  \"[normalize_path "$QSYS_SIMDIR/../intel_niosv_g_unit_220/sim/mentor/niosv_multiplier.sv"]\"  -work intel_niosv_g_unit_220"                                                        
    lappend design_files "vlog -sv $USER_DEFINED_VERILOG_COMPILE_OPTIONS $USER_DEFINED_COMPILE_OPTIONS  \"[normalize_path "$QSYS_SIMDIR/../intel_niosv_g_unit_220/sim/mentor/niosv_divider.sv"]\"  -work intel_niosv_g_unit_220"                                                           
    lappend design_files "vlog -sv $USER_DEFINED_VERILOG_COMPILE_OPTIONS $USER_DEFINED_COMPILE_OPTIONS  \"[normalize_path "$QSYS_SIMDIR/../intel_niosv_g_unit_220/sim/mentor/niosv_instr_buffer.sv"]\"  -work intel_niosv_g_unit_220"                                                      
    lappend design_files "vlog -sv $USER_DEFINED_VERILOG_COMPILE_OPTIONS $USER_DEFINED_COMPILE_OPTIONS  \"[normalize_path "$QSYS_SIMDIR/../intel_niosv_g_unit_220/sim/mentor/niosv_victim_buffer.sv"]\"  -work intel_niosv_g_unit_220"                                                     
    lappend design_files "vlog -sv $USER_DEFINED_VERILOG_COMPILE_OPTIONS $USER_DEFINED_COMPILE_OPTIONS  \"[normalize_path "$QSYS_SIMDIR/../intel_niosv_g_unit_220/sim/mentor/niosv_g_fetch.sv"]\"  -work intel_niosv_g_unit_220"                                                           
    lappend design_files "vlog -sv $USER_DEFINED_VERILOG_COMPILE_OPTIONS $USER_DEFINED_COMPILE_OPTIONS  \"[normalize_path "$QSYS_SIMDIR/../intel_niosv_g_unit_220/sim/mentor/niosv_g_dcache.sv"]\"  -work intel_niosv_g_unit_220"                                                          
    lappend design_files "vlog -sv $USER_DEFINED_VERILOG_COMPILE_OPTIONS $USER_DEFINED_COMPILE_OPTIONS  \"[normalize_path "$QSYS_SIMDIR/../intel_niosv_g_unit_220/sim/mentor/niosv_g_instr_cache.sv"]\"  -work intel_niosv_g_unit_220"                                                     
    lappend design_files "vlog -sv $USER_DEFINED_VERILOG_COMPILE_OPTIONS $USER_DEFINED_COMPILE_OPTIONS  \"[normalize_path "$QSYS_SIMDIR/../intel_niosv_g_unit_220/sim/mentor/niosv_g_fp_def.sv"]\"  -work intel_niosv_g_unit_220"                                                          
    lappend design_files "vlog -sv $USER_DEFINED_VERILOG_COMPILE_OPTIONS $USER_DEFINED_COMPILE_OPTIONS  \"[normalize_path "$QSYS_SIMDIR/../intel_niosv_g_unit_220/sim/mentor/niosv_g_fpu.sv"]\"  -work intel_niosv_g_unit_220"                                                             
    lappend design_files "vlog -sv $USER_DEFINED_VERILOG_COMPILE_OPTIONS $USER_DEFINED_COMPILE_OPTIONS  \"[normalize_path "$QSYS_SIMDIR/../intel_niosv_g_unit_220/sim/mentor/int33_to_fp32.sv"]\"  -work intel_niosv_g_unit_220"                                                           
    lappend design_files "vlog -sv $USER_DEFINED_VERILOG_COMPILE_OPTIONS $USER_DEFINED_COMPILE_OPTIONS  \"[normalize_path "$QSYS_SIMDIR/../intel_niosv_g_unit_220/sim/mentor/fp32_to_int32.sv"]\"  -work intel_niosv_g_unit_220"                                                           
    lappend design_files "vlog -sv $USER_DEFINED_VERILOG_COMPILE_OPTIONS $USER_DEFINED_COMPILE_OPTIONS  \"[normalize_path "$QSYS_SIMDIR/../intel_niosv_g_unit_220/sim/mentor/fp32_to_uint32.sv"]\"  -work intel_niosv_g_unit_220"                                                          
    lappend design_files "vlog -sv $USER_DEFINED_VERILOG_COMPILE_OPTIONS $USER_DEFINED_COMPILE_OPTIONS  \"[normalize_path "$QSYS_SIMDIR/../intel_niosv_g_unit_220/sim/mentor/fp32_alu_cycloneivgx.sv"]\"  -work intel_niosv_g_unit_220"                                                    
    lappend design_files "vlog -sv $USER_DEFINED_VERILOG_COMPILE_OPTIONS $USER_DEFINED_COMPILE_OPTIONS  \"[normalize_path "$QSYS_SIMDIR/../intel_niosv_g_unit_220/sim/mentor/fp32_mult_cycloneivgx.sv"]\"  -work intel_niosv_g_unit_220"                                                   
    lappend design_files "vlog -sv $USER_DEFINED_VERILOG_COMPILE_OPTIONS $USER_DEFINED_COMPILE_OPTIONS  \"[normalize_path "$QSYS_SIMDIR/../intel_niosv_g_unit_220/sim/mentor/fp32_div.sv"]\"  -work intel_niosv_g_unit_220"                                                                
    lappend design_files "vlog -sv $USER_DEFINED_VERILOG_COMPILE_OPTIONS $USER_DEFINED_COMPILE_OPTIONS  \"[normalize_path "$QSYS_SIMDIR/../intel_niosv_g_unit_220/sim/mentor/fp32_sqrt.sv"]\"  -work intel_niosv_g_unit_220"                                                               
    lappend design_files "vlog -sv $USER_DEFINED_VERILOG_COMPILE_OPTIONS $USER_DEFINED_COMPILE_OPTIONS  \"[normalize_path "$QSYS_SIMDIR/../intel_niosv_g_unit_220/sim/mentor/niosv_g_dspba_library_ver.sv"]\"  -work intel_niosv_g_unit_220"                                               
    lappend design_files "vlog -sv $USER_DEFINED_VERILOG_COMPILE_OPTIONS $USER_DEFINED_COMPILE_OPTIONS  \"[normalize_path "$QSYS_SIMDIR/../intel_niosv_g_unit_220/sim/mentor/instr_decoder_PR_Test_Top_intel_niosv_g_0_intel_niosv_g_unit_220_2s7cy6a.sv"]\"  -work intel_niosv_g_unit_220"
    lappend design_files "vlog -sv $USER_DEFINED_VERILOG_COMPILE_OPTIONS $USER_DEFINED_COMPILE_OPTIONS  \"[normalize_path "$QSYS_SIMDIR/../intel_niosv_g_unit_220/sim/mentor/niosv_g_core_PR_Test_Top_intel_niosv_g_0_intel_niosv_g_unit_220_2s7cy6a.sv"]\"  -work intel_niosv_g_unit_220" 
    lappend design_files "vlog -sv $USER_DEFINED_VERILOG_COMPILE_OPTIONS $USER_DEFINED_COMPILE_OPTIONS  \"[normalize_path "$QSYS_SIMDIR/../intel_niosv_g_unit_220/sim/mentor/PR_Test_Top_intel_niosv_g_0_intel_niosv_g_unit_220_2s7cy6a.sv"]\"  -work intel_niosv_g_unit_220"              
    lappend design_files "vlog -sv $USER_DEFINED_VERILOG_COMPILE_OPTIONS $USER_DEFINED_COMPILE_OPTIONS  \"[normalize_path "$QSYS_SIMDIR/../intel_niosv_g_unit_220/sim/altera_avalon_sc_fifo.v"]\"  -work intel_niosv_g_unit_220"                                                           
    lappend design_files "vlog -sv $USER_DEFINED_VERILOG_COMPILE_OPTIONS $USER_DEFINED_COMPILE_OPTIONS  \"[normalize_path "$QSYS_SIMDIR/../intel_niosv_timer_msip_120/sim/mentor/niosv_timer_msip.sv"]\"  -work intel_niosv_timer_msip_120"                                                
    lappend design_files "vlog -sv $USER_DEFINED_VERILOG_COMPILE_OPTIONS $USER_DEFINED_COMPILE_OPTIONS  \"[normalize_path "$QSYS_SIMDIR/../intel_niosv_dbg_mod_210/sim/mentor/niosv_dm_def.sv"]\"  -work intel_niosv_dbg_mod_210"                                                          
    lappend design_files "vlog -sv $USER_DEFINED_VERILOG_COMPILE_OPTIONS $USER_DEFINED_COMPILE_OPTIONS  \"[normalize_path "$QSYS_SIMDIR/../intel_niosv_dbg_mod_210/sim/mentor/niosv_ram.sv"]\"  -work intel_niosv_dbg_mod_210"                                                             
    lappend design_files "vlog -sv $USER_DEFINED_VERILOG_COMPILE_OPTIONS $USER_DEFINED_COMPILE_OPTIONS  \"[normalize_path "$QSYS_SIMDIR/../intel_niosv_dbg_mod_210/sim/mentor/niosv_dm_jtag2mm.sv"]\"  -work intel_niosv_dbg_mod_210"                                                      
    lappend design_files "vlog -sv $USER_DEFINED_VERILOG_COMPILE_OPTIONS $USER_DEFINED_COMPILE_OPTIONS  \"[normalize_path "$QSYS_SIMDIR/../intel_niosv_dbg_mod_210/sim/mentor/niosv_dm_top.sv"]\"  -work intel_niosv_dbg_mod_210"                                                          
    lappend design_files "vlog -sv $USER_DEFINED_VERILOG_COMPILE_OPTIONS $USER_DEFINED_COMPILE_OPTIONS  \"[normalize_path "$QSYS_SIMDIR/../intel_niosv_dbg_mod_210/sim/mentor/niosv_debug_module.sv"]\"  -work intel_niosv_dbg_mod_210"                                                    
    lappend design_files "vlog -sv $USER_DEFINED_VERILOG_COMPILE_OPTIONS $USER_DEFINED_COMPILE_OPTIONS  \"[normalize_path "$QSYS_SIMDIR/../intel_niosv_dbg_mod_210/sim/altera_std_synchronizer_bundle.v"]\"  -work intel_niosv_dbg_mod_210"                                                
    lappend design_files "vlog -sv $USER_DEFINED_VERILOG_COMPILE_OPTIONS $USER_DEFINED_COMPILE_OPTIONS  \"[normalize_path "$QSYS_SIMDIR/../intel_niosv_dbg_mod_210/sim/altera_std_synchronizer_nocut.v"]\"  -work intel_niosv_dbg_mod_210"                                                 
    lappend design_files "vlog -sv $USER_DEFINED_VERILOG_COMPILE_OPTIONS $USER_DEFINED_COMPILE_OPTIONS  \"[normalize_path "$QSYS_SIMDIR/../intel_niosv_dbg_mod_210/sim/altera_std_synchronizer.v"]\"  -work intel_niosv_dbg_mod_210"                                                       
    lappend design_files "vlog -sv $USER_DEFINED_VERILOG_COMPILE_OPTIONS $USER_DEFINED_COMPILE_OPTIONS  \"[normalize_path "$QSYS_SIMDIR/../intel_niosv_dbg_mod_210/sim/altera_avalon_st_clock_crosser.v"]\"  -work intel_niosv_dbg_mod_210"                                                
    lappend design_files "vlog -sv $USER_DEFINED_VERILOG_COMPILE_OPTIONS $USER_DEFINED_COMPILE_OPTIONS  \"[normalize_path "$QSYS_SIMDIR/../intel_niosv_dbg_mod_210/sim/altera_avalon_st_handshake_clock_crosser.v"]\"  -work intel_niosv_dbg_mod_210"                                      
    lappend design_files "vlog -sv $USER_DEFINED_VERILOG_COMPILE_OPTIONS $USER_DEFINED_COMPILE_OPTIONS  \"[normalize_path "$QSYS_SIMDIR/../intel_niosv_dbg_mod_210/sim/altera_avalon_st_pipeline_base.v"]\"  -work intel_niosv_dbg_mod_210"                                                
    lappend design_files "vlog -sv $USER_DEFINED_VERILOG_COMPILE_OPTIONS $USER_DEFINED_COMPILE_OPTIONS  \"[normalize_path "$QSYS_SIMDIR/../intel_niosv_dbg_mod_210/sim/altera_avalon_st_pipeline_stage.sv"]\"  -work intel_niosv_dbg_mod_210"                                              
    lappend design_files "vlog -sv $USER_DEFINED_VERILOG_COMPILE_OPTIONS $USER_DEFINED_COMPILE_OPTIONS  \"[normalize_path "$QSYS_SIMDIR/../intel_niosv_dbg_mod_210/sim/altera_reset_synchronizer.v"]\"  -work intel_niosv_dbg_mod_210"                                                     
    lappend design_files "vlog -sv $USER_DEFINED_VERILOG_COMPILE_OPTIONS $USER_DEFINED_COMPILE_OPTIONS  \"[normalize_path "$QSYS_SIMDIR/../intel_niosv_dbg_mod_210/sim/altera_reset_controller.v"]\"  -work intel_niosv_dbg_mod_210"                                                       
    lappend design_files "vlog -sv $USER_DEFINED_VERILOG_COMPILE_OPTIONS $USER_DEFINED_COMPILE_OPTIONS  \"[normalize_path "$QSYS_SIMDIR/../altera_irq_mapper_2001/sim/PR_Test_Top_intel_niosv_g_0_altera_irq_mapper_2001_3jqx4ly.sv"]\"  -work altera_irq_mapper_2001"                     
    lappend design_files "vcom $USER_DEFINED_VHDL_COMPILE_OPTIONS $USER_DEFINED_COMPILE_OPTIONS  \"[normalize_path "$QSYS_SIMDIR/../intel_niosv_g_220/sim/PR_Test_Top_intel_niosv_g_0_intel_niosv_g_220_n637f5y.vhd"]\"  -work intel_niosv_g_220"                                          
    lappend design_files "vcom $USER_DEFINED_VHDL_COMPILE_OPTIONS $USER_DEFINED_COMPILE_OPTIONS  \"[normalize_path "$QSYS_SIMDIR/PR_Test_Top_intel_niosv_g_0.vhd"]\"  -work PR_Test_Top_intel_niosv_g_0"                                                                                   
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
