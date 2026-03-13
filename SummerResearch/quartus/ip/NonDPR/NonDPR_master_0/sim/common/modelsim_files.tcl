
namespace eval NonDPR_master_0 {
  proc get_design_libraries {} {
    set libraries [dict create]
    dict set libraries altera_jtag_dc_streaming_191           1
    dict set libraries timing_adapter_1940                    1
    dict set libraries altera_avalon_sc_fifo_1932             1
    dict set libraries altera_avalon_st_bytes_to_packets_1922 1
    dict set libraries altera_avalon_st_packets_to_bytes_1922 1
    dict set libraries altera_avalon_packets_to_master_1922   1
    dict set libraries channel_adapter_1921                   1
    dict set libraries altera_reset_controller_1922           1
    dict set libraries altera_jtag_avalon_master_191          1
    dict set libraries NonDPR_master_0                        1
    return $libraries
  }
  
  proc get_memory_files {QSYS_SIMDIR} {
    set memory_files [list]
    return $memory_files
  }
  
  proc get_common_design_files {USER_DEFINED_COMPILE_OPTIONS USER_DEFINED_VERILOG_COMPILE_OPTIONS USER_DEFINED_VHDL_COMPILE_OPTIONS QSYS_SIMDIR} {
    set design_files [dict create]
    return $design_files
  }
  
  proc get_design_files {USER_DEFINED_COMPILE_OPTIONS USER_DEFINED_VERILOG_COMPILE_OPTIONS USER_DEFINED_VHDL_COMPILE_OPTIONS QSYS_SIMDIR} {
    set design_files [list]
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
    lappend design_files "vlog -sv $USER_DEFINED_VERILOG_COMPILE_OPTIONS $USER_DEFINED_COMPILE_OPTIONS  \"[normalize_path "$QSYS_SIMDIR/../timing_adapter_1940/sim/NonDPR_master_0_timing_adapter_1940_5ju4ddy.sv"]\"  -work timing_adapter_1940"                        
    lappend design_files "vlog $USER_DEFINED_VERILOG_COMPILE_OPTIONS $USER_DEFINED_COMPILE_OPTIONS  \"[normalize_path "$QSYS_SIMDIR/../altera_avalon_sc_fifo_1932/sim/NonDPR_master_0_altera_avalon_sc_fifo_1932_5j7ufsq.v"]\"  -work altera_avalon_sc_fifo_1932"        
    lappend design_files "vlog $USER_DEFINED_VERILOG_COMPILE_OPTIONS $USER_DEFINED_COMPILE_OPTIONS  \"[normalize_path "$QSYS_SIMDIR/../altera_avalon_st_bytes_to_packets_1922/sim/altera_avalon_st_bytes_to_packets.v"]\"  -work altera_avalon_st_bytes_to_packets_1922" 
    lappend design_files "vlog $USER_DEFINED_VERILOG_COMPILE_OPTIONS $USER_DEFINED_COMPILE_OPTIONS  \"[normalize_path "$QSYS_SIMDIR/../altera_avalon_st_packets_to_bytes_1922/sim/altera_avalon_st_packets_to_bytes.v"]\"  -work altera_avalon_st_packets_to_bytes_1922" 
    lappend design_files "vlog $USER_DEFINED_VERILOG_COMPILE_OPTIONS $USER_DEFINED_COMPILE_OPTIONS  \"[normalize_path "$QSYS_SIMDIR/../altera_avalon_packets_to_master_1922/sim/altera_avalon_packets_to_master.v"]\"  -work altera_avalon_packets_to_master_1922"       
    lappend design_files "vlog -sv $USER_DEFINED_VERILOG_COMPILE_OPTIONS $USER_DEFINED_COMPILE_OPTIONS  \"[normalize_path "$QSYS_SIMDIR/../channel_adapter_1921/sim/NonDPR_master_0_channel_adapter_1921_5wnzrci.sv"]\"  -work channel_adapter_1921"                     
    lappend design_files "vlog -sv $USER_DEFINED_VERILOG_COMPILE_OPTIONS $USER_DEFINED_COMPILE_OPTIONS  \"[normalize_path "$QSYS_SIMDIR/../channel_adapter_1921/sim/NonDPR_master_0_channel_adapter_1921_fkajlia.sv"]\"  -work channel_adapter_1921"                     
    lappend design_files "vlog $USER_DEFINED_VERILOG_COMPILE_OPTIONS $USER_DEFINED_COMPILE_OPTIONS  \"[normalize_path "$QSYS_SIMDIR/../altera_reset_controller_1922/sim/mentor/altera_reset_controller.v"]\"  -work altera_reset_controller_1922"                        
    lappend design_files "vlog $USER_DEFINED_VERILOG_COMPILE_OPTIONS $USER_DEFINED_COMPILE_OPTIONS  \"[normalize_path "$QSYS_SIMDIR/../altera_reset_controller_1922/sim/mentor/altera_reset_synchronizer.v"]\"  -work altera_reset_controller_1922"                      
    lappend design_files "vcom $USER_DEFINED_VHDL_COMPILE_OPTIONS $USER_DEFINED_COMPILE_OPTIONS  \"[normalize_path "$QSYS_SIMDIR/../altera_jtag_avalon_master_191/sim/NonDPR_master_0_altera_jtag_avalon_master_191_3zppvky.vhd"]\"  -work altera_jtag_avalon_master_191"
    lappend design_files "vcom $USER_DEFINED_VHDL_COMPILE_OPTIONS $USER_DEFINED_COMPILE_OPTIONS  \"[normalize_path "$QSYS_SIMDIR/NonDPR_master_0.vhd"]\"  -work NonDPR_master_0"                                                                                         
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
