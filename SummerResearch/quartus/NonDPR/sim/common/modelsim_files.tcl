source [file join [file dirname [info script]] ./../../../ip/NonDPR/NonDPR_master_0/sim/common/modelsim_files.tcl]
source [file join [file dirname [info script]] ./../../../ip/NonDPR/weight_mem/sim/common/modelsim_files.tcl]
source [file join [file dirname [info script]] ./../../../ip/NonDPR/data_mem/sim/common/modelsim_files.tcl]
source [file join [file dirname [info script]] ./../../../ip/NonDPR/out_mem/sim/common/modelsim_files.tcl]
source [file join [file dirname [info script]] ./../../../ip/PR_Test_Top/PR_Test_Top_reset_in/sim/common/modelsim_files.tcl]
source [file join [file dirname [info script]] ./../../../ip/NonDPR/NonDPR_npu_system_28/sim/common/modelsim_files.tcl]

namespace eval NonDPR {
  proc get_design_libraries {} {
    set libraries [dict create]
    set libraries [dict merge $libraries [NonDPR_master_0::get_design_libraries]]
    set libraries [dict merge $libraries [weight_mem::get_design_libraries]]
    set libraries [dict merge $libraries [data_mem::get_design_libraries]]
    set libraries [dict merge $libraries [out_mem::get_design_libraries]]
    set libraries [dict merge $libraries [PR_Test_Top_reset_in::get_design_libraries]]
    set libraries [dict merge $libraries [NonDPR_npu_system_28::get_design_libraries]]
    dict set libraries altera_merlin_master_translator_192 1
    dict set libraries altera_merlin_slave_translator_191  1
    dict set libraries altera_mm_interconnect_1920         1
    dict set libraries altera_merlin_master_agent_1922     1
    dict set libraries altera_merlin_slave_agent_1921      1
    dict set libraries altera_avalon_sc_fifo_1932          1
    dict set libraries altera_merlin_router_1921           1
    dict set libraries altera_merlin_traffic_limiter_1921  1
    dict set libraries altera_merlin_demultiplexer_1921    1
    dict set libraries altera_merlin_multiplexer_1922      1
    dict set libraries altera_reset_controller_1922        1
    dict set libraries NonDPR                              1
    return $libraries
  }
  
  proc get_memory_files {QSYS_SIMDIR} {
    set memory_files [list]
    set memory_files [concat $memory_files [NonDPR_master_0::get_memory_files "$QSYS_SIMDIR/../../ip/NonDPR/NonDPR_master_0/sim/"]]
    set memory_files [concat $memory_files [weight_mem::get_memory_files "$QSYS_SIMDIR/../../ip/NonDPR/weight_mem/sim/"]]
    set memory_files [concat $memory_files [data_mem::get_memory_files "$QSYS_SIMDIR/../../ip/NonDPR/data_mem/sim/"]]
    set memory_files [concat $memory_files [out_mem::get_memory_files "$QSYS_SIMDIR/../../ip/NonDPR/out_mem/sim/"]]
    set memory_files [concat $memory_files [PR_Test_Top_reset_in::get_memory_files "$QSYS_SIMDIR/../../ip/PR_Test_Top/PR_Test_Top_reset_in/sim/"]]
    set memory_files [concat $memory_files [NonDPR_npu_system_28::get_memory_files "$QSYS_SIMDIR/../../ip/NonDPR/NonDPR_npu_system_28/sim/"]]
    return $memory_files
  }
  
  proc get_common_design_files {USER_DEFINED_COMPILE_OPTIONS USER_DEFINED_VERILOG_COMPILE_OPTIONS USER_DEFINED_VHDL_COMPILE_OPTIONS QSYS_SIMDIR} {
    set design_files [dict create]
    set design_files [dict merge $design_files [NonDPR_master_0::get_common_design_files $USER_DEFINED_COMPILE_OPTIONS $USER_DEFINED_VERILOG_COMPILE_OPTIONS $USER_DEFINED_VHDL_COMPILE_OPTIONS "$QSYS_SIMDIR/../../ip/NonDPR/NonDPR_master_0/sim/"]]
    set design_files [dict merge $design_files [weight_mem::get_common_design_files $USER_DEFINED_COMPILE_OPTIONS $USER_DEFINED_VERILOG_COMPILE_OPTIONS $USER_DEFINED_VHDL_COMPILE_OPTIONS "$QSYS_SIMDIR/../../ip/NonDPR/weight_mem/sim/"]]
    set design_files [dict merge $design_files [data_mem::get_common_design_files $USER_DEFINED_COMPILE_OPTIONS $USER_DEFINED_VERILOG_COMPILE_OPTIONS $USER_DEFINED_VHDL_COMPILE_OPTIONS "$QSYS_SIMDIR/../../ip/NonDPR/data_mem/sim/"]]
    set design_files [dict merge $design_files [out_mem::get_common_design_files $USER_DEFINED_COMPILE_OPTIONS $USER_DEFINED_VERILOG_COMPILE_OPTIONS $USER_DEFINED_VHDL_COMPILE_OPTIONS "$QSYS_SIMDIR/../../ip/NonDPR/out_mem/sim/"]]
    set design_files [dict merge $design_files [PR_Test_Top_reset_in::get_common_design_files $USER_DEFINED_COMPILE_OPTIONS $USER_DEFINED_VERILOG_COMPILE_OPTIONS $USER_DEFINED_VHDL_COMPILE_OPTIONS "$QSYS_SIMDIR/../../ip/PR_Test_Top/PR_Test_Top_reset_in/sim/"]]
    set design_files [dict merge $design_files [NonDPR_npu_system_28::get_common_design_files $USER_DEFINED_COMPILE_OPTIONS $USER_DEFINED_VERILOG_COMPILE_OPTIONS $USER_DEFINED_VHDL_COMPILE_OPTIONS "$QSYS_SIMDIR/../../ip/NonDPR/NonDPR_npu_system_28/sim/"]]
    return $design_files
  }
  
  proc get_design_files {USER_DEFINED_COMPILE_OPTIONS USER_DEFINED_VERILOG_COMPILE_OPTIONS USER_DEFINED_VHDL_COMPILE_OPTIONS QSYS_SIMDIR} {
    set design_files [list]
    set design_files [concat $design_files [NonDPR_master_0::get_design_files $USER_DEFINED_COMPILE_OPTIONS $USER_DEFINED_VERILOG_COMPILE_OPTIONS $USER_DEFINED_VHDL_COMPILE_OPTIONS "$QSYS_SIMDIR/../../ip/NonDPR/NonDPR_master_0/sim/"]]
    set design_files [concat $design_files [weight_mem::get_design_files $USER_DEFINED_COMPILE_OPTIONS $USER_DEFINED_VERILOG_COMPILE_OPTIONS $USER_DEFINED_VHDL_COMPILE_OPTIONS "$QSYS_SIMDIR/../../ip/NonDPR/weight_mem/sim/"]]
    set design_files [concat $design_files [data_mem::get_design_files $USER_DEFINED_COMPILE_OPTIONS $USER_DEFINED_VERILOG_COMPILE_OPTIONS $USER_DEFINED_VHDL_COMPILE_OPTIONS "$QSYS_SIMDIR/../../ip/NonDPR/data_mem/sim/"]]
    set design_files [concat $design_files [out_mem::get_design_files $USER_DEFINED_COMPILE_OPTIONS $USER_DEFINED_VERILOG_COMPILE_OPTIONS $USER_DEFINED_VHDL_COMPILE_OPTIONS "$QSYS_SIMDIR/../../ip/NonDPR/out_mem/sim/"]]
    set design_files [concat $design_files [PR_Test_Top_reset_in::get_design_files $USER_DEFINED_COMPILE_OPTIONS $USER_DEFINED_VERILOG_COMPILE_OPTIONS $USER_DEFINED_VHDL_COMPILE_OPTIONS "$QSYS_SIMDIR/../../ip/PR_Test_Top/PR_Test_Top_reset_in/sim/"]]
    set design_files [concat $design_files [NonDPR_npu_system_28::get_design_files $USER_DEFINED_COMPILE_OPTIONS $USER_DEFINED_VERILOG_COMPILE_OPTIONS $USER_DEFINED_VHDL_COMPILE_OPTIONS "$QSYS_SIMDIR/../../ip/NonDPR/NonDPR_npu_system_28/sim/"]]
    lappend design_files "vlog -sv $USER_DEFINED_VERILOG_COMPILE_OPTIONS $USER_DEFINED_COMPILE_OPTIONS  \"[normalize_path "$QSYS_SIMDIR/../altera_merlin_master_translator_192/sim/NonDPR_altera_merlin_master_translator_192_lykd4la.sv"]\"  -work altera_merlin_master_translator_192"             
    lappend design_files "vlog -sv $USER_DEFINED_VERILOG_COMPILE_OPTIONS $USER_DEFINED_COMPILE_OPTIONS  \"[normalize_path "$QSYS_SIMDIR/../altera_merlin_slave_translator_191/sim/NonDPR_altera_merlin_slave_translator_191_x56fcki.sv"]\"  -work altera_merlin_slave_translator_191"                
    lappend design_files "vcom $USER_DEFINED_VHDL_COMPILE_OPTIONS $USER_DEFINED_COMPILE_OPTIONS  \"[normalize_path "$QSYS_SIMDIR/../altera_mm_interconnect_1920/sim/NonDPR_altera_mm_interconnect_1920_y3bfzfy.vhd"]\"  -work altera_mm_interconnect_1920"                                           
    lappend design_files "vlog -sv $USER_DEFINED_VERILOG_COMPILE_OPTIONS $USER_DEFINED_COMPILE_OPTIONS  \"[normalize_path "$QSYS_SIMDIR/../altera_merlin_master_agent_1922/sim/NonDPR_altera_merlin_master_agent_1922_fy3n5ti.sv"]\"  -work altera_merlin_master_agent_1922"                         
    lappend design_files "vlog -sv $USER_DEFINED_VERILOG_COMPILE_OPTIONS $USER_DEFINED_COMPILE_OPTIONS  \"[normalize_path "$QSYS_SIMDIR/../altera_merlin_slave_agent_1921/sim/NonDPR_altera_merlin_slave_agent_1921_b6r3djy.sv"]\"  -work altera_merlin_slave_agent_1921"                            
    lappend design_files "vlog -sv $USER_DEFINED_VERILOG_COMPILE_OPTIONS $USER_DEFINED_COMPILE_OPTIONS  \"[normalize_path "$QSYS_SIMDIR/../altera_merlin_slave_agent_1921/sim/altera_merlin_burst_uncompressor.sv"]\"  -work altera_merlin_slave_agent_1921"                                         
    lappend design_files "vlog $USER_DEFINED_VERILOG_COMPILE_OPTIONS $USER_DEFINED_COMPILE_OPTIONS  \"[normalize_path "$QSYS_SIMDIR/../altera_avalon_sc_fifo_1932/sim/NonDPR_altera_avalon_sc_fifo_1932_5j7ufsq.v"]\"  -work altera_avalon_sc_fifo_1932"                                             
    lappend design_files "vlog -sv $USER_DEFINED_VERILOG_COMPILE_OPTIONS $USER_DEFINED_COMPILE_OPTIONS  \"[normalize_path "$QSYS_SIMDIR/../altera_merlin_router_1921/sim/NonDPR_altera_merlin_router_1921_slix4ri.sv"]\"  -work altera_merlin_router_1921"                                           
    lappend design_files "vlog -sv $USER_DEFINED_VERILOG_COMPILE_OPTIONS $USER_DEFINED_COMPILE_OPTIONS  \"[normalize_path "$QSYS_SIMDIR/../altera_merlin_router_1921/sim/NonDPR_altera_merlin_router_1921_ykxdoni.sv"]\"  -work altera_merlin_router_1921"                                           
    lappend design_files "vcom $USER_DEFINED_VHDL_COMPILE_OPTIONS $USER_DEFINED_COMPILE_OPTIONS  \"[normalize_path "$QSYS_SIMDIR/../altera_merlin_traffic_limiter_1921/sim/NonDPR_altera_merlin_traffic_limiter_altera_avalon_sc_fifo_1921_2nqdnlq.vhd"]\"  -work altera_merlin_traffic_limiter_1921"
    lappend design_files "vlog -sv $USER_DEFINED_VERILOG_COMPILE_OPTIONS $USER_DEFINED_COMPILE_OPTIONS  \"[normalize_path "$QSYS_SIMDIR/../altera_merlin_traffic_limiter_1921/sim/altera_merlin_reorder_memory.sv"]\"  -work altera_merlin_traffic_limiter_1921"                                     
    lappend design_files "vlog -sv $USER_DEFINED_VERILOG_COMPILE_OPTIONS $USER_DEFINED_COMPILE_OPTIONS  \"[normalize_path "$QSYS_SIMDIR/../altera_merlin_traffic_limiter_1921/sim/altera_avalon_st_pipeline_base.v"]\"  -work altera_merlin_traffic_limiter_1921"                                    
    lappend design_files "vlog -sv $USER_DEFINED_VERILOG_COMPILE_OPTIONS $USER_DEFINED_COMPILE_OPTIONS  \"[normalize_path "$QSYS_SIMDIR/../altera_merlin_traffic_limiter_1921/sim/NonDPR_altera_merlin_traffic_limiter_1921_76whh4i.sv"]\"  -work altera_merlin_traffic_limiter_1921"                
    lappend design_files "vlog -sv $USER_DEFINED_VERILOG_COMPILE_OPTIONS $USER_DEFINED_COMPILE_OPTIONS  \"[normalize_path "$QSYS_SIMDIR/../altera_merlin_demultiplexer_1921/sim/NonDPR_altera_merlin_demultiplexer_1921_supv3ca.sv"]\"  -work altera_merlin_demultiplexer_1921"                      
    lappend design_files "vlog -sv $USER_DEFINED_VERILOG_COMPILE_OPTIONS $USER_DEFINED_COMPILE_OPTIONS  \"[normalize_path "$QSYS_SIMDIR/../altera_merlin_multiplexer_1922/sim/NonDPR_altera_merlin_multiplexer_1922_xt75xka.sv"]\"  -work altera_merlin_multiplexer_1922"                            
    lappend design_files "vlog -sv $USER_DEFINED_VERILOG_COMPILE_OPTIONS $USER_DEFINED_COMPILE_OPTIONS  \"[normalize_path "$QSYS_SIMDIR/../altera_merlin_multiplexer_1922/sim/altera_merlin_arbitrator.sv"]\"  -work altera_merlin_multiplexer_1922"                                                 
    lappend design_files "vlog -sv $USER_DEFINED_VERILOG_COMPILE_OPTIONS $USER_DEFINED_COMPILE_OPTIONS  \"[normalize_path "$QSYS_SIMDIR/../altera_merlin_demultiplexer_1921/sim/NonDPR_altera_merlin_demultiplexer_1921_gmzaysy.sv"]\"  -work altera_merlin_demultiplexer_1921"                      
    lappend design_files "vlog -sv $USER_DEFINED_VERILOG_COMPILE_OPTIONS $USER_DEFINED_COMPILE_OPTIONS  \"[normalize_path "$QSYS_SIMDIR/../altera_merlin_multiplexer_1922/sim/NonDPR_altera_merlin_multiplexer_1922_4qfwila.sv"]\"  -work altera_merlin_multiplexer_1922"                            
    lappend design_files "vlog -sv $USER_DEFINED_VERILOG_COMPILE_OPTIONS $USER_DEFINED_COMPILE_OPTIONS  \"[normalize_path "$QSYS_SIMDIR/../altera_merlin_multiplexer_1922/sim/altera_merlin_arbitrator.sv"]\"  -work altera_merlin_multiplexer_1922"                                                 
    lappend design_files "vcom $USER_DEFINED_VHDL_COMPILE_OPTIONS $USER_DEFINED_COMPILE_OPTIONS  \"[normalize_path "$QSYS_SIMDIR/../altera_mm_interconnect_1920/sim/nondpr_altera_mm_interconnect_1920_efcorai_npu_system_0_avalon_slave_0_translator.vhd"]\"  -work altera_mm_interconnect_1920"    
    lappend design_files "vcom $USER_DEFINED_VHDL_COMPILE_OPTIONS $USER_DEFINED_COMPILE_OPTIONS  \"[normalize_path "$QSYS_SIMDIR/../altera_mm_interconnect_1920/sim/nondpr_altera_mm_interconnect_1920_efcorai_data_mem_s1_translator.vhd"]\"  -work altera_mm_interconnect_1920"                    
    lappend design_files "vcom $USER_DEFINED_VHDL_COMPILE_OPTIONS $USER_DEFINED_COMPILE_OPTIONS  \"[normalize_path "$QSYS_SIMDIR/../altera_mm_interconnect_1920/sim/NonDPR_altera_mm_interconnect_1920_efcorai.vhd"]\"  -work altera_mm_interconnect_1920"                                           
    lappend design_files "vcom $USER_DEFINED_VHDL_COMPILE_OPTIONS $USER_DEFINED_COMPILE_OPTIONS  \"[normalize_path "$QSYS_SIMDIR/../altera_mm_interconnect_1920/sim/NonDPR_altera_mm_interconnect_1920_ak3ihbq.vhd"]\"  -work altera_mm_interconnect_1920"                                           
    lappend design_files "vcom $USER_DEFINED_VHDL_COMPILE_OPTIONS $USER_DEFINED_COMPILE_OPTIONS  \"[normalize_path "$QSYS_SIMDIR/../altera_mm_interconnect_1920/sim/NonDPR_altera_mm_interconnect_1920_iwcms7y.vhd"]\"  -work altera_mm_interconnect_1920"                                           
    lappend design_files "vlog $USER_DEFINED_VERILOG_COMPILE_OPTIONS $USER_DEFINED_COMPILE_OPTIONS  \"[normalize_path "$QSYS_SIMDIR/../altera_reset_controller_1922/sim/mentor/altera_reset_controller.v"]\"  -work altera_reset_controller_1922"                                                    
    lappend design_files "vlog $USER_DEFINED_VERILOG_COMPILE_OPTIONS $USER_DEFINED_COMPILE_OPTIONS  \"[normalize_path "$QSYS_SIMDIR/../altera_reset_controller_1922/sim/mentor/altera_reset_synchronizer.v"]\"  -work altera_reset_controller_1922"                                                  
    lappend design_files "vcom $USER_DEFINED_VHDL_COMPILE_OPTIONS $USER_DEFINED_COMPILE_OPTIONS  \"[normalize_path "$QSYS_SIMDIR/NonDPR.vhd"]\"  -work NonDPR"                                                                                                                                       
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
    append ELAB_OPTIONS [get_non_duplicate_elab_option $ELAB_OPTIONS [NonDPR_master_0::get_elab_options $SIMULATOR_TOOL_BITNESS]]
    append ELAB_OPTIONS [get_non_duplicate_elab_option $ELAB_OPTIONS [weight_mem::get_elab_options $SIMULATOR_TOOL_BITNESS]]
    append ELAB_OPTIONS [get_non_duplicate_elab_option $ELAB_OPTIONS [data_mem::get_elab_options $SIMULATOR_TOOL_BITNESS]]
    append ELAB_OPTIONS [get_non_duplicate_elab_option $ELAB_OPTIONS [out_mem::get_elab_options $SIMULATOR_TOOL_BITNESS]]
    append ELAB_OPTIONS [get_non_duplicate_elab_option $ELAB_OPTIONS [PR_Test_Top_reset_in::get_elab_options $SIMULATOR_TOOL_BITNESS]]
    append ELAB_OPTIONS [get_non_duplicate_elab_option $ELAB_OPTIONS [NonDPR_npu_system_28::get_elab_options $SIMULATOR_TOOL_BITNESS]]
    if ![ string match "bit_64" $SIMULATOR_TOOL_BITNESS ] {
    } else {
    }
    append ELAB_OPTIONS { -t fs}
    return $ELAB_OPTIONS
  }
  
  
  proc get_sim_options {SIMULATOR_TOOL_BITNESS} {
    set SIM_OPTIONS ""
    append SIM_OPTIONS [NonDPR_master_0::get_sim_options $SIMULATOR_TOOL_BITNESS]
    append SIM_OPTIONS [weight_mem::get_sim_options $SIMULATOR_TOOL_BITNESS]
    append SIM_OPTIONS [data_mem::get_sim_options $SIMULATOR_TOOL_BITNESS]
    append SIM_OPTIONS [out_mem::get_sim_options $SIMULATOR_TOOL_BITNESS]
    append SIM_OPTIONS [PR_Test_Top_reset_in::get_sim_options $SIMULATOR_TOOL_BITNESS]
    append SIM_OPTIONS [NonDPR_npu_system_28::get_sim_options $SIMULATOR_TOOL_BITNESS]
    if ![ string match "bit_64" $SIMULATOR_TOOL_BITNESS ] {
    } else {
    }
    return $SIM_OPTIONS
  }
  
  
  proc get_env_variables {SIMULATOR_TOOL_BITNESS} {
    set ENV_VARIABLES [dict create]
    set LD_LIBRARY_PATH [dict create]
    set LD_LIBRARY_PATH [dict merge $LD_LIBRARY_PATH [dict get [NonDPR_master_0::get_env_variables $SIMULATOR_TOOL_BITNESS] "LD_LIBRARY_PATH"]]
    set LD_LIBRARY_PATH [dict merge $LD_LIBRARY_PATH [dict get [weight_mem::get_env_variables $SIMULATOR_TOOL_BITNESS] "LD_LIBRARY_PATH"]]
    set LD_LIBRARY_PATH [dict merge $LD_LIBRARY_PATH [dict get [data_mem::get_env_variables $SIMULATOR_TOOL_BITNESS] "LD_LIBRARY_PATH"]]
    set LD_LIBRARY_PATH [dict merge $LD_LIBRARY_PATH [dict get [out_mem::get_env_variables $SIMULATOR_TOOL_BITNESS] "LD_LIBRARY_PATH"]]
    set LD_LIBRARY_PATH [dict merge $LD_LIBRARY_PATH [dict get [PR_Test_Top_reset_in::get_env_variables $SIMULATOR_TOOL_BITNESS] "LD_LIBRARY_PATH"]]
    set LD_LIBRARY_PATH [dict merge $LD_LIBRARY_PATH [dict get [NonDPR_npu_system_28::get_env_variables $SIMULATOR_TOOL_BITNESS] "LD_LIBRARY_PATH"]]
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
    set libraries [dict merge $libraries [NonDPR_master_0::get_dpi_libraries "$QSYS_SIMDIR/../../ip/NonDPR/NonDPR_master_0/sim/"]]
    set libraries [dict merge $libraries [weight_mem::get_dpi_libraries "$QSYS_SIMDIR/../../ip/NonDPR/weight_mem/sim/"]]
    set libraries [dict merge $libraries [data_mem::get_dpi_libraries "$QSYS_SIMDIR/../../ip/NonDPR/data_mem/sim/"]]
    set libraries [dict merge $libraries [out_mem::get_dpi_libraries "$QSYS_SIMDIR/../../ip/NonDPR/out_mem/sim/"]]
    set libraries [dict merge $libraries [PR_Test_Top_reset_in::get_dpi_libraries "$QSYS_SIMDIR/../../ip/PR_Test_Top/PR_Test_Top_reset_in/sim/"]]
    set libraries [dict merge $libraries [NonDPR_npu_system_28::get_dpi_libraries "$QSYS_SIMDIR/../../ip/NonDPR/NonDPR_npu_system_28/sim/"]]
    
    return $libraries
  }
  
}
