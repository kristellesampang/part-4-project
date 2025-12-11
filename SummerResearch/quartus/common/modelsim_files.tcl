
proc get_design_libraries {} {
  set libraries [dict create]
  dict set libraries ram_2port_2041 1
  dict set libraries DataBuffer1    1
  dict set libraries ram_2port_2060 1
  dict set libraries DataBuffer2    1
  dict set libraries WeightBuffer1  1
  dict set libraries WeightBuffer2  1
  dict set libraries OutputBuffer   1
  return $libraries
}

proc get_memory_files {QSYS_SIMDIR} {
  set memory_files [list]
  return $memory_files
}

proc get_common_design_files {QSYS_SIMDIR} {
  set design_files [dict create]
  return $design_files
}

proc get_design_files {QSYS_SIMDIR} {
  set design_files [list]
  lappend design_files "-makelib ram_2port_2041 \"[normalize_path "$QSYS_SIMDIR/../Memory/Buffers/DataBuffer1/ram_2port_2041/sim/DataBuffer1_ram_2port_2041_tqhxgji.vhd"]\"   -end"  
  lappend design_files "-makelib DataBuffer1 \"[normalize_path "$QSYS_SIMDIR/../Memory/Buffers/DataBuffer1/sim/DataBuffer1.vhd"]\"   -end"                                           
  lappend design_files "-makelib ram_2port_2060 \"[normalize_path "$QSYS_SIMDIR/../Memory/Buffers/DataBuffer2/ram_2port_2060/sim/DataBuffer2_ram_2port_2060_btmqeni.v"]\"   -end"    
  lappend design_files "-makelib DataBuffer2 \"[normalize_path "$QSYS_SIMDIR/../Memory/Buffers/DataBuffer2/sim/DataBuffer2.vhd"]\"   -end"                                           
  lappend design_files "-makelib ram_2port_2060 \"[normalize_path "$QSYS_SIMDIR/../Memory/Buffers/WeightBuffer1/ram_2port_2060/sim/WeightBuffer1_ram_2port_2060_o45v4qi.v"]\"   -end"
  lappend design_files "-makelib WeightBuffer1 \"[normalize_path "$QSYS_SIMDIR/../Memory/Buffers/WeightBuffer1/sim/WeightBuffer1.vhd"]\"   -end"                                     
  lappend design_files "-makelib ram_2port_2060 \"[normalize_path "$QSYS_SIMDIR/../Memory/Buffers/WeightBuffer2/ram_2port_2060/sim/WeightBuffer2_ram_2port_2060_u3jegsa.v"]\"   -end"
  lappend design_files "-makelib WeightBuffer2 \"[normalize_path "$QSYS_SIMDIR/../Memory/Buffers/WeightBuffer2/sim/WeightBuffer2.vhd"]\"   -end"                                     
  lappend design_files "-makelib ram_2port_2041 \"[normalize_path "$QSYS_SIMDIR/../Memory/Buffers/OutputBuffer/ram_2port_2041/sim/OutputBuffer_ram_2port_2041_375fvbi.vhd"]\"   -end"
  lappend design_files "-makelib OutputBuffer \"[normalize_path "$QSYS_SIMDIR/../Memory/Buffers/OutputBuffer/sim/OutputBuffer.vhd"]\"   -end"                                        
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

