library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.numeric_std.all;

entity SDRAM_altera_emif_arch_nf_191_65stwui is
   generic (
      PROTOCOL_ENUM                                      : string                                   := "";
      PHY_TARGET_IS_ES                                   : boolean                                  := false;
      PHY_TARGET_IS_ES2                                  : boolean                                  := false;
      PHY_TARGET_IS_PRODUCTION                           : boolean                                  := false;
      PHY_CONFIG_ENUM                                    : string                                   := "";
      PHY_PING_PONG_EN                                   : boolean                                  := false;
      PHY_DLL_CORE_UPDN_EN                               : boolean                                  := false;
      PHY_CORE_CLKS_SHARING_ENUM                         : string                                   := "";
      PHY_CALIBRATED_OCT                                 : boolean                                  := false;
      PHY_AC_CALIBRATED_OCT                              : boolean                                  := false;
      PHY_CK_CALIBRATED_OCT                              : boolean                                  := false;
      PHY_DATA_CALIBRATED_OCT                            : boolean                                  := false;
      PHY_HPS_ENABLE_EARLY_RELEASE                       : boolean                                  := false;
      PLL_NUM_OF_EXTRA_CLKS                              : integer                                  := 0;
      MEM_FORMAT_ENUM                                    : string                                   := "";
      MEM_BURST_LENGTH                                   : integer                                  := 0;
      MEM_DATA_MASK_EN                                   : boolean                                  := false;
      MEM_TTL_DATA_WIDTH                                 : integer                                  := 0;
      MEM_TTL_NUM_OF_READ_GROUPS                         : integer                                  := 0;
      MEM_TTL_NUM_OF_WRITE_GROUPS                        : integer                                  := 0;
      DIAG_SIM_REGTEST_MODE                              : boolean                                  := false;
      DIAG_SYNTH_FOR_SIM                                 : boolean                                  := false;
      DIAG_ECLIPSE_DEBUG                                 : boolean                                  := false;
      DIAG_EXPORT_VJI                                    : boolean                                  := false;
      DIAG_INTERFACE_ID                                  : integer                                  := 0;
      DIAG_USE_ABSTRACT_PHY                              : boolean                                  := false;
      DIAG_SIM_VERBOSE_LEVEL                             : integer                                  := 0;
      DIAG_FAST_SIM                                      : boolean                                  := false;
      SILICON_REV                                        : string                                   := "";
      IS_HPS                                             : boolean                                  := false;
      IS_VID                                             : boolean                                  := false;
      USER_CLK_RATIO                                     : integer                                  := 0;
      C2P_P2C_CLK_RATIO                                  : integer                                  := 0;
      PHY_HMC_CLK_RATIO                                  : integer                                  := 0;
      DIAG_ABSTRACT_PHY_WLAT                             : integer                                  := 0;
      DIAG_ABSTRACT_PHY_RLAT                             : integer                                  := 0;
      DIAG_CPA_OUT_1_EN                                  : boolean                                  := false;
      DIAG_USE_CPA_LOCK                                  : boolean                                  := false;
      DQS_BUS_MODE_ENUM                                  : string                                   := "";
      AC_PIN_MAP_SCHEME                                  : string                                   := "";
      NUM_OF_HMC_PORTS                                   : integer                                  := 0;
      HMC_AVL_PROTOCOL_ENUM                              : string                                   := "";
      HMC_CTRL_DIMM_TYPE                                 : string                                   := "";
      REGISTER_AFI                                       : boolean                                  := false;
      SEQ_SYNTH_CPU_CLK_DIVIDE                           : integer                                  := 0;
      SEQ_SYNTH_CAL_CLK_DIVIDE                           : integer                                  := 0;
      SEQ_SIM_CPU_CLK_DIVIDE                             : integer                                  := 0;
      SEQ_SIM_CAL_CLK_DIVIDE                             : integer                                  := 0;
      SEQ_SYNTH_OSC_FREQ_MHZ                             : integer                                  := 0;
      SEQ_SIM_OSC_FREQ_MHZ                               : integer                                  := 0;
      NUM_OF_RTL_TILES                                   : integer                                  := 0;
      PRI_RDATA_TILE_INDEX                               : integer                                  := 0;
      PRI_RDATA_LANE_INDEX                               : integer                                  := 0;
      PRI_WDATA_TILE_INDEX                               : integer                                  := 0;
      PRI_WDATA_LANE_INDEX                               : integer                                  := 0;
      PRI_AC_TILE_INDEX                                  : integer                                  := 0;
      SEC_RDATA_TILE_INDEX                               : integer                                  := 0;
      SEC_RDATA_LANE_INDEX                               : integer                                  := 0;
      SEC_WDATA_TILE_INDEX                               : integer                                  := 0;
      SEC_WDATA_LANE_INDEX                               : integer                                  := 0;
      SEC_AC_TILE_INDEX                                  : integer                                  := 0;
      LANES_USAGE_0                                      : integer                                  := 0;
      LANES_USAGE_1                                      : integer                                  := 0;
      LANES_USAGE_2                                      : integer                                  := 0;
      LANES_USAGE_3                                      : integer                                  := 0;
      LANES_USAGE_AUTOGEN_WCNT                           : integer                                  := 0;
      PINS_USAGE_0                                       : integer                                  := 0;
      PINS_USAGE_1                                       : integer                                  := 0;
      PINS_USAGE_2                                       : integer                                  := 0;
      PINS_USAGE_3                                       : integer                                  := 0;
      PINS_USAGE_4                                       : integer                                  := 0;
      PINS_USAGE_5                                       : integer                                  := 0;
      PINS_USAGE_6                                       : integer                                  := 0;
      PINS_USAGE_7                                       : integer                                  := 0;
      PINS_USAGE_8                                       : integer                                  := 0;
      PINS_USAGE_9                                       : integer                                  := 0;
      PINS_USAGE_10                                      : integer                                  := 0;
      PINS_USAGE_11                                      : integer                                  := 0;
      PINS_USAGE_12                                      : integer                                  := 0;
      PINS_USAGE_AUTOGEN_WCNT                            : integer                                  := 0;
      PINS_RATE_0                                        : integer                                  := 0;
      PINS_RATE_1                                        : integer                                  := 0;
      PINS_RATE_2                                        : integer                                  := 0;
      PINS_RATE_3                                        : integer                                  := 0;
      PINS_RATE_4                                        : integer                                  := 0;
      PINS_RATE_5                                        : integer                                  := 0;
      PINS_RATE_6                                        : integer                                  := 0;
      PINS_RATE_7                                        : integer                                  := 0;
      PINS_RATE_8                                        : integer                                  := 0;
      PINS_RATE_9                                        : integer                                  := 0;
      PINS_RATE_10                                       : integer                                  := 0;
      PINS_RATE_11                                       : integer                                  := 0;
      PINS_RATE_12                                       : integer                                  := 0;
      PINS_RATE_AUTOGEN_WCNT                             : integer                                  := 0;
      PINS_WDB_0                                         : integer                                  := 0;
      PINS_WDB_1                                         : integer                                  := 0;
      PINS_WDB_2                                         : integer                                  := 0;
      PINS_WDB_3                                         : integer                                  := 0;
      PINS_WDB_4                                         : integer                                  := 0;
      PINS_WDB_5                                         : integer                                  := 0;
      PINS_WDB_6                                         : integer                                  := 0;
      PINS_WDB_7                                         : integer                                  := 0;
      PINS_WDB_8                                         : integer                                  := 0;
      PINS_WDB_9                                         : integer                                  := 0;
      PINS_WDB_10                                        : integer                                  := 0;
      PINS_WDB_11                                        : integer                                  := 0;
      PINS_WDB_12                                        : integer                                  := 0;
      PINS_WDB_13                                        : integer                                  := 0;
      PINS_WDB_14                                        : integer                                  := 0;
      PINS_WDB_15                                        : integer                                  := 0;
      PINS_WDB_16                                        : integer                                  := 0;
      PINS_WDB_17                                        : integer                                  := 0;
      PINS_WDB_18                                        : integer                                  := 0;
      PINS_WDB_19                                        : integer                                  := 0;
      PINS_WDB_20                                        : integer                                  := 0;
      PINS_WDB_21                                        : integer                                  := 0;
      PINS_WDB_22                                        : integer                                  := 0;
      PINS_WDB_23                                        : integer                                  := 0;
      PINS_WDB_24                                        : integer                                  := 0;
      PINS_WDB_25                                        : integer                                  := 0;
      PINS_WDB_26                                        : integer                                  := 0;
      PINS_WDB_27                                        : integer                                  := 0;
      PINS_WDB_28                                        : integer                                  := 0;
      PINS_WDB_29                                        : integer                                  := 0;
      PINS_WDB_30                                        : integer                                  := 0;
      PINS_WDB_31                                        : integer                                  := 0;
      PINS_WDB_32                                        : integer                                  := 0;
      PINS_WDB_33                                        : integer                                  := 0;
      PINS_WDB_34                                        : integer                                  := 0;
      PINS_WDB_35                                        : integer                                  := 0;
      PINS_WDB_36                                        : integer                                  := 0;
      PINS_WDB_37                                        : integer                                  := 0;
      PINS_WDB_38                                        : integer                                  := 0;
      PINS_WDB_AUTOGEN_WCNT                              : integer                                  := 0;
      PINS_DATA_IN_MODE_0                                : integer                                  := 0;
      PINS_DATA_IN_MODE_1                                : integer                                  := 0;
      PINS_DATA_IN_MODE_2                                : integer                                  := 0;
      PINS_DATA_IN_MODE_3                                : integer                                  := 0;
      PINS_DATA_IN_MODE_4                                : integer                                  := 0;
      PINS_DATA_IN_MODE_5                                : integer                                  := 0;
      PINS_DATA_IN_MODE_6                                : integer                                  := 0;
      PINS_DATA_IN_MODE_7                                : integer                                  := 0;
      PINS_DATA_IN_MODE_8                                : integer                                  := 0;
      PINS_DATA_IN_MODE_9                                : integer                                  := 0;
      PINS_DATA_IN_MODE_10                               : integer                                  := 0;
      PINS_DATA_IN_MODE_11                               : integer                                  := 0;
      PINS_DATA_IN_MODE_12                               : integer                                  := 0;
      PINS_DATA_IN_MODE_13                               : integer                                  := 0;
      PINS_DATA_IN_MODE_14                               : integer                                  := 0;
      PINS_DATA_IN_MODE_15                               : integer                                  := 0;
      PINS_DATA_IN_MODE_16                               : integer                                  := 0;
      PINS_DATA_IN_MODE_17                               : integer                                  := 0;
      PINS_DATA_IN_MODE_18                               : integer                                  := 0;
      PINS_DATA_IN_MODE_19                               : integer                                  := 0;
      PINS_DATA_IN_MODE_20                               : integer                                  := 0;
      PINS_DATA_IN_MODE_21                               : integer                                  := 0;
      PINS_DATA_IN_MODE_22                               : integer                                  := 0;
      PINS_DATA_IN_MODE_23                               : integer                                  := 0;
      PINS_DATA_IN_MODE_24                               : integer                                  := 0;
      PINS_DATA_IN_MODE_25                               : integer                                  := 0;
      PINS_DATA_IN_MODE_26                               : integer                                  := 0;
      PINS_DATA_IN_MODE_27                               : integer                                  := 0;
      PINS_DATA_IN_MODE_28                               : integer                                  := 0;
      PINS_DATA_IN_MODE_29                               : integer                                  := 0;
      PINS_DATA_IN_MODE_30                               : integer                                  := 0;
      PINS_DATA_IN_MODE_31                               : integer                                  := 0;
      PINS_DATA_IN_MODE_32                               : integer                                  := 0;
      PINS_DATA_IN_MODE_33                               : integer                                  := 0;
      PINS_DATA_IN_MODE_34                               : integer                                  := 0;
      PINS_DATA_IN_MODE_35                               : integer                                  := 0;
      PINS_DATA_IN_MODE_36                               : integer                                  := 0;
      PINS_DATA_IN_MODE_37                               : integer                                  := 0;
      PINS_DATA_IN_MODE_38                               : integer                                  := 0;
      PINS_DATA_IN_MODE_AUTOGEN_WCNT                     : integer                                  := 0;
      PINS_C2L_DRIVEN_0                                  : integer                                  := 0;
      PINS_C2L_DRIVEN_1                                  : integer                                  := 0;
      PINS_C2L_DRIVEN_2                                  : integer                                  := 0;
      PINS_C2L_DRIVEN_3                                  : integer                                  := 0;
      PINS_C2L_DRIVEN_4                                  : integer                                  := 0;
      PINS_C2L_DRIVEN_5                                  : integer                                  := 0;
      PINS_C2L_DRIVEN_6                                  : integer                                  := 0;
      PINS_C2L_DRIVEN_7                                  : integer                                  := 0;
      PINS_C2L_DRIVEN_8                                  : integer                                  := 0;
      PINS_C2L_DRIVEN_9                                  : integer                                  := 0;
      PINS_C2L_DRIVEN_10                                 : integer                                  := 0;
      PINS_C2L_DRIVEN_11                                 : integer                                  := 0;
      PINS_C2L_DRIVEN_12                                 : integer                                  := 0;
      PINS_C2L_DRIVEN_AUTOGEN_WCNT                       : integer                                  := 0;
      PINS_DB_IN_BYPASS_0                                : integer                                  := 0;
      PINS_DB_IN_BYPASS_1                                : integer                                  := 0;
      PINS_DB_IN_BYPASS_2                                : integer                                  := 0;
      PINS_DB_IN_BYPASS_3                                : integer                                  := 0;
      PINS_DB_IN_BYPASS_4                                : integer                                  := 0;
      PINS_DB_IN_BYPASS_5                                : integer                                  := 0;
      PINS_DB_IN_BYPASS_6                                : integer                                  := 0;
      PINS_DB_IN_BYPASS_7                                : integer                                  := 0;
      PINS_DB_IN_BYPASS_8                                : integer                                  := 0;
      PINS_DB_IN_BYPASS_9                                : integer                                  := 0;
      PINS_DB_IN_BYPASS_10                               : integer                                  := 0;
      PINS_DB_IN_BYPASS_11                               : integer                                  := 0;
      PINS_DB_IN_BYPASS_12                               : integer                                  := 0;
      PINS_DB_IN_BYPASS_AUTOGEN_WCNT                     : integer                                  := 0;
      PINS_DB_OUT_BYPASS_0                               : integer                                  := 0;
      PINS_DB_OUT_BYPASS_1                               : integer                                  := 0;
      PINS_DB_OUT_BYPASS_2                               : integer                                  := 0;
      PINS_DB_OUT_BYPASS_3                               : integer                                  := 0;
      PINS_DB_OUT_BYPASS_4                               : integer                                  := 0;
      PINS_DB_OUT_BYPASS_5                               : integer                                  := 0;
      PINS_DB_OUT_BYPASS_6                               : integer                                  := 0;
      PINS_DB_OUT_BYPASS_7                               : integer                                  := 0;
      PINS_DB_OUT_BYPASS_8                               : integer                                  := 0;
      PINS_DB_OUT_BYPASS_9                               : integer                                  := 0;
      PINS_DB_OUT_BYPASS_10                              : integer                                  := 0;
      PINS_DB_OUT_BYPASS_11                              : integer                                  := 0;
      PINS_DB_OUT_BYPASS_12                              : integer                                  := 0;
      PINS_DB_OUT_BYPASS_AUTOGEN_WCNT                    : integer                                  := 0;
      PINS_DB_OE_BYPASS_0                                : integer                                  := 0;
      PINS_DB_OE_BYPASS_1                                : integer                                  := 0;
      PINS_DB_OE_BYPASS_2                                : integer                                  := 0;
      PINS_DB_OE_BYPASS_3                                : integer                                  := 0;
      PINS_DB_OE_BYPASS_4                                : integer                                  := 0;
      PINS_DB_OE_BYPASS_5                                : integer                                  := 0;
      PINS_DB_OE_BYPASS_6                                : integer                                  := 0;
      PINS_DB_OE_BYPASS_7                                : integer                                  := 0;
      PINS_DB_OE_BYPASS_8                                : integer                                  := 0;
      PINS_DB_OE_BYPASS_9                                : integer                                  := 0;
      PINS_DB_OE_BYPASS_10                               : integer                                  := 0;
      PINS_DB_OE_BYPASS_11                               : integer                                  := 0;
      PINS_DB_OE_BYPASS_12                               : integer                                  := 0;
      PINS_DB_OE_BYPASS_AUTOGEN_WCNT                     : integer                                  := 0;
      PINS_INVERT_WR_0                                   : integer                                  := 0;
      PINS_INVERT_WR_1                                   : integer                                  := 0;
      PINS_INVERT_WR_2                                   : integer                                  := 0;
      PINS_INVERT_WR_3                                   : integer                                  := 0;
      PINS_INVERT_WR_4                                   : integer                                  := 0;
      PINS_INVERT_WR_5                                   : integer                                  := 0;
      PINS_INVERT_WR_6                                   : integer                                  := 0;
      PINS_INVERT_WR_7                                   : integer                                  := 0;
      PINS_INVERT_WR_8                                   : integer                                  := 0;
      PINS_INVERT_WR_9                                   : integer                                  := 0;
      PINS_INVERT_WR_10                                  : integer                                  := 0;
      PINS_INVERT_WR_11                                  : integer                                  := 0;
      PINS_INVERT_WR_12                                  : integer                                  := 0;
      PINS_INVERT_WR_AUTOGEN_WCNT                        : integer                                  := 0;
      PINS_INVERT_OE_0                                   : integer                                  := 0;
      PINS_INVERT_OE_1                                   : integer                                  := 0;
      PINS_INVERT_OE_2                                   : integer                                  := 0;
      PINS_INVERT_OE_3                                   : integer                                  := 0;
      PINS_INVERT_OE_4                                   : integer                                  := 0;
      PINS_INVERT_OE_5                                   : integer                                  := 0;
      PINS_INVERT_OE_6                                   : integer                                  := 0;
      PINS_INVERT_OE_7                                   : integer                                  := 0;
      PINS_INVERT_OE_8                                   : integer                                  := 0;
      PINS_INVERT_OE_9                                   : integer                                  := 0;
      PINS_INVERT_OE_10                                  : integer                                  := 0;
      PINS_INVERT_OE_11                                  : integer                                  := 0;
      PINS_INVERT_OE_12                                  : integer                                  := 0;
      PINS_INVERT_OE_AUTOGEN_WCNT                        : integer                                  := 0;
      PINS_AC_HMC_DATA_OVERRIDE_ENA_0                    : integer                                  := 0;
      PINS_AC_HMC_DATA_OVERRIDE_ENA_1                    : integer                                  := 0;
      PINS_AC_HMC_DATA_OVERRIDE_ENA_2                    : integer                                  := 0;
      PINS_AC_HMC_DATA_OVERRIDE_ENA_3                    : integer                                  := 0;
      PINS_AC_HMC_DATA_OVERRIDE_ENA_4                    : integer                                  := 0;
      PINS_AC_HMC_DATA_OVERRIDE_ENA_5                    : integer                                  := 0;
      PINS_AC_HMC_DATA_OVERRIDE_ENA_6                    : integer                                  := 0;
      PINS_AC_HMC_DATA_OVERRIDE_ENA_7                    : integer                                  := 0;
      PINS_AC_HMC_DATA_OVERRIDE_ENA_8                    : integer                                  := 0;
      PINS_AC_HMC_DATA_OVERRIDE_ENA_9                    : integer                                  := 0;
      PINS_AC_HMC_DATA_OVERRIDE_ENA_10                   : integer                                  := 0;
      PINS_AC_HMC_DATA_OVERRIDE_ENA_11                   : integer                                  := 0;
      PINS_AC_HMC_DATA_OVERRIDE_ENA_12                   : integer                                  := 0;
      PINS_AC_HMC_DATA_OVERRIDE_ENA_AUTOGEN_WCNT         : integer                                  := 0;
      PINS_OCT_MODE_0                                    : integer                                  := 0;
      PINS_OCT_MODE_1                                    : integer                                  := 0;
      PINS_OCT_MODE_2                                    : integer                                  := 0;
      PINS_OCT_MODE_3                                    : integer                                  := 0;
      PINS_OCT_MODE_4                                    : integer                                  := 0;
      PINS_OCT_MODE_5                                    : integer                                  := 0;
      PINS_OCT_MODE_6                                    : integer                                  := 0;
      PINS_OCT_MODE_7                                    : integer                                  := 0;
      PINS_OCT_MODE_8                                    : integer                                  := 0;
      PINS_OCT_MODE_9                                    : integer                                  := 0;
      PINS_OCT_MODE_10                                   : integer                                  := 0;
      PINS_OCT_MODE_11                                   : integer                                  := 0;
      PINS_OCT_MODE_12                                   : integer                                  := 0;
      PINS_OCT_MODE_AUTOGEN_WCNT                         : integer                                  := 0;
      PINS_GPIO_MODE_0                                   : integer                                  := 0;
      PINS_GPIO_MODE_1                                   : integer                                  := 0;
      PINS_GPIO_MODE_2                                   : integer                                  := 0;
      PINS_GPIO_MODE_3                                   : integer                                  := 0;
      PINS_GPIO_MODE_4                                   : integer                                  := 0;
      PINS_GPIO_MODE_5                                   : integer                                  := 0;
      PINS_GPIO_MODE_6                                   : integer                                  := 0;
      PINS_GPIO_MODE_7                                   : integer                                  := 0;
      PINS_GPIO_MODE_8                                   : integer                                  := 0;
      PINS_GPIO_MODE_9                                   : integer                                  := 0;
      PINS_GPIO_MODE_10                                  : integer                                  := 0;
      PINS_GPIO_MODE_11                                  : integer                                  := 0;
      PINS_GPIO_MODE_12                                  : integer                                  := 0;
      PINS_GPIO_MODE_AUTOGEN_WCNT                        : integer                                  := 0;
      UNUSED_MEM_PINS_PINLOC_0                           : integer                                  := 0;
      UNUSED_MEM_PINS_PINLOC_1                           : integer                                  := 0;
      UNUSED_MEM_PINS_PINLOC_2                           : integer                                  := 0;
      UNUSED_MEM_PINS_PINLOC_3                           : integer                                  := 0;
      UNUSED_MEM_PINS_PINLOC_4                           : integer                                  := 0;
      UNUSED_MEM_PINS_PINLOC_5                           : integer                                  := 0;
      UNUSED_MEM_PINS_PINLOC_6                           : integer                                  := 0;
      UNUSED_MEM_PINS_PINLOC_7                           : integer                                  := 0;
      UNUSED_MEM_PINS_PINLOC_8                           : integer                                  := 0;
      UNUSED_MEM_PINS_PINLOC_9                           : integer                                  := 0;
      UNUSED_MEM_PINS_PINLOC_10                          : integer                                  := 0;
      UNUSED_MEM_PINS_PINLOC_11                          : integer                                  := 0;
      UNUSED_MEM_PINS_PINLOC_12                          : integer                                  := 0;
      UNUSED_MEM_PINS_PINLOC_13                          : integer                                  := 0;
      UNUSED_MEM_PINS_PINLOC_14                          : integer                                  := 0;
      UNUSED_MEM_PINS_PINLOC_15                          : integer                                  := 0;
      UNUSED_MEM_PINS_PINLOC_16                          : integer                                  := 0;
      UNUSED_MEM_PINS_PINLOC_17                          : integer                                  := 0;
      UNUSED_MEM_PINS_PINLOC_18                          : integer                                  := 0;
      UNUSED_MEM_PINS_PINLOC_19                          : integer                                  := 0;
      UNUSED_MEM_PINS_PINLOC_20                          : integer                                  := 0;
      UNUSED_MEM_PINS_PINLOC_21                          : integer                                  := 0;
      UNUSED_MEM_PINS_PINLOC_22                          : integer                                  := 0;
      UNUSED_MEM_PINS_PINLOC_23                          : integer                                  := 0;
      UNUSED_MEM_PINS_PINLOC_24                          : integer                                  := 0;
      UNUSED_MEM_PINS_PINLOC_25                          : integer                                  := 0;
      UNUSED_MEM_PINS_PINLOC_26                          : integer                                  := 0;
      UNUSED_MEM_PINS_PINLOC_27                          : integer                                  := 0;
      UNUSED_MEM_PINS_PINLOC_28                          : integer                                  := 0;
      UNUSED_MEM_PINS_PINLOC_29                          : integer                                  := 0;
      UNUSED_MEM_PINS_PINLOC_30                          : integer                                  := 0;
      UNUSED_MEM_PINS_PINLOC_31                          : integer                                  := 0;
      UNUSED_MEM_PINS_PINLOC_32                          : integer                                  := 0;
      UNUSED_MEM_PINS_PINLOC_33                          : integer                                  := 0;
      UNUSED_MEM_PINS_PINLOC_34                          : integer                                  := 0;
      UNUSED_MEM_PINS_PINLOC_35                          : integer                                  := 0;
      UNUSED_MEM_PINS_PINLOC_36                          : integer                                  := 0;
      UNUSED_MEM_PINS_PINLOC_37                          : integer                                  := 0;
      UNUSED_MEM_PINS_PINLOC_38                          : integer                                  := 0;
      UNUSED_MEM_PINS_PINLOC_39                          : integer                                  := 0;
      UNUSED_MEM_PINS_PINLOC_40                          : integer                                  := 0;
      UNUSED_MEM_PINS_PINLOC_41                          : integer                                  := 0;
      UNUSED_MEM_PINS_PINLOC_42                          : integer                                  := 0;
      UNUSED_MEM_PINS_PINLOC_43                          : integer                                  := 0;
      UNUSED_MEM_PINS_PINLOC_44                          : integer                                  := 0;
      UNUSED_MEM_PINS_PINLOC_45                          : integer                                  := 0;
      UNUSED_MEM_PINS_PINLOC_46                          : integer                                  := 0;
      UNUSED_MEM_PINS_PINLOC_47                          : integer                                  := 0;
      UNUSED_MEM_PINS_PINLOC_48                          : integer                                  := 0;
      UNUSED_MEM_PINS_PINLOC_49                          : integer                                  := 0;
      UNUSED_MEM_PINS_PINLOC_50                          : integer                                  := 0;
      UNUSED_MEM_PINS_PINLOC_51                          : integer                                  := 0;
      UNUSED_MEM_PINS_PINLOC_52                          : integer                                  := 0;
      UNUSED_MEM_PINS_PINLOC_53                          : integer                                  := 0;
      UNUSED_MEM_PINS_PINLOC_54                          : integer                                  := 0;
      UNUSED_MEM_PINS_PINLOC_55                          : integer                                  := 0;
      UNUSED_MEM_PINS_PINLOC_56                          : integer                                  := 0;
      UNUSED_MEM_PINS_PINLOC_57                          : integer                                  := 0;
      UNUSED_MEM_PINS_PINLOC_58                          : integer                                  := 0;
      UNUSED_MEM_PINS_PINLOC_59                          : integer                                  := 0;
      UNUSED_MEM_PINS_PINLOC_60                          : integer                                  := 0;
      UNUSED_MEM_PINS_PINLOC_61                          : integer                                  := 0;
      UNUSED_MEM_PINS_PINLOC_62                          : integer                                  := 0;
      UNUSED_MEM_PINS_PINLOC_63                          : integer                                  := 0;
      UNUSED_MEM_PINS_PINLOC_64                          : integer                                  := 0;
      UNUSED_MEM_PINS_PINLOC_65                          : integer                                  := 0;
      UNUSED_MEM_PINS_PINLOC_66                          : integer                                  := 0;
      UNUSED_MEM_PINS_PINLOC_67                          : integer                                  := 0;
      UNUSED_MEM_PINS_PINLOC_68                          : integer                                  := 0;
      UNUSED_MEM_PINS_PINLOC_69                          : integer                                  := 0;
      UNUSED_MEM_PINS_PINLOC_70                          : integer                                  := 0;
      UNUSED_MEM_PINS_PINLOC_71                          : integer                                  := 0;
      UNUSED_MEM_PINS_PINLOC_72                          : integer                                  := 0;
      UNUSED_MEM_PINS_PINLOC_73                          : integer                                  := 0;
      UNUSED_MEM_PINS_PINLOC_74                          : integer                                  := 0;
      UNUSED_MEM_PINS_PINLOC_75                          : integer                                  := 0;
      UNUSED_MEM_PINS_PINLOC_76                          : integer                                  := 0;
      UNUSED_MEM_PINS_PINLOC_77                          : integer                                  := 0;
      UNUSED_MEM_PINS_PINLOC_78                          : integer                                  := 0;
      UNUSED_MEM_PINS_PINLOC_79                          : integer                                  := 0;
      UNUSED_MEM_PINS_PINLOC_80                          : integer                                  := 0;
      UNUSED_MEM_PINS_PINLOC_81                          : integer                                  := 0;
      UNUSED_MEM_PINS_PINLOC_82                          : integer                                  := 0;
      UNUSED_MEM_PINS_PINLOC_83                          : integer                                  := 0;
      UNUSED_MEM_PINS_PINLOC_84                          : integer                                  := 0;
      UNUSED_MEM_PINS_PINLOC_85                          : integer                                  := 0;
      UNUSED_MEM_PINS_PINLOC_86                          : integer                                  := 0;
      UNUSED_MEM_PINS_PINLOC_87                          : integer                                  := 0;
      UNUSED_MEM_PINS_PINLOC_88                          : integer                                  := 0;
      UNUSED_MEM_PINS_PINLOC_89                          : integer                                  := 0;
      UNUSED_MEM_PINS_PINLOC_90                          : integer                                  := 0;
      UNUSED_MEM_PINS_PINLOC_91                          : integer                                  := 0;
      UNUSED_MEM_PINS_PINLOC_92                          : integer                                  := 0;
      UNUSED_MEM_PINS_PINLOC_93                          : integer                                  := 0;
      UNUSED_MEM_PINS_PINLOC_94                          : integer                                  := 0;
      UNUSED_MEM_PINS_PINLOC_95                          : integer                                  := 0;
      UNUSED_MEM_PINS_PINLOC_96                          : integer                                  := 0;
      UNUSED_MEM_PINS_PINLOC_97                          : integer                                  := 0;
      UNUSED_MEM_PINS_PINLOC_98                          : integer                                  := 0;
      UNUSED_MEM_PINS_PINLOC_99                          : integer                                  := 0;
      UNUSED_MEM_PINS_PINLOC_100                         : integer                                  := 0;
      UNUSED_MEM_PINS_PINLOC_101                         : integer                                  := 0;
      UNUSED_MEM_PINS_PINLOC_102                         : integer                                  := 0;
      UNUSED_MEM_PINS_PINLOC_103                         : integer                                  := 0;
      UNUSED_MEM_PINS_PINLOC_104                         : integer                                  := 0;
      UNUSED_MEM_PINS_PINLOC_105                         : integer                                  := 0;
      UNUSED_MEM_PINS_PINLOC_106                         : integer                                  := 0;
      UNUSED_MEM_PINS_PINLOC_107                         : integer                                  := 0;
      UNUSED_MEM_PINS_PINLOC_108                         : integer                                  := 0;
      UNUSED_MEM_PINS_PINLOC_109                         : integer                                  := 0;
      UNUSED_MEM_PINS_PINLOC_110                         : integer                                  := 0;
      UNUSED_MEM_PINS_PINLOC_111                         : integer                                  := 0;
      UNUSED_MEM_PINS_PINLOC_112                         : integer                                  := 0;
      UNUSED_MEM_PINS_PINLOC_113                         : integer                                  := 0;
      UNUSED_MEM_PINS_PINLOC_114                         : integer                                  := 0;
      UNUSED_MEM_PINS_PINLOC_115                         : integer                                  := 0;
      UNUSED_MEM_PINS_PINLOC_116                         : integer                                  := 0;
      UNUSED_MEM_PINS_PINLOC_117                         : integer                                  := 0;
      UNUSED_MEM_PINS_PINLOC_118                         : integer                                  := 0;
      UNUSED_MEM_PINS_PINLOC_119                         : integer                                  := 0;
      UNUSED_MEM_PINS_PINLOC_120                         : integer                                  := 0;
      UNUSED_MEM_PINS_PINLOC_121                         : integer                                  := 0;
      UNUSED_MEM_PINS_PINLOC_122                         : integer                                  := 0;
      UNUSED_MEM_PINS_PINLOC_123                         : integer                                  := 0;
      UNUSED_MEM_PINS_PINLOC_124                         : integer                                  := 0;
      UNUSED_MEM_PINS_PINLOC_125                         : integer                                  := 0;
      UNUSED_MEM_PINS_PINLOC_126                         : integer                                  := 0;
      UNUSED_MEM_PINS_PINLOC_127                         : integer                                  := 0;
      UNUSED_MEM_PINS_PINLOC_128                         : integer                                  := 0;
      UNUSED_MEM_PINS_PINLOC_AUTOGEN_WCNT                : integer                                  := 0;
      UNUSED_DQS_BUSES_LANELOC_0                         : integer                                  := 0;
      UNUSED_DQS_BUSES_LANELOC_1                         : integer                                  := 0;
      UNUSED_DQS_BUSES_LANELOC_2                         : integer                                  := 0;
      UNUSED_DQS_BUSES_LANELOC_3                         : integer                                  := 0;
      UNUSED_DQS_BUSES_LANELOC_4                         : integer                                  := 0;
      UNUSED_DQS_BUSES_LANELOC_5                         : integer                                  := 0;
      UNUSED_DQS_BUSES_LANELOC_6                         : integer                                  := 0;
      UNUSED_DQS_BUSES_LANELOC_7                         : integer                                  := 0;
      UNUSED_DQS_BUSES_LANELOC_8                         : integer                                  := 0;
      UNUSED_DQS_BUSES_LANELOC_9                         : integer                                  := 0;
      UNUSED_DQS_BUSES_LANELOC_10                        : integer                                  := 0;
      UNUSED_DQS_BUSES_LANELOC_AUTOGEN_WCNT              : integer                                  := 0;
      CENTER_TIDS_0                                      : integer                                  := 0;
      CENTER_TIDS_1                                      : integer                                  := 0;
      CENTER_TIDS_2                                      : integer                                  := 0;
      CENTER_TIDS_AUTOGEN_WCNT                           : integer                                  := 0;
      HMC_TIDS_0                                         : integer                                  := 0;
      HMC_TIDS_1                                         : integer                                  := 0;
      HMC_TIDS_2                                         : integer                                  := 0;
      HMC_TIDS_AUTOGEN_WCNT                              : integer                                  := 0;
      LANE_TIDS_0                                        : integer                                  := 0;
      LANE_TIDS_1                                        : integer                                  := 0;
      LANE_TIDS_2                                        : integer                                  := 0;
      LANE_TIDS_3                                        : integer                                  := 0;
      LANE_TIDS_4                                        : integer                                  := 0;
      LANE_TIDS_5                                        : integer                                  := 0;
      LANE_TIDS_6                                        : integer                                  := 0;
      LANE_TIDS_7                                        : integer                                  := 0;
      LANE_TIDS_8                                        : integer                                  := 0;
      LANE_TIDS_9                                        : integer                                  := 0;
      LANE_TIDS_AUTOGEN_WCNT                             : integer                                  := 0;
      PREAMBLE_MODE                                      : string                                   := "";
      DBI_WR_ENABLE                                      : string                                   := "";
      DBI_RD_ENABLE                                      : string                                   := "";
      CRC_EN                                             : string                                   := "";
      SWAP_DQS_A_B                                       : string                                   := "";
      DQS_PACK_MODE                                      : string                                   := "";
      OCT_SIZE                                           : integer                                  := 0;
      DBC_WB_RESERVED_ENTRY                              : integer                                  := 0;
      DLL_MODE                                           : string                                   := "";
      DLL_CODEWORD                                       : integer                                  := 0;
      ABPHY_WRITE_PROTOCOL                               : integer                                  := 0;
      PHY_USERMODE_OCT                                   : boolean                                  := false;
      PHY_PERIODIC_OCT_RECAL                             : boolean                                  := false;
      PHY_HAS_DCC                                        : boolean                                  := false;
      PRI_HMC_CFG_ENABLE_ECC                             : string                                   := "";
      PRI_HMC_CFG_REORDER_DATA                           : string                                   := "";
      PRI_HMC_CFG_REORDER_READ                           : string                                   := "";
      PRI_HMC_CFG_REORDER_RDATA                          : string                                   := "";
      PRI_HMC_CFG_STARVE_LIMIT                           : integer                                  := 0;
      PRI_HMC_CFG_DQS_TRACKING_EN                        : string                                   := "";
      PRI_HMC_CFG_ARBITER_TYPE                           : string                                   := "";
      PRI_HMC_CFG_OPEN_PAGE_EN                           : string                                   := "";
      PRI_HMC_CFG_GEAR_DOWN_EN                           : string                                   := "";
      PRI_HMC_CFG_RLD3_MULTIBANK_MODE                    : string                                   := "";
      PRI_HMC_CFG_PING_PONG_MODE                         : string                                   := "";
      PRI_HMC_CFG_SLOT_ROTATE_EN                         : integer                                  := 0;
      PRI_HMC_CFG_SLOT_OFFSET                            : integer                                  := 0;
      PRI_HMC_CFG_COL_CMD_SLOT                           : integer                                  := 0;
      PRI_HMC_CFG_ROW_CMD_SLOT                           : integer                                  := 0;
      PRI_HMC_CFG_ENABLE_RC                              : string                                   := "";
      PRI_HMC_CFG_CS_TO_CHIP_MAPPING                     : integer                                  := 0;
      PRI_HMC_CFG_RB_RESERVED_ENTRY                      : integer                                  := 0;
      PRI_HMC_CFG_WB_RESERVED_ENTRY                      : integer                                  := 0;
      PRI_HMC_CFG_TCL                                    : integer                                  := 0;
      PRI_HMC_CFG_POWER_SAVING_EXIT_CYC                  : integer                                  := 0;
      PRI_HMC_CFG_MEM_CLK_DISABLE_ENTRY_CYC              : integer                                  := 0;
      PRI_HMC_CFG_WRITE_ODT_CHIP                         : integer                                  := 0;
      PRI_HMC_CFG_READ_ODT_CHIP                          : integer                                  := 0;
      PRI_HMC_CFG_WR_ODT_ON                              : integer                                  := 0;
      PRI_HMC_CFG_RD_ODT_ON                              : integer                                  := 0;
      PRI_HMC_CFG_WR_ODT_PERIOD                          : integer                                  := 0;
      PRI_HMC_CFG_RD_ODT_PERIOD                          : integer                                  := 0;
      PRI_HMC_CFG_RLD3_REFRESH_SEQ0                      : integer                                  := 0;
      PRI_HMC_CFG_RLD3_REFRESH_SEQ1                      : integer                                  := 0;
      PRI_HMC_CFG_RLD3_REFRESH_SEQ2                      : integer                                  := 0;
      PRI_HMC_CFG_RLD3_REFRESH_SEQ3                      : integer                                  := 0;
      PRI_HMC_CFG_SRF_ZQCAL_DISABLE                      : string                                   := "";
      PRI_HMC_CFG_MPS_ZQCAL_DISABLE                      : string                                   := "";
      PRI_HMC_CFG_MPS_DQSTRK_DISABLE                     : string                                   := "";
      PRI_HMC_CFG_SHORT_DQSTRK_CTRL_EN                   : string                                   := "";
      PRI_HMC_CFG_PERIOD_DQSTRK_CTRL_EN                  : string                                   := "";
      PRI_HMC_CFG_PERIOD_DQSTRK_INTERVAL                 : integer                                  := 0;
      PRI_HMC_CFG_DQSTRK_TO_VALID_LAST                   : integer                                  := 0;
      PRI_HMC_CFG_DQSTRK_TO_VALID                        : integer                                  := 0;
      PRI_HMC_CFG_RFSH_WARN_THRESHOLD                    : integer                                  := 0;
      PRI_HMC_CFG_SB_CG_DISABLE                          : string                                   := "";
      PRI_HMC_CFG_USER_RFSH_EN                           : string                                   := "";
      PRI_HMC_CFG_SRF_AUTOEXIT_EN                        : string                                   := "";
      PRI_HMC_CFG_SRF_ENTRY_EXIT_BLOCK                   : string                                   := "";
      PRI_HMC_CFG_SB_DDR4_MR3                            : integer                                  := 0;
      PRI_HMC_CFG_SB_DDR4_MR4                            : integer                                  := 0;
      PRI_HMC_CFG_SB_DDR4_MR5                            : integer                                  := 0;
      PRI_HMC_CFG_DDR4_MPS_ADDR_MIRROR                   : integer                                  := 0;
      PRI_HMC_CFG_MEM_IF_COLADDR_WIDTH                   : string                                   := "";
      PRI_HMC_CFG_MEM_IF_ROWADDR_WIDTH                   : string                                   := "";
      PRI_HMC_CFG_MEM_IF_BANKADDR_WIDTH                  : string                                   := "";
      PRI_HMC_CFG_MEM_IF_BGADDR_WIDTH                    : string                                   := "";
      PRI_HMC_CFG_LOCAL_IF_CS_WIDTH                      : string                                   := "";
      PRI_HMC_CFG_ADDR_ORDER                             : string                                   := "";
      PRI_HMC_CFG_ACT_TO_RDWR                            : integer                                  := 0;
      PRI_HMC_CFG_ACT_TO_PCH                             : integer                                  := 0;
      PRI_HMC_CFG_ACT_TO_ACT                             : integer                                  := 0;
      PRI_HMC_CFG_ACT_TO_ACT_DIFF_BANK                   : integer                                  := 0;
      PRI_HMC_CFG_ACT_TO_ACT_DIFF_BG                     : integer                                  := 0;
      PRI_HMC_CFG_RD_TO_RD                               : integer                                  := 0;
      PRI_HMC_CFG_RD_TO_RD_DIFF_CHIP                     : integer                                  := 0;
      PRI_HMC_CFG_RD_TO_RD_DIFF_BG                       : integer                                  := 0;
      PRI_HMC_CFG_RD_TO_WR                               : integer                                  := 0;
      PRI_HMC_CFG_RD_TO_WR_DIFF_CHIP                     : integer                                  := 0;
      PRI_HMC_CFG_RD_TO_WR_DIFF_BG                       : integer                                  := 0;
      PRI_HMC_CFG_RD_TO_PCH                              : integer                                  := 0;
      PRI_HMC_CFG_RD_AP_TO_VALID                         : integer                                  := 0;
      PRI_HMC_CFG_WR_TO_WR                               : integer                                  := 0;
      PRI_HMC_CFG_WR_TO_WR_DIFF_CHIP                     : integer                                  := 0;
      PRI_HMC_CFG_WR_TO_WR_DIFF_BG                       : integer                                  := 0;
      PRI_HMC_CFG_WR_TO_RD                               : integer                                  := 0;
      PRI_HMC_CFG_WR_TO_RD_DIFF_CHIP                     : integer                                  := 0;
      PRI_HMC_CFG_WR_TO_RD_DIFF_BG                       : integer                                  := 0;
      PRI_HMC_CFG_WR_TO_PCH                              : integer                                  := 0;
      PRI_HMC_CFG_WR_AP_TO_VALID                         : integer                                  := 0;
      PRI_HMC_CFG_PCH_TO_VALID                           : integer                                  := 0;
      PRI_HMC_CFG_PCH_ALL_TO_VALID                       : integer                                  := 0;
      PRI_HMC_CFG_ARF_TO_VALID                           : integer                                  := 0;
      PRI_HMC_CFG_PDN_TO_VALID                           : integer                                  := 0;
      PRI_HMC_CFG_SRF_TO_VALID                           : integer                                  := 0;
      PRI_HMC_CFG_SRF_TO_ZQ_CAL                          : integer                                  := 0;
      PRI_HMC_CFG_ARF_PERIOD                             : integer                                  := 0;
      PRI_HMC_CFG_PDN_PERIOD                             : integer                                  := 0;
      PRI_HMC_CFG_ZQCL_TO_VALID                          : integer                                  := 0;
      PRI_HMC_CFG_ZQCS_TO_VALID                          : integer                                  := 0;
      PRI_HMC_CFG_MRS_TO_VALID                           : integer                                  := 0;
      PRI_HMC_CFG_MPS_TO_VALID                           : integer                                  := 0;
      PRI_HMC_CFG_MRR_TO_VALID                           : integer                                  := 0;
      PRI_HMC_CFG_MPR_TO_VALID                           : integer                                  := 0;
      PRI_HMC_CFG_MPS_EXIT_CS_TO_CKE                     : integer                                  := 0;
      PRI_HMC_CFG_MPS_EXIT_CKE_TO_CS                     : integer                                  := 0;
      PRI_HMC_CFG_RLD3_MULTIBANK_REF_DELAY               : integer                                  := 0;
      PRI_HMC_CFG_MMR_CMD_TO_VALID                       : integer                                  := 0;
      PRI_HMC_CFG_4_ACT_TO_ACT                           : integer                                  := 0;
      PRI_HMC_CFG_16_ACT_TO_ACT                          : integer                                  := 0;
      SEC_HMC_CFG_ENABLE_ECC                             : string                                   := "";
      SEC_HMC_CFG_REORDER_DATA                           : string                                   := "";
      SEC_HMC_CFG_REORDER_READ                           : string                                   := "";
      SEC_HMC_CFG_REORDER_RDATA                          : string                                   := "";
      SEC_HMC_CFG_STARVE_LIMIT                           : integer                                  := 0;
      SEC_HMC_CFG_DQS_TRACKING_EN                        : string                                   := "";
      SEC_HMC_CFG_ARBITER_TYPE                           : string                                   := "";
      SEC_HMC_CFG_OPEN_PAGE_EN                           : string                                   := "";
      SEC_HMC_CFG_GEAR_DOWN_EN                           : string                                   := "";
      SEC_HMC_CFG_RLD3_MULTIBANK_MODE                    : string                                   := "";
      SEC_HMC_CFG_PING_PONG_MODE                         : string                                   := "";
      SEC_HMC_CFG_SLOT_ROTATE_EN                         : integer                                  := 0;
      SEC_HMC_CFG_SLOT_OFFSET                            : integer                                  := 0;
      SEC_HMC_CFG_COL_CMD_SLOT                           : integer                                  := 0;
      SEC_HMC_CFG_ROW_CMD_SLOT                           : integer                                  := 0;
      SEC_HMC_CFG_ENABLE_RC                              : string                                   := "";
      SEC_HMC_CFG_CS_TO_CHIP_MAPPING                     : integer                                  := 0;
      SEC_HMC_CFG_RB_RESERVED_ENTRY                      : integer                                  := 0;
      SEC_HMC_CFG_WB_RESERVED_ENTRY                      : integer                                  := 0;
      SEC_HMC_CFG_TCL                                    : integer                                  := 0;
      SEC_HMC_CFG_POWER_SAVING_EXIT_CYC                  : integer                                  := 0;
      SEC_HMC_CFG_MEM_CLK_DISABLE_ENTRY_CYC              : integer                                  := 0;
      SEC_HMC_CFG_WRITE_ODT_CHIP                         : integer                                  := 0;
      SEC_HMC_CFG_READ_ODT_CHIP                          : integer                                  := 0;
      SEC_HMC_CFG_WR_ODT_ON                              : integer                                  := 0;
      SEC_HMC_CFG_RD_ODT_ON                              : integer                                  := 0;
      SEC_HMC_CFG_WR_ODT_PERIOD                          : integer                                  := 0;
      SEC_HMC_CFG_RD_ODT_PERIOD                          : integer                                  := 0;
      SEC_HMC_CFG_RLD3_REFRESH_SEQ0                      : integer                                  := 0;
      SEC_HMC_CFG_RLD3_REFRESH_SEQ1                      : integer                                  := 0;
      SEC_HMC_CFG_RLD3_REFRESH_SEQ2                      : integer                                  := 0;
      SEC_HMC_CFG_RLD3_REFRESH_SEQ3                      : integer                                  := 0;
      SEC_HMC_CFG_SRF_ZQCAL_DISABLE                      : string                                   := "";
      SEC_HMC_CFG_MPS_ZQCAL_DISABLE                      : string                                   := "";
      SEC_HMC_CFG_MPS_DQSTRK_DISABLE                     : string                                   := "";
      SEC_HMC_CFG_SHORT_DQSTRK_CTRL_EN                   : string                                   := "";
      SEC_HMC_CFG_PERIOD_DQSTRK_CTRL_EN                  : string                                   := "";
      SEC_HMC_CFG_PERIOD_DQSTRK_INTERVAL                 : integer                                  := 0;
      SEC_HMC_CFG_DQSTRK_TO_VALID_LAST                   : integer                                  := 0;
      SEC_HMC_CFG_DQSTRK_TO_VALID                        : integer                                  := 0;
      SEC_HMC_CFG_RFSH_WARN_THRESHOLD                    : integer                                  := 0;
      SEC_HMC_CFG_SB_CG_DISABLE                          : string                                   := "";
      SEC_HMC_CFG_USER_RFSH_EN                           : string                                   := "";
      SEC_HMC_CFG_SRF_AUTOEXIT_EN                        : string                                   := "";
      SEC_HMC_CFG_SRF_ENTRY_EXIT_BLOCK                   : string                                   := "";
      SEC_HMC_CFG_SB_DDR4_MR3                            : integer                                  := 0;
      SEC_HMC_CFG_SB_DDR4_MR4                            : integer                                  := 0;
      SEC_HMC_CFG_SB_DDR4_MR5                            : integer                                  := 0;
      SEC_HMC_CFG_DDR4_MPS_ADDR_MIRROR                   : integer                                  := 0;
      SEC_HMC_CFG_MEM_IF_COLADDR_WIDTH                   : string                                   := "";
      SEC_HMC_CFG_MEM_IF_ROWADDR_WIDTH                   : string                                   := "";
      SEC_HMC_CFG_MEM_IF_BANKADDR_WIDTH                  : string                                   := "";
      SEC_HMC_CFG_MEM_IF_BGADDR_WIDTH                    : string                                   := "";
      SEC_HMC_CFG_LOCAL_IF_CS_WIDTH                      : string                                   := "";
      SEC_HMC_CFG_ADDR_ORDER                             : string                                   := "";
      SEC_HMC_CFG_ACT_TO_RDWR                            : integer                                  := 0;
      SEC_HMC_CFG_ACT_TO_PCH                             : integer                                  := 0;
      SEC_HMC_CFG_ACT_TO_ACT                             : integer                                  := 0;
      SEC_HMC_CFG_ACT_TO_ACT_DIFF_BANK                   : integer                                  := 0;
      SEC_HMC_CFG_ACT_TO_ACT_DIFF_BG                     : integer                                  := 0;
      SEC_HMC_CFG_RD_TO_RD                               : integer                                  := 0;
      SEC_HMC_CFG_RD_TO_RD_DIFF_CHIP                     : integer                                  := 0;
      SEC_HMC_CFG_RD_TO_RD_DIFF_BG                       : integer                                  := 0;
      SEC_HMC_CFG_RD_TO_WR                               : integer                                  := 0;
      SEC_HMC_CFG_RD_TO_WR_DIFF_CHIP                     : integer                                  := 0;
      SEC_HMC_CFG_RD_TO_WR_DIFF_BG                       : integer                                  := 0;
      SEC_HMC_CFG_RD_TO_PCH                              : integer                                  := 0;
      SEC_HMC_CFG_RD_AP_TO_VALID                         : integer                                  := 0;
      SEC_HMC_CFG_WR_TO_WR                               : integer                                  := 0;
      SEC_HMC_CFG_WR_TO_WR_DIFF_CHIP                     : integer                                  := 0;
      SEC_HMC_CFG_WR_TO_WR_DIFF_BG                       : integer                                  := 0;
      SEC_HMC_CFG_WR_TO_RD                               : integer                                  := 0;
      SEC_HMC_CFG_WR_TO_RD_DIFF_CHIP                     : integer                                  := 0;
      SEC_HMC_CFG_WR_TO_RD_DIFF_BG                       : integer                                  := 0;
      SEC_HMC_CFG_WR_TO_PCH                              : integer                                  := 0;
      SEC_HMC_CFG_WR_AP_TO_VALID                         : integer                                  := 0;
      SEC_HMC_CFG_PCH_TO_VALID                           : integer                                  := 0;
      SEC_HMC_CFG_PCH_ALL_TO_VALID                       : integer                                  := 0;
      SEC_HMC_CFG_ARF_TO_VALID                           : integer                                  := 0;
      SEC_HMC_CFG_PDN_TO_VALID                           : integer                                  := 0;
      SEC_HMC_CFG_SRF_TO_VALID                           : integer                                  := 0;
      SEC_HMC_CFG_SRF_TO_ZQ_CAL                          : integer                                  := 0;
      SEC_HMC_CFG_ARF_PERIOD                             : integer                                  := 0;
      SEC_HMC_CFG_PDN_PERIOD                             : integer                                  := 0;
      SEC_HMC_CFG_ZQCL_TO_VALID                          : integer                                  := 0;
      SEC_HMC_CFG_ZQCS_TO_VALID                          : integer                                  := 0;
      SEC_HMC_CFG_MRS_TO_VALID                           : integer                                  := 0;
      SEC_HMC_CFG_MPS_TO_VALID                           : integer                                  := 0;
      SEC_HMC_CFG_MRR_TO_VALID                           : integer                                  := 0;
      SEC_HMC_CFG_MPR_TO_VALID                           : integer                                  := 0;
      SEC_HMC_CFG_MPS_EXIT_CS_TO_CKE                     : integer                                  := 0;
      SEC_HMC_CFG_MPS_EXIT_CKE_TO_CS                     : integer                                  := 0;
      SEC_HMC_CFG_RLD3_MULTIBANK_REF_DELAY               : integer                                  := 0;
      SEC_HMC_CFG_MMR_CMD_TO_VALID                       : integer                                  := 0;
      SEC_HMC_CFG_4_ACT_TO_ACT                           : integer                                  := 0;
      SEC_HMC_CFG_16_ACT_TO_ACT                          : integer                                  := 0;
      PINS_PER_LANE                                      : integer                                  := 0;
      LANES_PER_TILE                                     : integer                                  := 0;
      OCT_CONTROL_WIDTH                                  : integer                                  := 0;
      PORT_MEM_CK_WIDTH                                  : integer                                  := 0;
      PORT_MEM_CK_PINLOC_0                               : integer                                  := 0;
      PORT_MEM_CK_PINLOC_1                               : integer                                  := 0;
      PORT_MEM_CK_PINLOC_2                               : integer                                  := 0;
      PORT_MEM_CK_PINLOC_3                               : integer                                  := 0;
      PORT_MEM_CK_PINLOC_4                               : integer                                  := 0;
      PORT_MEM_CK_PINLOC_5                               : integer                                  := 0;
      PORT_MEM_CK_PINLOC_AUTOGEN_WCNT                    : integer                                  := 0;
      PORT_MEM_CK_N_WIDTH                                : integer                                  := 0;
      PORT_MEM_CK_N_PINLOC_0                             : integer                                  := 0;
      PORT_MEM_CK_N_PINLOC_1                             : integer                                  := 0;
      PORT_MEM_CK_N_PINLOC_2                             : integer                                  := 0;
      PORT_MEM_CK_N_PINLOC_3                             : integer                                  := 0;
      PORT_MEM_CK_N_PINLOC_4                             : integer                                  := 0;
      PORT_MEM_CK_N_PINLOC_5                             : integer                                  := 0;
      PORT_MEM_CK_N_PINLOC_AUTOGEN_WCNT                  : integer                                  := 0;
      PORT_MEM_CK_BIDIR_WIDTH                            : integer                                  := 0;
      PORT_MEM_CK_BIDIR_PINLOC_0                         : integer                                  := 0;
      PORT_MEM_CK_BIDIR_PINLOC_1                         : integer                                  := 0;
      PORT_MEM_CK_BIDIR_PINLOC_2                         : integer                                  := 0;
      PORT_MEM_CK_BIDIR_PINLOC_3                         : integer                                  := 0;
      PORT_MEM_CK_BIDIR_PINLOC_4                         : integer                                  := 0;
      PORT_MEM_CK_BIDIR_PINLOC_5                         : integer                                  := 0;
      PORT_MEM_CK_BIDIR_PINLOC_AUTOGEN_WCNT              : integer                                  := 0;
      PORT_MEM_CK_BIDIR_N_WIDTH                          : integer                                  := 0;
      PORT_MEM_CK_BIDIR_N_PINLOC_0                       : integer                                  := 0;
      PORT_MEM_CK_BIDIR_N_PINLOC_1                       : integer                                  := 0;
      PORT_MEM_CK_BIDIR_N_PINLOC_2                       : integer                                  := 0;
      PORT_MEM_CK_BIDIR_N_PINLOC_3                       : integer                                  := 0;
      PORT_MEM_CK_BIDIR_N_PINLOC_4                       : integer                                  := 0;
      PORT_MEM_CK_BIDIR_N_PINLOC_5                       : integer                                  := 0;
      PORT_MEM_CK_BIDIR_N_PINLOC_AUTOGEN_WCNT            : integer                                  := 0;
      PORT_MEM_DK_WIDTH                                  : integer                                  := 0;
      PORT_MEM_DK_PINLOC_0                               : integer                                  := 0;
      PORT_MEM_DK_PINLOC_1                               : integer                                  := 0;
      PORT_MEM_DK_PINLOC_2                               : integer                                  := 0;
      PORT_MEM_DK_PINLOC_3                               : integer                                  := 0;
      PORT_MEM_DK_PINLOC_4                               : integer                                  := 0;
      PORT_MEM_DK_PINLOC_5                               : integer                                  := 0;
      PORT_MEM_DK_PINLOC_AUTOGEN_WCNT                    : integer                                  := 0;
      PORT_MEM_DK_N_WIDTH                                : integer                                  := 0;
      PORT_MEM_DK_N_PINLOC_0                             : integer                                  := 0;
      PORT_MEM_DK_N_PINLOC_1                             : integer                                  := 0;
      PORT_MEM_DK_N_PINLOC_2                             : integer                                  := 0;
      PORT_MEM_DK_N_PINLOC_3                             : integer                                  := 0;
      PORT_MEM_DK_N_PINLOC_4                             : integer                                  := 0;
      PORT_MEM_DK_N_PINLOC_5                             : integer                                  := 0;
      PORT_MEM_DK_N_PINLOC_AUTOGEN_WCNT                  : integer                                  := 0;
      PORT_MEM_DKA_WIDTH                                 : integer                                  := 0;
      PORT_MEM_DKA_PINLOC_0                              : integer                                  := 0;
      PORT_MEM_DKA_PINLOC_1                              : integer                                  := 0;
      PORT_MEM_DKA_PINLOC_2                              : integer                                  := 0;
      PORT_MEM_DKA_PINLOC_3                              : integer                                  := 0;
      PORT_MEM_DKA_PINLOC_4                              : integer                                  := 0;
      PORT_MEM_DKA_PINLOC_5                              : integer                                  := 0;
      PORT_MEM_DKA_PINLOC_AUTOGEN_WCNT                   : integer                                  := 0;
      PORT_MEM_DKA_N_WIDTH                               : integer                                  := 0;
      PORT_MEM_DKA_N_PINLOC_0                            : integer                                  := 0;
      PORT_MEM_DKA_N_PINLOC_1                            : integer                                  := 0;
      PORT_MEM_DKA_N_PINLOC_2                            : integer                                  := 0;
      PORT_MEM_DKA_N_PINLOC_3                            : integer                                  := 0;
      PORT_MEM_DKA_N_PINLOC_4                            : integer                                  := 0;
      PORT_MEM_DKA_N_PINLOC_5                            : integer                                  := 0;
      PORT_MEM_DKA_N_PINLOC_AUTOGEN_WCNT                 : integer                                  := 0;
      PORT_MEM_DKB_WIDTH                                 : integer                                  := 0;
      PORT_MEM_DKB_PINLOC_0                              : integer                                  := 0;
      PORT_MEM_DKB_PINLOC_1                              : integer                                  := 0;
      PORT_MEM_DKB_PINLOC_2                              : integer                                  := 0;
      PORT_MEM_DKB_PINLOC_3                              : integer                                  := 0;
      PORT_MEM_DKB_PINLOC_4                              : integer                                  := 0;
      PORT_MEM_DKB_PINLOC_5                              : integer                                  := 0;
      PORT_MEM_DKB_PINLOC_AUTOGEN_WCNT                   : integer                                  := 0;
      PORT_MEM_DKB_N_WIDTH                               : integer                                  := 0;
      PORT_MEM_DKB_N_PINLOC_0                            : integer                                  := 0;
      PORT_MEM_DKB_N_PINLOC_1                            : integer                                  := 0;
      PORT_MEM_DKB_N_PINLOC_2                            : integer                                  := 0;
      PORT_MEM_DKB_N_PINLOC_3                            : integer                                  := 0;
      PORT_MEM_DKB_N_PINLOC_4                            : integer                                  := 0;
      PORT_MEM_DKB_N_PINLOC_5                            : integer                                  := 0;
      PORT_MEM_DKB_N_PINLOC_AUTOGEN_WCNT                 : integer                                  := 0;
      PORT_MEM_K_WIDTH                                   : integer                                  := 0;
      PORT_MEM_K_PINLOC_0                                : integer                                  := 0;
      PORT_MEM_K_PINLOC_1                                : integer                                  := 0;
      PORT_MEM_K_PINLOC_2                                : integer                                  := 0;
      PORT_MEM_K_PINLOC_3                                : integer                                  := 0;
      PORT_MEM_K_PINLOC_4                                : integer                                  := 0;
      PORT_MEM_K_PINLOC_5                                : integer                                  := 0;
      PORT_MEM_K_PINLOC_AUTOGEN_WCNT                     : integer                                  := 0;
      PORT_MEM_K_N_WIDTH                                 : integer                                  := 0;
      PORT_MEM_K_N_PINLOC_0                              : integer                                  := 0;
      PORT_MEM_K_N_PINLOC_1                              : integer                                  := 0;
      PORT_MEM_K_N_PINLOC_2                              : integer                                  := 0;
      PORT_MEM_K_N_PINLOC_3                              : integer                                  := 0;
      PORT_MEM_K_N_PINLOC_4                              : integer                                  := 0;
      PORT_MEM_K_N_PINLOC_5                              : integer                                  := 0;
      PORT_MEM_K_N_PINLOC_AUTOGEN_WCNT                   : integer                                  := 0;
      PORT_MEM_A_WIDTH                                   : integer                                  := 0;
      PORT_MEM_A_PINLOC_0                                : integer                                  := 0;
      PORT_MEM_A_PINLOC_1                                : integer                                  := 0;
      PORT_MEM_A_PINLOC_2                                : integer                                  := 0;
      PORT_MEM_A_PINLOC_3                                : integer                                  := 0;
      PORT_MEM_A_PINLOC_4                                : integer                                  := 0;
      PORT_MEM_A_PINLOC_5                                : integer                                  := 0;
      PORT_MEM_A_PINLOC_6                                : integer                                  := 0;
      PORT_MEM_A_PINLOC_7                                : integer                                  := 0;
      PORT_MEM_A_PINLOC_8                                : integer                                  := 0;
      PORT_MEM_A_PINLOC_9                                : integer                                  := 0;
      PORT_MEM_A_PINLOC_10                               : integer                                  := 0;
      PORT_MEM_A_PINLOC_11                               : integer                                  := 0;
      PORT_MEM_A_PINLOC_12                               : integer                                  := 0;
      PORT_MEM_A_PINLOC_13                               : integer                                  := 0;
      PORT_MEM_A_PINLOC_14                               : integer                                  := 0;
      PORT_MEM_A_PINLOC_15                               : integer                                  := 0;
      PORT_MEM_A_PINLOC_16                               : integer                                  := 0;
      PORT_MEM_A_PINLOC_AUTOGEN_WCNT                     : integer                                  := 0;
      PORT_MEM_BA_WIDTH                                  : integer                                  := 0;
      PORT_MEM_BA_PINLOC_0                               : integer                                  := 0;
      PORT_MEM_BA_PINLOC_1                               : integer                                  := 0;
      PORT_MEM_BA_PINLOC_2                               : integer                                  := 0;
      PORT_MEM_BA_PINLOC_3                               : integer                                  := 0;
      PORT_MEM_BA_PINLOC_4                               : integer                                  := 0;
      PORT_MEM_BA_PINLOC_5                               : integer                                  := 0;
      PORT_MEM_BA_PINLOC_AUTOGEN_WCNT                    : integer                                  := 0;
      PORT_MEM_BG_WIDTH                                  : integer                                  := 0;
      PORT_MEM_BG_PINLOC_0                               : integer                                  := 0;
      PORT_MEM_BG_PINLOC_1                               : integer                                  := 0;
      PORT_MEM_BG_PINLOC_2                               : integer                                  := 0;
      PORT_MEM_BG_PINLOC_3                               : integer                                  := 0;
      PORT_MEM_BG_PINLOC_4                               : integer                                  := 0;
      PORT_MEM_BG_PINLOC_5                               : integer                                  := 0;
      PORT_MEM_BG_PINLOC_AUTOGEN_WCNT                    : integer                                  := 0;
      PORT_MEM_C_WIDTH                                   : integer                                  := 0;
      PORT_MEM_C_PINLOC_0                                : integer                                  := 0;
      PORT_MEM_C_PINLOC_1                                : integer                                  := 0;
      PORT_MEM_C_PINLOC_2                                : integer                                  := 0;
      PORT_MEM_C_PINLOC_3                                : integer                                  := 0;
      PORT_MEM_C_PINLOC_4                                : integer                                  := 0;
      PORT_MEM_C_PINLOC_5                                : integer                                  := 0;
      PORT_MEM_C_PINLOC_AUTOGEN_WCNT                     : integer                                  := 0;
      PORT_MEM_CKE_WIDTH                                 : integer                                  := 0;
      PORT_MEM_CKE_PINLOC_0                              : integer                                  := 0;
      PORT_MEM_CKE_PINLOC_1                              : integer                                  := 0;
      PORT_MEM_CKE_PINLOC_2                              : integer                                  := 0;
      PORT_MEM_CKE_PINLOC_3                              : integer                                  := 0;
      PORT_MEM_CKE_PINLOC_4                              : integer                                  := 0;
      PORT_MEM_CKE_PINLOC_5                              : integer                                  := 0;
      PORT_MEM_CKE_PINLOC_AUTOGEN_WCNT                   : integer                                  := 0;
      PORT_MEM_CS_N_WIDTH                                : integer                                  := 0;
      PORT_MEM_CS_N_PINLOC_0                             : integer                                  := 0;
      PORT_MEM_CS_N_PINLOC_1                             : integer                                  := 0;
      PORT_MEM_CS_N_PINLOC_2                             : integer                                  := 0;
      PORT_MEM_CS_N_PINLOC_3                             : integer                                  := 0;
      PORT_MEM_CS_N_PINLOC_4                             : integer                                  := 0;
      PORT_MEM_CS_N_PINLOC_5                             : integer                                  := 0;
      PORT_MEM_CS_N_PINLOC_AUTOGEN_WCNT                  : integer                                  := 0;
      PORT_MEM_RM_WIDTH                                  : integer                                  := 0;
      PORT_MEM_RM_PINLOC_0                               : integer                                  := 0;
      PORT_MEM_RM_PINLOC_1                               : integer                                  := 0;
      PORT_MEM_RM_PINLOC_2                               : integer                                  := 0;
      PORT_MEM_RM_PINLOC_3                               : integer                                  := 0;
      PORT_MEM_RM_PINLOC_4                               : integer                                  := 0;
      PORT_MEM_RM_PINLOC_5                               : integer                                  := 0;
      PORT_MEM_RM_PINLOC_AUTOGEN_WCNT                    : integer                                  := 0;
      PORT_MEM_ODT_WIDTH                                 : integer                                  := 0;
      PORT_MEM_ODT_PINLOC_0                              : integer                                  := 0;
      PORT_MEM_ODT_PINLOC_1                              : integer                                  := 0;
      PORT_MEM_ODT_PINLOC_2                              : integer                                  := 0;
      PORT_MEM_ODT_PINLOC_3                              : integer                                  := 0;
      PORT_MEM_ODT_PINLOC_4                              : integer                                  := 0;
      PORT_MEM_ODT_PINLOC_5                              : integer                                  := 0;
      PORT_MEM_ODT_PINLOC_AUTOGEN_WCNT                   : integer                                  := 0;
      PORT_MEM_REQ_N_WIDTH                               : integer                                  := 0;
      PORT_MEM_REQ_N_PINLOC_0                            : integer                                  := 0;
      PORT_MEM_REQ_N_PINLOC_1                            : integer                                  := 0;
      PORT_MEM_REQ_N_PINLOC_2                            : integer                                  := 0;
      PORT_MEM_REQ_N_PINLOC_3                            : integer                                  := 0;
      PORT_MEM_REQ_N_PINLOC_4                            : integer                                  := 0;
      PORT_MEM_REQ_N_PINLOC_5                            : integer                                  := 0;
      PORT_MEM_REQ_N_PINLOC_AUTOGEN_WCNT                 : integer                                  := 0;
      PORT_MEM_GNT_N_WIDTH                               : integer                                  := 0;
      PORT_MEM_GNT_N_PINLOC_0                            : integer                                  := 0;
      PORT_MEM_GNT_N_PINLOC_1                            : integer                                  := 0;
      PORT_MEM_GNT_N_PINLOC_2                            : integer                                  := 0;
      PORT_MEM_GNT_N_PINLOC_3                            : integer                                  := 0;
      PORT_MEM_GNT_N_PINLOC_4                            : integer                                  := 0;
      PORT_MEM_GNT_N_PINLOC_5                            : integer                                  := 0;
      PORT_MEM_GNT_N_PINLOC_AUTOGEN_WCNT                 : integer                                  := 0;
      PORT_MEM_ERR_N_WIDTH                               : integer                                  := 0;
      PORT_MEM_ERR_N_PINLOC_0                            : integer                                  := 0;
      PORT_MEM_ERR_N_PINLOC_1                            : integer                                  := 0;
      PORT_MEM_ERR_N_PINLOC_2                            : integer                                  := 0;
      PORT_MEM_ERR_N_PINLOC_3                            : integer                                  := 0;
      PORT_MEM_ERR_N_PINLOC_4                            : integer                                  := 0;
      PORT_MEM_ERR_N_PINLOC_5                            : integer                                  := 0;
      PORT_MEM_ERR_N_PINLOC_AUTOGEN_WCNT                 : integer                                  := 0;
      PORT_MEM_RAS_N_WIDTH                               : integer                                  := 0;
      PORT_MEM_RAS_N_PINLOC_0                            : integer                                  := 0;
      PORT_MEM_RAS_N_PINLOC_1                            : integer                                  := 0;
      PORT_MEM_RAS_N_PINLOC_AUTOGEN_WCNT                 : integer                                  := 0;
      PORT_MEM_CAS_N_WIDTH                               : integer                                  := 0;
      PORT_MEM_CAS_N_PINLOC_0                            : integer                                  := 0;
      PORT_MEM_CAS_N_PINLOC_1                            : integer                                  := 0;
      PORT_MEM_CAS_N_PINLOC_AUTOGEN_WCNT                 : integer                                  := 0;
      PORT_MEM_WE_N_WIDTH                                : integer                                  := 0;
      PORT_MEM_WE_N_PINLOC_0                             : integer                                  := 0;
      PORT_MEM_WE_N_PINLOC_1                             : integer                                  := 0;
      PORT_MEM_WE_N_PINLOC_AUTOGEN_WCNT                  : integer                                  := 0;
      PORT_MEM_RESET_N_WIDTH                             : integer                                  := 0;
      PORT_MEM_RESET_N_PINLOC_0                          : integer                                  := 0;
      PORT_MEM_RESET_N_PINLOC_1                          : integer                                  := 0;
      PORT_MEM_RESET_N_PINLOC_AUTOGEN_WCNT               : integer                                  := 0;
      PORT_MEM_ACT_N_WIDTH                               : integer                                  := 0;
      PORT_MEM_ACT_N_PINLOC_0                            : integer                                  := 0;
      PORT_MEM_ACT_N_PINLOC_1                            : integer                                  := 0;
      PORT_MEM_ACT_N_PINLOC_AUTOGEN_WCNT                 : integer                                  := 0;
      PORT_MEM_PAR_WIDTH                                 : integer                                  := 0;
      PORT_MEM_PAR_PINLOC_0                              : integer                                  := 0;
      PORT_MEM_PAR_PINLOC_1                              : integer                                  := 0;
      PORT_MEM_PAR_PINLOC_AUTOGEN_WCNT                   : integer                                  := 0;
      PORT_MEM_CA_WIDTH                                  : integer                                  := 0;
      PORT_MEM_CA_PINLOC_0                               : integer                                  := 0;
      PORT_MEM_CA_PINLOC_1                               : integer                                  := 0;
      PORT_MEM_CA_PINLOC_2                               : integer                                  := 0;
      PORT_MEM_CA_PINLOC_3                               : integer                                  := 0;
      PORT_MEM_CA_PINLOC_4                               : integer                                  := 0;
      PORT_MEM_CA_PINLOC_5                               : integer                                  := 0;
      PORT_MEM_CA_PINLOC_6                               : integer                                  := 0;
      PORT_MEM_CA_PINLOC_7                               : integer                                  := 0;
      PORT_MEM_CA_PINLOC_8                               : integer                                  := 0;
      PORT_MEM_CA_PINLOC_9                               : integer                                  := 0;
      PORT_MEM_CA_PINLOC_10                              : integer                                  := 0;
      PORT_MEM_CA_PINLOC_11                              : integer                                  := 0;
      PORT_MEM_CA_PINLOC_12                              : integer                                  := 0;
      PORT_MEM_CA_PINLOC_13                              : integer                                  := 0;
      PORT_MEM_CA_PINLOC_14                              : integer                                  := 0;
      PORT_MEM_CA_PINLOC_15                              : integer                                  := 0;
      PORT_MEM_CA_PINLOC_16                              : integer                                  := 0;
      PORT_MEM_CA_PINLOC_AUTOGEN_WCNT                    : integer                                  := 0;
      PORT_MEM_REF_N_WIDTH                               : integer                                  := 0;
      PORT_MEM_REF_N_PINLOC_0                            : integer                                  := 0;
      PORT_MEM_REF_N_PINLOC_AUTOGEN_WCNT                 : integer                                  := 0;
      PORT_MEM_WPS_N_WIDTH                               : integer                                  := 0;
      PORT_MEM_WPS_N_PINLOC_0                            : integer                                  := 0;
      PORT_MEM_WPS_N_PINLOC_AUTOGEN_WCNT                 : integer                                  := 0;
      PORT_MEM_RPS_N_WIDTH                               : integer                                  := 0;
      PORT_MEM_RPS_N_PINLOC_0                            : integer                                  := 0;
      PORT_MEM_RPS_N_PINLOC_AUTOGEN_WCNT                 : integer                                  := 0;
      PORT_MEM_DOFF_N_WIDTH                              : integer                                  := 0;
      PORT_MEM_DOFF_N_PINLOC_0                           : integer                                  := 0;
      PORT_MEM_DOFF_N_PINLOC_AUTOGEN_WCNT                : integer                                  := 0;
      PORT_MEM_LDA_N_WIDTH                               : integer                                  := 0;
      PORT_MEM_LDA_N_PINLOC_0                            : integer                                  := 0;
      PORT_MEM_LDA_N_PINLOC_AUTOGEN_WCNT                 : integer                                  := 0;
      PORT_MEM_LDB_N_WIDTH                               : integer                                  := 0;
      PORT_MEM_LDB_N_PINLOC_0                            : integer                                  := 0;
      PORT_MEM_LDB_N_PINLOC_AUTOGEN_WCNT                 : integer                                  := 0;
      PORT_MEM_RWA_N_WIDTH                               : integer                                  := 0;
      PORT_MEM_RWA_N_PINLOC_0                            : integer                                  := 0;
      PORT_MEM_RWA_N_PINLOC_AUTOGEN_WCNT                 : integer                                  := 0;
      PORT_MEM_RWB_N_WIDTH                               : integer                                  := 0;
      PORT_MEM_RWB_N_PINLOC_0                            : integer                                  := 0;
      PORT_MEM_RWB_N_PINLOC_AUTOGEN_WCNT                 : integer                                  := 0;
      PORT_MEM_LBK0_N_WIDTH                              : integer                                  := 0;
      PORT_MEM_LBK0_N_PINLOC_0                           : integer                                  := 0;
      PORT_MEM_LBK0_N_PINLOC_AUTOGEN_WCNT                : integer                                  := 0;
      PORT_MEM_LBK1_N_WIDTH                              : integer                                  := 0;
      PORT_MEM_LBK1_N_PINLOC_0                           : integer                                  := 0;
      PORT_MEM_LBK1_N_PINLOC_AUTOGEN_WCNT                : integer                                  := 0;
      PORT_MEM_CFG_N_WIDTH                               : integer                                  := 0;
      PORT_MEM_CFG_N_PINLOC_0                            : integer                                  := 0;
      PORT_MEM_CFG_N_PINLOC_AUTOGEN_WCNT                 : integer                                  := 0;
      PORT_MEM_AP_WIDTH                                  : integer                                  := 0;
      PORT_MEM_AP_PINLOC_0                               : integer                                  := 0;
      PORT_MEM_AP_PINLOC_AUTOGEN_WCNT                    : integer                                  := 0;
      PORT_MEM_AINV_WIDTH                                : integer                                  := 0;
      PORT_MEM_AINV_PINLOC_0                             : integer                                  := 0;
      PORT_MEM_AINV_PINLOC_AUTOGEN_WCNT                  : integer                                  := 0;
      PORT_MEM_DM_WIDTH                                  : integer                                  := 0;
      PORT_MEM_DM_PINLOC_0                               : integer                                  := 0;
      PORT_MEM_DM_PINLOC_1                               : integer                                  := 0;
      PORT_MEM_DM_PINLOC_2                               : integer                                  := 0;
      PORT_MEM_DM_PINLOC_3                               : integer                                  := 0;
      PORT_MEM_DM_PINLOC_4                               : integer                                  := 0;
      PORT_MEM_DM_PINLOC_5                               : integer                                  := 0;
      PORT_MEM_DM_PINLOC_6                               : integer                                  := 0;
      PORT_MEM_DM_PINLOC_7                               : integer                                  := 0;
      PORT_MEM_DM_PINLOC_8                               : integer                                  := 0;
      PORT_MEM_DM_PINLOC_9                               : integer                                  := 0;
      PORT_MEM_DM_PINLOC_10                              : integer                                  := 0;
      PORT_MEM_DM_PINLOC_11                              : integer                                  := 0;
      PORT_MEM_DM_PINLOC_12                              : integer                                  := 0;
      PORT_MEM_DM_PINLOC_AUTOGEN_WCNT                    : integer                                  := 0;
      PORT_MEM_BWS_N_WIDTH                               : integer                                  := 0;
      PORT_MEM_BWS_N_PINLOC_0                            : integer                                  := 0;
      PORT_MEM_BWS_N_PINLOC_1                            : integer                                  := 0;
      PORT_MEM_BWS_N_PINLOC_2                            : integer                                  := 0;
      PORT_MEM_BWS_N_PINLOC_AUTOGEN_WCNT                 : integer                                  := 0;
      PORT_MEM_D_WIDTH                                   : integer                                  := 0;
      PORT_MEM_D_PINLOC_0                                : integer                                  := 0;
      PORT_MEM_D_PINLOC_1                                : integer                                  := 0;
      PORT_MEM_D_PINLOC_2                                : integer                                  := 0;
      PORT_MEM_D_PINLOC_3                                : integer                                  := 0;
      PORT_MEM_D_PINLOC_4                                : integer                                  := 0;
      PORT_MEM_D_PINLOC_5                                : integer                                  := 0;
      PORT_MEM_D_PINLOC_6                                : integer                                  := 0;
      PORT_MEM_D_PINLOC_7                                : integer                                  := 0;
      PORT_MEM_D_PINLOC_8                                : integer                                  := 0;
      PORT_MEM_D_PINLOC_9                                : integer                                  := 0;
      PORT_MEM_D_PINLOC_10                               : integer                                  := 0;
      PORT_MEM_D_PINLOC_11                               : integer                                  := 0;
      PORT_MEM_D_PINLOC_12                               : integer                                  := 0;
      PORT_MEM_D_PINLOC_13                               : integer                                  := 0;
      PORT_MEM_D_PINLOC_14                               : integer                                  := 0;
      PORT_MEM_D_PINLOC_15                               : integer                                  := 0;
      PORT_MEM_D_PINLOC_16                               : integer                                  := 0;
      PORT_MEM_D_PINLOC_17                               : integer                                  := 0;
      PORT_MEM_D_PINLOC_18                               : integer                                  := 0;
      PORT_MEM_D_PINLOC_19                               : integer                                  := 0;
      PORT_MEM_D_PINLOC_20                               : integer                                  := 0;
      PORT_MEM_D_PINLOC_21                               : integer                                  := 0;
      PORT_MEM_D_PINLOC_22                               : integer                                  := 0;
      PORT_MEM_D_PINLOC_23                               : integer                                  := 0;
      PORT_MEM_D_PINLOC_24                               : integer                                  := 0;
      PORT_MEM_D_PINLOC_25                               : integer                                  := 0;
      PORT_MEM_D_PINLOC_26                               : integer                                  := 0;
      PORT_MEM_D_PINLOC_27                               : integer                                  := 0;
      PORT_MEM_D_PINLOC_28                               : integer                                  := 0;
      PORT_MEM_D_PINLOC_29                               : integer                                  := 0;
      PORT_MEM_D_PINLOC_30                               : integer                                  := 0;
      PORT_MEM_D_PINLOC_31                               : integer                                  := 0;
      PORT_MEM_D_PINLOC_32                               : integer                                  := 0;
      PORT_MEM_D_PINLOC_33                               : integer                                  := 0;
      PORT_MEM_D_PINLOC_34                               : integer                                  := 0;
      PORT_MEM_D_PINLOC_35                               : integer                                  := 0;
      PORT_MEM_D_PINLOC_36                               : integer                                  := 0;
      PORT_MEM_D_PINLOC_37                               : integer                                  := 0;
      PORT_MEM_D_PINLOC_38                               : integer                                  := 0;
      PORT_MEM_D_PINLOC_39                               : integer                                  := 0;
      PORT_MEM_D_PINLOC_40                               : integer                                  := 0;
      PORT_MEM_D_PINLOC_41                               : integer                                  := 0;
      PORT_MEM_D_PINLOC_42                               : integer                                  := 0;
      PORT_MEM_D_PINLOC_43                               : integer                                  := 0;
      PORT_MEM_D_PINLOC_44                               : integer                                  := 0;
      PORT_MEM_D_PINLOC_45                               : integer                                  := 0;
      PORT_MEM_D_PINLOC_46                               : integer                                  := 0;
      PORT_MEM_D_PINLOC_47                               : integer                                  := 0;
      PORT_MEM_D_PINLOC_48                               : integer                                  := 0;
      PORT_MEM_D_PINLOC_AUTOGEN_WCNT                     : integer                                  := 0;
      PORT_MEM_DQ_WIDTH                                  : integer                                  := 0;
      PORT_MEM_DQ_PINLOC_0                               : integer                                  := 0;
      PORT_MEM_DQ_PINLOC_1                               : integer                                  := 0;
      PORT_MEM_DQ_PINLOC_2                               : integer                                  := 0;
      PORT_MEM_DQ_PINLOC_3                               : integer                                  := 0;
      PORT_MEM_DQ_PINLOC_4                               : integer                                  := 0;
      PORT_MEM_DQ_PINLOC_5                               : integer                                  := 0;
      PORT_MEM_DQ_PINLOC_6                               : integer                                  := 0;
      PORT_MEM_DQ_PINLOC_7                               : integer                                  := 0;
      PORT_MEM_DQ_PINLOC_8                               : integer                                  := 0;
      PORT_MEM_DQ_PINLOC_9                               : integer                                  := 0;
      PORT_MEM_DQ_PINLOC_10                              : integer                                  := 0;
      PORT_MEM_DQ_PINLOC_11                              : integer                                  := 0;
      PORT_MEM_DQ_PINLOC_12                              : integer                                  := 0;
      PORT_MEM_DQ_PINLOC_13                              : integer                                  := 0;
      PORT_MEM_DQ_PINLOC_14                              : integer                                  := 0;
      PORT_MEM_DQ_PINLOC_15                              : integer                                  := 0;
      PORT_MEM_DQ_PINLOC_16                              : integer                                  := 0;
      PORT_MEM_DQ_PINLOC_17                              : integer                                  := 0;
      PORT_MEM_DQ_PINLOC_18                              : integer                                  := 0;
      PORT_MEM_DQ_PINLOC_19                              : integer                                  := 0;
      PORT_MEM_DQ_PINLOC_20                              : integer                                  := 0;
      PORT_MEM_DQ_PINLOC_21                              : integer                                  := 0;
      PORT_MEM_DQ_PINLOC_22                              : integer                                  := 0;
      PORT_MEM_DQ_PINLOC_23                              : integer                                  := 0;
      PORT_MEM_DQ_PINLOC_24                              : integer                                  := 0;
      PORT_MEM_DQ_PINLOC_25                              : integer                                  := 0;
      PORT_MEM_DQ_PINLOC_26                              : integer                                  := 0;
      PORT_MEM_DQ_PINLOC_27                              : integer                                  := 0;
      PORT_MEM_DQ_PINLOC_28                              : integer                                  := 0;
      PORT_MEM_DQ_PINLOC_29                              : integer                                  := 0;
      PORT_MEM_DQ_PINLOC_30                              : integer                                  := 0;
      PORT_MEM_DQ_PINLOC_31                              : integer                                  := 0;
      PORT_MEM_DQ_PINLOC_32                              : integer                                  := 0;
      PORT_MEM_DQ_PINLOC_33                              : integer                                  := 0;
      PORT_MEM_DQ_PINLOC_34                              : integer                                  := 0;
      PORT_MEM_DQ_PINLOC_35                              : integer                                  := 0;
      PORT_MEM_DQ_PINLOC_36                              : integer                                  := 0;
      PORT_MEM_DQ_PINLOC_37                              : integer                                  := 0;
      PORT_MEM_DQ_PINLOC_38                              : integer                                  := 0;
      PORT_MEM_DQ_PINLOC_39                              : integer                                  := 0;
      PORT_MEM_DQ_PINLOC_40                              : integer                                  := 0;
      PORT_MEM_DQ_PINLOC_41                              : integer                                  := 0;
      PORT_MEM_DQ_PINLOC_42                              : integer                                  := 0;
      PORT_MEM_DQ_PINLOC_43                              : integer                                  := 0;
      PORT_MEM_DQ_PINLOC_44                              : integer                                  := 0;
      PORT_MEM_DQ_PINLOC_45                              : integer                                  := 0;
      PORT_MEM_DQ_PINLOC_46                              : integer                                  := 0;
      PORT_MEM_DQ_PINLOC_47                              : integer                                  := 0;
      PORT_MEM_DQ_PINLOC_48                              : integer                                  := 0;
      PORT_MEM_DQ_PINLOC_AUTOGEN_WCNT                    : integer                                  := 0;
      PORT_MEM_DBI_N_WIDTH                               : integer                                  := 0;
      PORT_MEM_DBI_N_PINLOC_0                            : integer                                  := 0;
      PORT_MEM_DBI_N_PINLOC_1                            : integer                                  := 0;
      PORT_MEM_DBI_N_PINLOC_2                            : integer                                  := 0;
      PORT_MEM_DBI_N_PINLOC_3                            : integer                                  := 0;
      PORT_MEM_DBI_N_PINLOC_4                            : integer                                  := 0;
      PORT_MEM_DBI_N_PINLOC_5                            : integer                                  := 0;
      PORT_MEM_DBI_N_PINLOC_6                            : integer                                  := 0;
      PORT_MEM_DBI_N_PINLOC_AUTOGEN_WCNT                 : integer                                  := 0;
      PORT_MEM_DQA_WIDTH                                 : integer                                  := 0;
      PORT_MEM_DQA_PINLOC_0                              : integer                                  := 0;
      PORT_MEM_DQA_PINLOC_1                              : integer                                  := 0;
      PORT_MEM_DQA_PINLOC_2                              : integer                                  := 0;
      PORT_MEM_DQA_PINLOC_3                              : integer                                  := 0;
      PORT_MEM_DQA_PINLOC_4                              : integer                                  := 0;
      PORT_MEM_DQA_PINLOC_5                              : integer                                  := 0;
      PORT_MEM_DQA_PINLOC_6                              : integer                                  := 0;
      PORT_MEM_DQA_PINLOC_7                              : integer                                  := 0;
      PORT_MEM_DQA_PINLOC_8                              : integer                                  := 0;
      PORT_MEM_DQA_PINLOC_9                              : integer                                  := 0;
      PORT_MEM_DQA_PINLOC_10                             : integer                                  := 0;
      PORT_MEM_DQA_PINLOC_11                             : integer                                  := 0;
      PORT_MEM_DQA_PINLOC_12                             : integer                                  := 0;
      PORT_MEM_DQA_PINLOC_13                             : integer                                  := 0;
      PORT_MEM_DQA_PINLOC_14                             : integer                                  := 0;
      PORT_MEM_DQA_PINLOC_15                             : integer                                  := 0;
      PORT_MEM_DQA_PINLOC_16                             : integer                                  := 0;
      PORT_MEM_DQA_PINLOC_17                             : integer                                  := 0;
      PORT_MEM_DQA_PINLOC_18                             : integer                                  := 0;
      PORT_MEM_DQA_PINLOC_19                             : integer                                  := 0;
      PORT_MEM_DQA_PINLOC_20                             : integer                                  := 0;
      PORT_MEM_DQA_PINLOC_21                             : integer                                  := 0;
      PORT_MEM_DQA_PINLOC_22                             : integer                                  := 0;
      PORT_MEM_DQA_PINLOC_23                             : integer                                  := 0;
      PORT_MEM_DQA_PINLOC_24                             : integer                                  := 0;
      PORT_MEM_DQA_PINLOC_25                             : integer                                  := 0;
      PORT_MEM_DQA_PINLOC_26                             : integer                                  := 0;
      PORT_MEM_DQA_PINLOC_27                             : integer                                  := 0;
      PORT_MEM_DQA_PINLOC_28                             : integer                                  := 0;
      PORT_MEM_DQA_PINLOC_29                             : integer                                  := 0;
      PORT_MEM_DQA_PINLOC_30                             : integer                                  := 0;
      PORT_MEM_DQA_PINLOC_31                             : integer                                  := 0;
      PORT_MEM_DQA_PINLOC_32                             : integer                                  := 0;
      PORT_MEM_DQA_PINLOC_33                             : integer                                  := 0;
      PORT_MEM_DQA_PINLOC_34                             : integer                                  := 0;
      PORT_MEM_DQA_PINLOC_35                             : integer                                  := 0;
      PORT_MEM_DQA_PINLOC_36                             : integer                                  := 0;
      PORT_MEM_DQA_PINLOC_37                             : integer                                  := 0;
      PORT_MEM_DQA_PINLOC_38                             : integer                                  := 0;
      PORT_MEM_DQA_PINLOC_39                             : integer                                  := 0;
      PORT_MEM_DQA_PINLOC_40                             : integer                                  := 0;
      PORT_MEM_DQA_PINLOC_41                             : integer                                  := 0;
      PORT_MEM_DQA_PINLOC_42                             : integer                                  := 0;
      PORT_MEM_DQA_PINLOC_43                             : integer                                  := 0;
      PORT_MEM_DQA_PINLOC_44                             : integer                                  := 0;
      PORT_MEM_DQA_PINLOC_45                             : integer                                  := 0;
      PORT_MEM_DQA_PINLOC_46                             : integer                                  := 0;
      PORT_MEM_DQA_PINLOC_47                             : integer                                  := 0;
      PORT_MEM_DQA_PINLOC_48                             : integer                                  := 0;
      PORT_MEM_DQA_PINLOC_AUTOGEN_WCNT                   : integer                                  := 0;
      PORT_MEM_DQB_WIDTH                                 : integer                                  := 0;
      PORT_MEM_DQB_PINLOC_0                              : integer                                  := 0;
      PORT_MEM_DQB_PINLOC_1                              : integer                                  := 0;
      PORT_MEM_DQB_PINLOC_2                              : integer                                  := 0;
      PORT_MEM_DQB_PINLOC_3                              : integer                                  := 0;
      PORT_MEM_DQB_PINLOC_4                              : integer                                  := 0;
      PORT_MEM_DQB_PINLOC_5                              : integer                                  := 0;
      PORT_MEM_DQB_PINLOC_6                              : integer                                  := 0;
      PORT_MEM_DQB_PINLOC_7                              : integer                                  := 0;
      PORT_MEM_DQB_PINLOC_8                              : integer                                  := 0;
      PORT_MEM_DQB_PINLOC_9                              : integer                                  := 0;
      PORT_MEM_DQB_PINLOC_10                             : integer                                  := 0;
      PORT_MEM_DQB_PINLOC_11                             : integer                                  := 0;
      PORT_MEM_DQB_PINLOC_12                             : integer                                  := 0;
      PORT_MEM_DQB_PINLOC_13                             : integer                                  := 0;
      PORT_MEM_DQB_PINLOC_14                             : integer                                  := 0;
      PORT_MEM_DQB_PINLOC_15                             : integer                                  := 0;
      PORT_MEM_DQB_PINLOC_16                             : integer                                  := 0;
      PORT_MEM_DQB_PINLOC_17                             : integer                                  := 0;
      PORT_MEM_DQB_PINLOC_18                             : integer                                  := 0;
      PORT_MEM_DQB_PINLOC_19                             : integer                                  := 0;
      PORT_MEM_DQB_PINLOC_20                             : integer                                  := 0;
      PORT_MEM_DQB_PINLOC_21                             : integer                                  := 0;
      PORT_MEM_DQB_PINLOC_22                             : integer                                  := 0;
      PORT_MEM_DQB_PINLOC_23                             : integer                                  := 0;
      PORT_MEM_DQB_PINLOC_24                             : integer                                  := 0;
      PORT_MEM_DQB_PINLOC_25                             : integer                                  := 0;
      PORT_MEM_DQB_PINLOC_26                             : integer                                  := 0;
      PORT_MEM_DQB_PINLOC_27                             : integer                                  := 0;
      PORT_MEM_DQB_PINLOC_28                             : integer                                  := 0;
      PORT_MEM_DQB_PINLOC_29                             : integer                                  := 0;
      PORT_MEM_DQB_PINLOC_30                             : integer                                  := 0;
      PORT_MEM_DQB_PINLOC_31                             : integer                                  := 0;
      PORT_MEM_DQB_PINLOC_32                             : integer                                  := 0;
      PORT_MEM_DQB_PINLOC_33                             : integer                                  := 0;
      PORT_MEM_DQB_PINLOC_34                             : integer                                  := 0;
      PORT_MEM_DQB_PINLOC_35                             : integer                                  := 0;
      PORT_MEM_DQB_PINLOC_36                             : integer                                  := 0;
      PORT_MEM_DQB_PINLOC_37                             : integer                                  := 0;
      PORT_MEM_DQB_PINLOC_38                             : integer                                  := 0;
      PORT_MEM_DQB_PINLOC_39                             : integer                                  := 0;
      PORT_MEM_DQB_PINLOC_40                             : integer                                  := 0;
      PORT_MEM_DQB_PINLOC_41                             : integer                                  := 0;
      PORT_MEM_DQB_PINLOC_42                             : integer                                  := 0;
      PORT_MEM_DQB_PINLOC_43                             : integer                                  := 0;
      PORT_MEM_DQB_PINLOC_44                             : integer                                  := 0;
      PORT_MEM_DQB_PINLOC_45                             : integer                                  := 0;
      PORT_MEM_DQB_PINLOC_46                             : integer                                  := 0;
      PORT_MEM_DQB_PINLOC_47                             : integer                                  := 0;
      PORT_MEM_DQB_PINLOC_48                             : integer                                  := 0;
      PORT_MEM_DQB_PINLOC_AUTOGEN_WCNT                   : integer                                  := 0;
      PORT_MEM_DINVA_WIDTH                               : integer                                  := 0;
      PORT_MEM_DINVA_PINLOC_0                            : integer                                  := 0;
      PORT_MEM_DINVA_PINLOC_1                            : integer                                  := 0;
      PORT_MEM_DINVA_PINLOC_2                            : integer                                  := 0;
      PORT_MEM_DINVA_PINLOC_AUTOGEN_WCNT                 : integer                                  := 0;
      PORT_MEM_DINVB_WIDTH                               : integer                                  := 0;
      PORT_MEM_DINVB_PINLOC_0                            : integer                                  := 0;
      PORT_MEM_DINVB_PINLOC_1                            : integer                                  := 0;
      PORT_MEM_DINVB_PINLOC_2                            : integer                                  := 0;
      PORT_MEM_DINVB_PINLOC_AUTOGEN_WCNT                 : integer                                  := 0;
      PORT_MEM_Q_WIDTH                                   : integer                                  := 0;
      PORT_MEM_Q_PINLOC_0                                : integer                                  := 0;
      PORT_MEM_Q_PINLOC_1                                : integer                                  := 0;
      PORT_MEM_Q_PINLOC_2                                : integer                                  := 0;
      PORT_MEM_Q_PINLOC_3                                : integer                                  := 0;
      PORT_MEM_Q_PINLOC_4                                : integer                                  := 0;
      PORT_MEM_Q_PINLOC_5                                : integer                                  := 0;
      PORT_MEM_Q_PINLOC_6                                : integer                                  := 0;
      PORT_MEM_Q_PINLOC_7                                : integer                                  := 0;
      PORT_MEM_Q_PINLOC_8                                : integer                                  := 0;
      PORT_MEM_Q_PINLOC_9                                : integer                                  := 0;
      PORT_MEM_Q_PINLOC_10                               : integer                                  := 0;
      PORT_MEM_Q_PINLOC_11                               : integer                                  := 0;
      PORT_MEM_Q_PINLOC_12                               : integer                                  := 0;
      PORT_MEM_Q_PINLOC_13                               : integer                                  := 0;
      PORT_MEM_Q_PINLOC_14                               : integer                                  := 0;
      PORT_MEM_Q_PINLOC_15                               : integer                                  := 0;
      PORT_MEM_Q_PINLOC_16                               : integer                                  := 0;
      PORT_MEM_Q_PINLOC_17                               : integer                                  := 0;
      PORT_MEM_Q_PINLOC_18                               : integer                                  := 0;
      PORT_MEM_Q_PINLOC_19                               : integer                                  := 0;
      PORT_MEM_Q_PINLOC_20                               : integer                                  := 0;
      PORT_MEM_Q_PINLOC_21                               : integer                                  := 0;
      PORT_MEM_Q_PINLOC_22                               : integer                                  := 0;
      PORT_MEM_Q_PINLOC_23                               : integer                                  := 0;
      PORT_MEM_Q_PINLOC_24                               : integer                                  := 0;
      PORT_MEM_Q_PINLOC_25                               : integer                                  := 0;
      PORT_MEM_Q_PINLOC_26                               : integer                                  := 0;
      PORT_MEM_Q_PINLOC_27                               : integer                                  := 0;
      PORT_MEM_Q_PINLOC_28                               : integer                                  := 0;
      PORT_MEM_Q_PINLOC_29                               : integer                                  := 0;
      PORT_MEM_Q_PINLOC_30                               : integer                                  := 0;
      PORT_MEM_Q_PINLOC_31                               : integer                                  := 0;
      PORT_MEM_Q_PINLOC_32                               : integer                                  := 0;
      PORT_MEM_Q_PINLOC_33                               : integer                                  := 0;
      PORT_MEM_Q_PINLOC_34                               : integer                                  := 0;
      PORT_MEM_Q_PINLOC_35                               : integer                                  := 0;
      PORT_MEM_Q_PINLOC_36                               : integer                                  := 0;
      PORT_MEM_Q_PINLOC_37                               : integer                                  := 0;
      PORT_MEM_Q_PINLOC_38                               : integer                                  := 0;
      PORT_MEM_Q_PINLOC_39                               : integer                                  := 0;
      PORT_MEM_Q_PINLOC_40                               : integer                                  := 0;
      PORT_MEM_Q_PINLOC_41                               : integer                                  := 0;
      PORT_MEM_Q_PINLOC_42                               : integer                                  := 0;
      PORT_MEM_Q_PINLOC_43                               : integer                                  := 0;
      PORT_MEM_Q_PINLOC_44                               : integer                                  := 0;
      PORT_MEM_Q_PINLOC_45                               : integer                                  := 0;
      PORT_MEM_Q_PINLOC_46                               : integer                                  := 0;
      PORT_MEM_Q_PINLOC_47                               : integer                                  := 0;
      PORT_MEM_Q_PINLOC_48                               : integer                                  := 0;
      PORT_MEM_Q_PINLOC_AUTOGEN_WCNT                     : integer                                  := 0;
      PORT_MEM_DQS_WIDTH                                 : integer                                  := 0;
      PORT_MEM_DQS_PINLOC_0                              : integer                                  := 0;
      PORT_MEM_DQS_PINLOC_1                              : integer                                  := 0;
      PORT_MEM_DQS_PINLOC_2                              : integer                                  := 0;
      PORT_MEM_DQS_PINLOC_3                              : integer                                  := 0;
      PORT_MEM_DQS_PINLOC_4                              : integer                                  := 0;
      PORT_MEM_DQS_PINLOC_5                              : integer                                  := 0;
      PORT_MEM_DQS_PINLOC_6                              : integer                                  := 0;
      PORT_MEM_DQS_PINLOC_7                              : integer                                  := 0;
      PORT_MEM_DQS_PINLOC_8                              : integer                                  := 0;
      PORT_MEM_DQS_PINLOC_9                              : integer                                  := 0;
      PORT_MEM_DQS_PINLOC_10                             : integer                                  := 0;
      PORT_MEM_DQS_PINLOC_11                             : integer                                  := 0;
      PORT_MEM_DQS_PINLOC_12                             : integer                                  := 0;
      PORT_MEM_DQS_PINLOC_AUTOGEN_WCNT                   : integer                                  := 0;
      PORT_MEM_DQS_N_WIDTH                               : integer                                  := 0;
      PORT_MEM_DQS_N_PINLOC_0                            : integer                                  := 0;
      PORT_MEM_DQS_N_PINLOC_1                            : integer                                  := 0;
      PORT_MEM_DQS_N_PINLOC_2                            : integer                                  := 0;
      PORT_MEM_DQS_N_PINLOC_3                            : integer                                  := 0;
      PORT_MEM_DQS_N_PINLOC_4                            : integer                                  := 0;
      PORT_MEM_DQS_N_PINLOC_5                            : integer                                  := 0;
      PORT_MEM_DQS_N_PINLOC_6                            : integer                                  := 0;
      PORT_MEM_DQS_N_PINLOC_7                            : integer                                  := 0;
      PORT_MEM_DQS_N_PINLOC_8                            : integer                                  := 0;
      PORT_MEM_DQS_N_PINLOC_9                            : integer                                  := 0;
      PORT_MEM_DQS_N_PINLOC_10                           : integer                                  := 0;
      PORT_MEM_DQS_N_PINLOC_11                           : integer                                  := 0;
      PORT_MEM_DQS_N_PINLOC_12                           : integer                                  := 0;
      PORT_MEM_DQS_N_PINLOC_AUTOGEN_WCNT                 : integer                                  := 0;
      PORT_MEM_QK_WIDTH                                  : integer                                  := 0;
      PORT_MEM_QK_PINLOC_0                               : integer                                  := 0;
      PORT_MEM_QK_PINLOC_1                               : integer                                  := 0;
      PORT_MEM_QK_PINLOC_2                               : integer                                  := 0;
      PORT_MEM_QK_PINLOC_3                               : integer                                  := 0;
      PORT_MEM_QK_PINLOC_4                               : integer                                  := 0;
      PORT_MEM_QK_PINLOC_5                               : integer                                  := 0;
      PORT_MEM_QK_PINLOC_AUTOGEN_WCNT                    : integer                                  := 0;
      PORT_MEM_QK_N_WIDTH                                : integer                                  := 0;
      PORT_MEM_QK_N_PINLOC_0                             : integer                                  := 0;
      PORT_MEM_QK_N_PINLOC_1                             : integer                                  := 0;
      PORT_MEM_QK_N_PINLOC_2                             : integer                                  := 0;
      PORT_MEM_QK_N_PINLOC_3                             : integer                                  := 0;
      PORT_MEM_QK_N_PINLOC_4                             : integer                                  := 0;
      PORT_MEM_QK_N_PINLOC_5                             : integer                                  := 0;
      PORT_MEM_QK_N_PINLOC_AUTOGEN_WCNT                  : integer                                  := 0;
      PORT_MEM_QKA_WIDTH                                 : integer                                  := 0;
      PORT_MEM_QKA_PINLOC_0                              : integer                                  := 0;
      PORT_MEM_QKA_PINLOC_1                              : integer                                  := 0;
      PORT_MEM_QKA_PINLOC_2                              : integer                                  := 0;
      PORT_MEM_QKA_PINLOC_3                              : integer                                  := 0;
      PORT_MEM_QKA_PINLOC_4                              : integer                                  := 0;
      PORT_MEM_QKA_PINLOC_5                              : integer                                  := 0;
      PORT_MEM_QKA_PINLOC_AUTOGEN_WCNT                   : integer                                  := 0;
      PORT_MEM_QKA_N_WIDTH                               : integer                                  := 0;
      PORT_MEM_QKA_N_PINLOC_0                            : integer                                  := 0;
      PORT_MEM_QKA_N_PINLOC_1                            : integer                                  := 0;
      PORT_MEM_QKA_N_PINLOC_2                            : integer                                  := 0;
      PORT_MEM_QKA_N_PINLOC_3                            : integer                                  := 0;
      PORT_MEM_QKA_N_PINLOC_4                            : integer                                  := 0;
      PORT_MEM_QKA_N_PINLOC_5                            : integer                                  := 0;
      PORT_MEM_QKA_N_PINLOC_AUTOGEN_WCNT                 : integer                                  := 0;
      PORT_MEM_QKB_WIDTH                                 : integer                                  := 0;
      PORT_MEM_QKB_PINLOC_0                              : integer                                  := 0;
      PORT_MEM_QKB_PINLOC_1                              : integer                                  := 0;
      PORT_MEM_QKB_PINLOC_2                              : integer                                  := 0;
      PORT_MEM_QKB_PINLOC_3                              : integer                                  := 0;
      PORT_MEM_QKB_PINLOC_4                              : integer                                  := 0;
      PORT_MEM_QKB_PINLOC_5                              : integer                                  := 0;
      PORT_MEM_QKB_PINLOC_AUTOGEN_WCNT                   : integer                                  := 0;
      PORT_MEM_QKB_N_WIDTH                               : integer                                  := 0;
      PORT_MEM_QKB_N_PINLOC_0                            : integer                                  := 0;
      PORT_MEM_QKB_N_PINLOC_1                            : integer                                  := 0;
      PORT_MEM_QKB_N_PINLOC_2                            : integer                                  := 0;
      PORT_MEM_QKB_N_PINLOC_3                            : integer                                  := 0;
      PORT_MEM_QKB_N_PINLOC_4                            : integer                                  := 0;
      PORT_MEM_QKB_N_PINLOC_5                            : integer                                  := 0;
      PORT_MEM_QKB_N_PINLOC_AUTOGEN_WCNT                 : integer                                  := 0;
      PORT_MEM_CQ_WIDTH                                  : integer                                  := 0;
      PORT_MEM_CQ_PINLOC_0                               : integer                                  := 0;
      PORT_MEM_CQ_PINLOC_1                               : integer                                  := 0;
      PORT_MEM_CQ_PINLOC_AUTOGEN_WCNT                    : integer                                  := 0;
      PORT_MEM_CQ_N_WIDTH                                : integer                                  := 0;
      PORT_MEM_CQ_N_PINLOC_0                             : integer                                  := 0;
      PORT_MEM_CQ_N_PINLOC_1                             : integer                                  := 0;
      PORT_MEM_CQ_N_PINLOC_AUTOGEN_WCNT                  : integer                                  := 0;
      PORT_MEM_ALERT_N_WIDTH                             : integer                                  := 0;
      PORT_MEM_ALERT_N_PINLOC_0                          : integer                                  := 0;
      PORT_MEM_ALERT_N_PINLOC_1                          : integer                                  := 0;
      PORT_MEM_ALERT_N_PINLOC_AUTOGEN_WCNT               : integer                                  := 0;
      PORT_MEM_PE_N_WIDTH                                : integer                                  := 0;
      PORT_MEM_PE_N_PINLOC_0                             : integer                                  := 0;
      PORT_MEM_PE_N_PINLOC_1                             : integer                                  := 0;
      PORT_MEM_PE_N_PINLOC_AUTOGEN_WCNT                  : integer                                  := 0;
      PORT_CLKS_SHARING_MASTER_OUT_WIDTH                 : integer                                  := 0;
      PORT_CLKS_SHARING_SLAVE_IN_WIDTH                   : integer                                  := 0;
      PORT_CLKS_SHARING_SLAVE_OUT_WIDTH                  : integer                                  := 0;
      PORT_AFI_RLAT_WIDTH                                : integer                                  := 0;
      PORT_AFI_WLAT_WIDTH                                : integer                                  := 0;
      PORT_AFI_SEQ_BUSY_WIDTH                            : integer                                  := 0;
      PORT_AFI_ADDR_WIDTH                                : integer                                  := 0;
      PORT_AFI_BA_WIDTH                                  : integer                                  := 0;
      PORT_AFI_BG_WIDTH                                  : integer                                  := 0;
      PORT_AFI_C_WIDTH                                   : integer                                  := 0;
      PORT_AFI_CKE_WIDTH                                 : integer                                  := 0;
      PORT_AFI_CS_N_WIDTH                                : integer                                  := 0;
      PORT_AFI_RM_WIDTH                                  : integer                                  := 0;
      PORT_AFI_ODT_WIDTH                                 : integer                                  := 0;
      PORT_AFI_RAS_N_WIDTH                               : integer                                  := 0;
      PORT_AFI_CAS_N_WIDTH                               : integer                                  := 0;
      PORT_AFI_WE_N_WIDTH                                : integer                                  := 0;
      PORT_AFI_RST_N_WIDTH                               : integer                                  := 0;
      PORT_AFI_ACT_N_WIDTH                               : integer                                  := 0;
      PORT_AFI_REQ_N_WIDTH                               : integer                                  := 0;
      PORT_AFI_GNT_N_WIDTH                               : integer                                  := 0;
      PORT_AFI_ERR_N_WIDTH                               : integer                                  := 0;
      PORT_AFI_PAR_WIDTH                                 : integer                                  := 0;
      PORT_AFI_CA_WIDTH                                  : integer                                  := 0;
      PORT_AFI_REF_N_WIDTH                               : integer                                  := 0;
      PORT_AFI_WPS_N_WIDTH                               : integer                                  := 0;
      PORT_AFI_RPS_N_WIDTH                               : integer                                  := 0;
      PORT_AFI_DOFF_N_WIDTH                              : integer                                  := 0;
      PORT_AFI_LD_N_WIDTH                                : integer                                  := 0;
      PORT_AFI_RW_N_WIDTH                                : integer                                  := 0;
      PORT_AFI_LBK0_N_WIDTH                              : integer                                  := 0;
      PORT_AFI_LBK1_N_WIDTH                              : integer                                  := 0;
      PORT_AFI_CFG_N_WIDTH                               : integer                                  := 0;
      PORT_AFI_AP_WIDTH                                  : integer                                  := 0;
      PORT_AFI_AINV_WIDTH                                : integer                                  := 0;
      PORT_AFI_DM_WIDTH                                  : integer                                  := 0;
      PORT_AFI_DM_N_WIDTH                                : integer                                  := 0;
      PORT_AFI_BWS_N_WIDTH                               : integer                                  := 0;
      PORT_AFI_RDATA_DBI_N_WIDTH                         : integer                                  := 0;
      PORT_AFI_WDATA_DBI_N_WIDTH                         : integer                                  := 0;
      PORT_AFI_RDATA_DINV_WIDTH                          : integer                                  := 0;
      PORT_AFI_WDATA_DINV_WIDTH                          : integer                                  := 0;
      PORT_AFI_DQS_BURST_WIDTH                           : integer                                  := 0;
      PORT_AFI_WDATA_VALID_WIDTH                         : integer                                  := 0;
      PORT_AFI_WDATA_WIDTH                               : integer                                  := 0;
      PORT_AFI_RDATA_EN_FULL_WIDTH                       : integer                                  := 0;
      PORT_AFI_RDATA_WIDTH                               : integer                                  := 0;
      PORT_AFI_RDATA_VALID_WIDTH                         : integer                                  := 0;
      PORT_AFI_RRANK_WIDTH                               : integer                                  := 0;
      PORT_AFI_WRANK_WIDTH                               : integer                                  := 0;
      PORT_AFI_ALERT_N_WIDTH                             : integer                                  := 0;
      PORT_AFI_PE_N_WIDTH                                : integer                                  := 0;
      PORT_CTRL_AST_CMD_DATA_WIDTH                       : integer                                  := 0;
      PORT_CTRL_AST_WR_DATA_WIDTH                        : integer                                  := 0;
      PORT_CTRL_AST_RD_DATA_WIDTH                        : integer                                  := 0;
      PORT_CTRL_AMM_ADDRESS_WIDTH                        : integer                                  := 0;
      PORT_CTRL_AMM_RDATA_WIDTH                          : integer                                  := 0;
      PORT_CTRL_AMM_WDATA_WIDTH                          : integer                                  := 0;
      PORT_CTRL_AMM_BCOUNT_WIDTH                         : integer                                  := 0;
      PORT_CTRL_AMM_BYTEEN_WIDTH                         : integer                                  := 0;
      PORT_CTRL_USER_REFRESH_REQ_WIDTH                   : integer                                  := 0;
      PORT_CTRL_USER_REFRESH_BANK_WIDTH                  : integer                                  := 0;
      PORT_CTRL_SELF_REFRESH_REQ_WIDTH                   : integer                                  := 0;
      PORT_CTRL_ECC_WRITE_INFO_WIDTH                     : integer                                  := 0;
      PORT_CTRL_ECC_RDATA_ID_WIDTH                       : integer                                  := 0;
      PORT_CTRL_ECC_READ_INFO_WIDTH                      : integer                                  := 0;
      PORT_CTRL_ECC_CMD_INFO_WIDTH                       : integer                                  := 0;
      PORT_CTRL_ECC_WB_POINTER_WIDTH                     : integer                                  := 0;
      PORT_CTRL_MMR_SLAVE_ADDRESS_WIDTH                  : integer                                  := 0;
      PORT_CTRL_MMR_SLAVE_RDATA_WIDTH                    : integer                                  := 0;
      PORT_CTRL_MMR_SLAVE_WDATA_WIDTH                    : integer                                  := 0;
      PORT_CTRL_MMR_SLAVE_BCOUNT_WIDTH                   : integer                                  := 0;
      PORT_HPS_EMIF_H2E_WIDTH                            : integer                                  := 0;
      PORT_HPS_EMIF_E2H_WIDTH                            : integer                                  := 0;
      PORT_HPS_EMIF_H2E_GP_WIDTH                         : integer                                  := 0;
      PORT_HPS_EMIF_E2H_GP_WIDTH                         : integer                                  := 0;
      PORT_CAL_DEBUG_ADDRESS_WIDTH                       : integer                                  := 0;
      PORT_CAL_DEBUG_RDATA_WIDTH                         : integer                                  := 0;
      PORT_CAL_DEBUG_WDATA_WIDTH                         : integer                                  := 0;
      PORT_CAL_DEBUG_BYTEEN_WIDTH                        : integer                                  := 0;
      PORT_CAL_DEBUG_OUT_ADDRESS_WIDTH                   : integer                                  := 0;
      PORT_CAL_DEBUG_OUT_RDATA_WIDTH                     : integer                                  := 0;
      PORT_CAL_DEBUG_OUT_WDATA_WIDTH                     : integer                                  := 0;
      PORT_CAL_DEBUG_OUT_BYTEEN_WIDTH                    : integer                                  := 0;
      PORT_CAL_MASTER_ADDRESS_WIDTH                      : integer                                  := 0;
      PORT_CAL_MASTER_RDATA_WIDTH                        : integer                                  := 0;
      PORT_CAL_MASTER_WDATA_WIDTH                        : integer                                  := 0;
      PORT_CAL_MASTER_BYTEEN_WIDTH                       : integer                                  := 0;
      PORT_DFT_NF_IOAUX_PIO_IN_WIDTH                     : integer                                  := 0;
      PORT_DFT_NF_IOAUX_PIO_OUT_WIDTH                    : integer                                  := 0;
      PORT_DFT_NF_PA_DPRIO_REG_ADDR_WIDTH                : integer                                  := 0;
      PORT_DFT_NF_PA_DPRIO_WRITEDATA_WIDTH               : integer                                  := 0;
      PORT_DFT_NF_PA_DPRIO_READDATA_WIDTH                : integer                                  := 0;
      PORT_DFT_NF_PLL_CNTSEL_WIDTH                       : integer                                  := 0;
      PORT_DFT_NF_PLL_NUM_SHIFT_WIDTH                    : integer                                  := 0;
      PORT_DFT_NF_PLL_CORE_REFCLK_WIDTH                  : integer                                  := 0;
      PORT_DFT_NF_CORE_CLK_BUF_OUT_WIDTH                 : integer                                  := 0;
      PORT_DFT_NF_CORE_CLK_LOCKED_WIDTH                  : integer                                  := 0;
      PLL_VCO_FREQ_MHZ_INT                               : integer                                  := 0;
      PLL_VCO_TO_MEM_CLK_FREQ_RATIO                      : integer                                  := 0;
      PLL_PHY_CLK_VCO_PHASE                              : integer                                  := 0;
      PLL_VCO_FREQ_PS_STR                                : string                                   := "";
      PLL_REF_CLK_FREQ_PS_STR                            : string                                   := "";
      PLL_REF_CLK_FREQ_PS                                : integer                                  := 0;
      PLL_SIM_VCO_FREQ_PS                                : integer                                  := 0;
      PLL_SIM_PHYCLK_0_FREQ_PS                           : integer                                  := 0;
      PLL_SIM_PHYCLK_1_FREQ_PS                           : integer                                  := 0;
      PLL_SIM_PHYCLK_FB_FREQ_PS                          : integer                                  := 0;
      PLL_SIM_PHY_CLK_VCO_PHASE_PS                       : integer                                  := 0;
      PLL_SIM_CAL_SLAVE_CLK_FREQ_PS                      : integer                                  := 0;
      PLL_SIM_CAL_MASTER_CLK_FREQ_PS                     : integer                                  := 0;
      PLL_M_CNT_HIGH                                     : integer                                  := 0;
      PLL_M_CNT_LOW                                      : integer                                  := 0;
      PLL_N_CNT_HIGH                                     : integer                                  := 0;
      PLL_N_CNT_LOW                                      : integer                                  := 0;
      PLL_M_CNT_BYPASS_EN                                : string                                   := "";
      PLL_N_CNT_BYPASS_EN                                : string                                   := "";
      PLL_M_CNT_EVEN_DUTY_EN                             : string                                   := "";
      PLL_N_CNT_EVEN_DUTY_EN                             : string                                   := "";
      PLL_FBCLK_MUX_1                                    : string                                   := "";
      PLL_FBCLK_MUX_2                                    : string                                   := "";
      PLL_M_CNT_IN_SRC                                   : string                                   := "";
      PLL_CP_SETTING                                     : string                                   := "";
      PLL_BW_CTRL                                        : string                                   := "";
      PLL_BW_SEL                                         : string                                   := "";
      PLL_C_CNT_HIGH_0                                   : integer                                  := 0;
      PLL_C_CNT_LOW_0                                    : integer                                  := 0;
      PLL_C_CNT_PRST_0                                   : integer                                  := 0;
      PLL_C_CNT_PH_MUX_PRST_0                            : integer                                  := 0;
      PLL_C_CNT_BYPASS_EN_0                              : string                                   := "";
      PLL_C_CNT_EVEN_DUTY_EN_0                           : string                                   := "";
      PLL_C_CNT_FREQ_PS_STR_0                            : string                                   := "";
      PLL_C_CNT_PHASE_PS_STR_0                           : string                                   := "";
      PLL_C_CNT_DUTY_CYCLE_0                             : integer                                  := 0;
      PLL_C_CNT_OUT_EN_0                                 : string                                   := "";
      PLL_C_CNT_HIGH_1                                   : integer                                  := 0;
      PLL_C_CNT_LOW_1                                    : integer                                  := 0;
      PLL_C_CNT_PRST_1                                   : integer                                  := 0;
      PLL_C_CNT_PH_MUX_PRST_1                            : integer                                  := 0;
      PLL_C_CNT_BYPASS_EN_1                              : string                                   := "";
      PLL_C_CNT_EVEN_DUTY_EN_1                           : string                                   := "";
      PLL_C_CNT_FREQ_PS_STR_1                            : string                                   := "";
      PLL_C_CNT_PHASE_PS_STR_1                           : string                                   := "";
      PLL_C_CNT_DUTY_CYCLE_1                             : integer                                  := 0;
      PLL_C_CNT_OUT_EN_1                                 : string                                   := "";
      PLL_C_CNT_HIGH_2                                   : integer                                  := 0;
      PLL_C_CNT_LOW_2                                    : integer                                  := 0;
      PLL_C_CNT_PRST_2                                   : integer                                  := 0;
      PLL_C_CNT_PH_MUX_PRST_2                            : integer                                  := 0;
      PLL_C_CNT_BYPASS_EN_2                              : string                                   := "";
      PLL_C_CNT_EVEN_DUTY_EN_2                           : string                                   := "";
      PLL_C_CNT_FREQ_PS_STR_2                            : string                                   := "";
      PLL_C_CNT_PHASE_PS_STR_2                           : string                                   := "";
      PLL_C_CNT_DUTY_CYCLE_2                             : integer                                  := 0;
      PLL_C_CNT_OUT_EN_2                                 : string                                   := "";
      PLL_C_CNT_HIGH_3                                   : integer                                  := 0;
      PLL_C_CNT_LOW_3                                    : integer                                  := 0;
      PLL_C_CNT_PRST_3                                   : integer                                  := 0;
      PLL_C_CNT_PH_MUX_PRST_3                            : integer                                  := 0;
      PLL_C_CNT_BYPASS_EN_3                              : string                                   := "";
      PLL_C_CNT_EVEN_DUTY_EN_3                           : string                                   := "";
      PLL_C_CNT_FREQ_PS_STR_3                            : string                                   := "";
      PLL_C_CNT_PHASE_PS_STR_3                           : string                                   := "";
      PLL_C_CNT_DUTY_CYCLE_3                             : integer                                  := 0;
      PLL_C_CNT_OUT_EN_3                                 : string                                   := "";
      PLL_C_CNT_HIGH_4                                   : integer                                  := 0;
      PLL_C_CNT_LOW_4                                    : integer                                  := 0;
      PLL_C_CNT_PRST_4                                   : integer                                  := 0;
      PLL_C_CNT_PH_MUX_PRST_4                            : integer                                  := 0;
      PLL_C_CNT_BYPASS_EN_4                              : string                                   := "";
      PLL_C_CNT_EVEN_DUTY_EN_4                           : string                                   := "";
      PLL_C_CNT_FREQ_PS_STR_4                            : string                                   := "";
      PLL_C_CNT_PHASE_PS_STR_4                           : string                                   := "";
      PLL_C_CNT_DUTY_CYCLE_4                             : integer                                  := 0;
      PLL_C_CNT_OUT_EN_4                                 : string                                   := "";
      PLL_C_CNT_HIGH_5                                   : integer                                  := 0;
      PLL_C_CNT_LOW_5                                    : integer                                  := 0;
      PLL_C_CNT_PRST_5                                   : integer                                  := 0;
      PLL_C_CNT_PH_MUX_PRST_5                            : integer                                  := 0;
      PLL_C_CNT_BYPASS_EN_5                              : string                                   := "";
      PLL_C_CNT_EVEN_DUTY_EN_5                           : string                                   := "";
      PLL_C_CNT_FREQ_PS_STR_5                            : string                                   := "";
      PLL_C_CNT_PHASE_PS_STR_5                           : string                                   := "";
      PLL_C_CNT_DUTY_CYCLE_5                             : integer                                  := 0;
      PLL_C_CNT_OUT_EN_5                                 : string                                   := "";
      PLL_C_CNT_HIGH_6                                   : integer                                  := 0;
      PLL_C_CNT_LOW_6                                    : integer                                  := 0;
      PLL_C_CNT_PRST_6                                   : integer                                  := 0;
      PLL_C_CNT_PH_MUX_PRST_6                            : integer                                  := 0;
      PLL_C_CNT_BYPASS_EN_6                              : string                                   := "";
      PLL_C_CNT_EVEN_DUTY_EN_6                           : string                                   := "";
      PLL_C_CNT_FREQ_PS_STR_6                            : string                                   := "";
      PLL_C_CNT_PHASE_PS_STR_6                           : string                                   := "";
      PLL_C_CNT_DUTY_CYCLE_6                             : integer                                  := 0;
      PLL_C_CNT_OUT_EN_6                                 : string                                   := "";
      PLL_C_CNT_HIGH_7                                   : integer                                  := 0;
      PLL_C_CNT_LOW_7                                    : integer                                  := 0;
      PLL_C_CNT_PRST_7                                   : integer                                  := 0;
      PLL_C_CNT_PH_MUX_PRST_7                            : integer                                  := 0;
      PLL_C_CNT_BYPASS_EN_7                              : string                                   := "";
      PLL_C_CNT_EVEN_DUTY_EN_7                           : string                                   := "";
      PLL_C_CNT_FREQ_PS_STR_7                            : string                                   := "";
      PLL_C_CNT_PHASE_PS_STR_7                           : string                                   := "";
      PLL_C_CNT_DUTY_CYCLE_7                             : integer                                  := 0;
      PLL_C_CNT_OUT_EN_7                                 : string                                   := "";
      PLL_C_CNT_HIGH_8                                   : integer                                  := 0;
      PLL_C_CNT_LOW_8                                    : integer                                  := 0;
      PLL_C_CNT_PRST_8                                   : integer                                  := 0;
      PLL_C_CNT_PH_MUX_PRST_8                            : integer                                  := 0;
      PLL_C_CNT_BYPASS_EN_8                              : string                                   := "";
      PLL_C_CNT_EVEN_DUTY_EN_8                           : string                                   := "";
      PLL_C_CNT_FREQ_PS_STR_8                            : string                                   := "";
      PLL_C_CNT_PHASE_PS_STR_8                           : string                                   := "";
      PLL_C_CNT_DUTY_CYCLE_8                             : integer                                  := 0;
      PLL_C_CNT_OUT_EN_8                                 : string                                   := ""
   );
   port (
      global_reset_n                 : in    std_logic;
      pll_ref_clk                    : in    std_logic;
      pll_locked                     : out   std_logic;
      pll_extra_clk_0                : out   std_logic;
      pll_extra_clk_1                : out   std_logic;
      pll_extra_clk_2                : out   std_logic;
      pll_extra_clk_3                : out   std_logic;
      oct_rzqin                      : in    std_logic;
      mem_ck                         : out   std_logic_vector(0 downto 0);
      mem_ck_n                       : out   std_logic_vector(0 downto 0);
      mem_a                          : out   std_logic_vector(16 downto 0);
      mem_act_n                      : out   std_logic_vector(0 downto 0);
      mem_ba                         : out   std_logic_vector(1 downto 0);
      mem_bg                         : out   std_logic_vector(0 downto 0);
      mem_c                          : out   std_logic_vector(0 downto 0);
      mem_cke                        : out   std_logic_vector(0 downto 0);
      mem_cs_n                       : out   std_logic_vector(0 downto 0);
      mem_rm                         : out   std_logic_vector(0 downto 0);
      mem_odt                        : out   std_logic_vector(0 downto 0);
      mem_reset_n                    : out   std_logic_vector(0 downto 0);
      mem_par                        : out   std_logic_vector(0 downto 0);
      mem_alert_n                    : in    std_logic_vector(0 downto 0);
      mem_dqs                        : inout std_logic_vector(7 downto 0);
      mem_dqs_n                      : inout std_logic_vector(7 downto 0);
      mem_dq                         : inout std_logic_vector(63 downto 0);
      mem_dbi_n                      : inout std_logic_vector(7 downto 0);
      mem_ck_bidir                   : inout std_logic_vector(0 downto 0);
      mem_ck_bidir_n                 : inout std_logic_vector(0 downto 0);
      mem_dk                         : out   std_logic_vector(0 downto 0);
      mem_dk_n                       : out   std_logic_vector(0 downto 0);
      mem_dka                        : out   std_logic_vector(0 downto 0);
      mem_dka_n                      : out   std_logic_vector(0 downto 0);
      mem_dkb                        : out   std_logic_vector(0 downto 0);
      mem_dkb_n                      : out   std_logic_vector(0 downto 0);
      mem_k                          : out   std_logic_vector(0 downto 0);
      mem_k_n                        : out   std_logic_vector(0 downto 0);
      mem_req_n                      : in    std_logic_vector(0 downto 0);
      mem_gnt_n                      : out   std_logic_vector(0 downto 0);
      mem_err_n                      : in    std_logic_vector(0 downto 0);
      mem_ras_n                      : out   std_logic_vector(0 downto 0);
      mem_cas_n                      : out   std_logic_vector(0 downto 0);
      mem_we_n                       : out   std_logic_vector(0 downto 0);
      mem_ca                         : out   std_logic_vector(0 downto 0);
      mem_ref_n                      : out   std_logic_vector(0 downto 0);
      mem_wps_n                      : out   std_logic_vector(0 downto 0);
      mem_rps_n                      : out   std_logic_vector(0 downto 0);
      mem_doff_n                     : out   std_logic_vector(0 downto 0);
      mem_lda_n                      : out   std_logic_vector(0 downto 0);
      mem_ldb_n                      : out   std_logic_vector(0 downto 0);
      mem_rwa_n                      : out   std_logic_vector(0 downto 0);
      mem_rwb_n                      : out   std_logic_vector(0 downto 0);
      mem_lbk0_n                     : out   std_logic_vector(0 downto 0);
      mem_lbk1_n                     : out   std_logic_vector(0 downto 0);
      mem_cfg_n                      : out   std_logic_vector(0 downto 0);
      mem_ap                         : out   std_logic_vector(0 downto 0);
      mem_ainv                       : out   std_logic_vector(0 downto 0);
      mem_dm                         : out   std_logic_vector(0 downto 0);
      mem_bws_n                      : out   std_logic_vector(0 downto 0);
      mem_d                          : out   std_logic_vector(0 downto 0);
      mem_dqa                        : inout std_logic_vector(0 downto 0);
      mem_dqb                        : inout std_logic_vector(0 downto 0);
      mem_dinva                      : inout std_logic_vector(0 downto 0);
      mem_dinvb                      : inout std_logic_vector(0 downto 0);
      mem_q                          : in    std_logic_vector(0 downto 0);
      mem_qk                         : in    std_logic_vector(0 downto 0);
      mem_qk_n                       : in    std_logic_vector(0 downto 0);
      mem_qka                        : in    std_logic_vector(0 downto 0);
      mem_qka_n                      : in    std_logic_vector(0 downto 0);
      mem_qkb                        : in    std_logic_vector(0 downto 0);
      mem_qkb_n                      : in    std_logic_vector(0 downto 0);
      mem_cq                         : in    std_logic_vector(0 downto 0);
      mem_cq_n                       : in    std_logic_vector(0 downto 0);
      mem_pe_n                       : in    std_logic_vector(0 downto 0);
      local_cal_success              : out   std_logic;
      local_cal_fail                 : out   std_logic;
      vid_cal_done_persist           : in    std_logic;
      afi_reset_n                    : out   std_logic;
      afi_clk                        : out   std_logic;
      afi_half_clk                   : out   std_logic;
      emif_usr_reset_n               : out   std_logic;
      emif_usr_clk                   : out   std_logic;
      emif_usr_half_clk              : out   std_logic;
      emif_usr_reset_n_sec           : out   std_logic;
      emif_usr_clk_sec               : out   std_logic;
      emif_usr_half_clk_sec          : out   std_logic;
      cal_master_reset_n             : out   std_logic;
      cal_master_clk                 : out   std_logic;
      cal_slave_reset_n              : out   std_logic;
      cal_slave_clk                  : out   std_logic;
      cal_slave_reset_n_in           : in    std_logic;
      cal_slave_clk_in               : in    std_logic;
      cal_debug_reset_n              : in    std_logic;
      cal_debug_clk                  : in    std_logic;
      cal_debug_out_reset_n          : out   std_logic;
      cal_debug_out_clk              : out   std_logic;
      clks_sharing_master_out        : out   std_logic_vector(31 downto 0);
      clks_sharing_slave_in          : in    std_logic_vector(31 downto 0);
      clks_sharing_slave_out         : out   std_logic_vector(31 downto 0);
      afi_cal_success                : out   std_logic;
      afi_cal_fail                   : out   std_logic;
      afi_cal_req                    : in    std_logic;
      afi_rlat                       : out   std_logic_vector(5 downto 0);
      afi_wlat                       : out   std_logic_vector(5 downto 0);
      afi_seq_busy                   : out   std_logic_vector(3 downto 0);
      afi_ctl_refresh_done           : in    std_logic;
      afi_ctl_long_idle              : in    std_logic;
      afi_mps_req                    : in    std_logic;
      afi_mps_ack                    : out   std_logic;
      afi_addr                       : in    std_logic_vector(0 downto 0);
      afi_ba                         : in    std_logic_vector(0 downto 0);
      afi_bg                         : in    std_logic_vector(0 downto 0);
      afi_c                          : in    std_logic_vector(0 downto 0);
      afi_cke                        : in    std_logic_vector(0 downto 0);
      afi_cs_n                       : in    std_logic_vector(0 downto 0);
      afi_rm                         : in    std_logic_vector(0 downto 0);
      afi_odt                        : in    std_logic_vector(0 downto 0);
      afi_ras_n                      : in    std_logic_vector(0 downto 0);
      afi_cas_n                      : in    std_logic_vector(0 downto 0);
      afi_we_n                       : in    std_logic_vector(0 downto 0);
      afi_rst_n                      : in    std_logic_vector(0 downto 0);
      afi_act_n                      : in    std_logic_vector(0 downto 0);
      afi_req_n                      : out   std_logic_vector(0 downto 0);
      afi_gnt_n                      : in    std_logic_vector(0 downto 0);
      afi_err_n                      : out   std_logic_vector(0 downto 0);
      afi_par                        : in    std_logic_vector(0 downto 0);
      afi_ca                         : in    std_logic_vector(0 downto 0);
      afi_ref_n                      : in    std_logic_vector(0 downto 0);
      afi_wps_n                      : in    std_logic_vector(0 downto 0);
      afi_rps_n                      : in    std_logic_vector(0 downto 0);
      afi_doff_n                     : in    std_logic_vector(0 downto 0);
      afi_ld_n                       : in    std_logic_vector(0 downto 0);
      afi_rw_n                       : in    std_logic_vector(0 downto 0);
      afi_lbk0_n                     : in    std_logic_vector(0 downto 0);
      afi_lbk1_n                     : in    std_logic_vector(0 downto 0);
      afi_cfg_n                      : in    std_logic_vector(0 downto 0);
      afi_ap                         : in    std_logic_vector(0 downto 0);
      afi_ainv                       : in    std_logic_vector(0 downto 0);
      afi_dm                         : in    std_logic_vector(0 downto 0);
      afi_dm_n                       : in    std_logic_vector(0 downto 0);
      afi_bws_n                      : in    std_logic_vector(0 downto 0);
      afi_rdata_dbi_n                : out   std_logic_vector(0 downto 0);
      afi_wdata_dbi_n                : in    std_logic_vector(0 downto 0);
      afi_rdata_dinv                 : out   std_logic_vector(0 downto 0);
      afi_wdata_dinv                 : in    std_logic_vector(0 downto 0);
      afi_dqs_burst                  : in    std_logic_vector(0 downto 0);
      afi_wdata_valid                : in    std_logic_vector(0 downto 0);
      afi_wdata                      : in    std_logic_vector(0 downto 0);
      afi_rdata_en_full              : in    std_logic_vector(0 downto 0);
      afi_rdata                      : out   std_logic_vector(0 downto 0);
      afi_rdata_valid                : out   std_logic_vector(0 downto 0);
      afi_rrank                      : in    std_logic_vector(0 downto 0);
      afi_wrank                      : in    std_logic_vector(0 downto 0);
      afi_alert_n                    : out   std_logic_vector(0 downto 0);
      afi_pe_n                       : out   std_logic_vector(0 downto 0);
      ast_cmd_data_0                 : in    std_logic_vector(0 downto 0);
      ast_cmd_valid_0                : in    std_logic;
      ast_cmd_ready_0                : out   std_logic;
      ast_cmd_data_1                 : in    std_logic_vector(0 downto 0);
      ast_cmd_valid_1                : in    std_logic;
      ast_cmd_ready_1                : out   std_logic;
      ast_wr_data_0                  : in    std_logic_vector(0 downto 0);
      ast_wr_valid_0                 : in    std_logic;
      ast_wr_ready_0                 : out   std_logic;
      ast_wr_data_1                  : in    std_logic_vector(0 downto 0);
      ast_wr_valid_1                 : in    std_logic;
      ast_wr_ready_1                 : out   std_logic;
      ast_rd_data_0                  : out   std_logic_vector(0 downto 0);
      ast_rd_valid_0                 : out   std_logic;
      ast_rd_ready_0                 : in    std_logic;
      ast_rd_data_1                  : out   std_logic_vector(0 downto 0);
      ast_rd_valid_1                 : out   std_logic;
      ast_rd_ready_1                 : in    std_logic;
      amm_ready_0                    : out   std_logic;
      amm_read_0                     : in    std_logic;
      amm_write_0                    : in    std_logic;
      amm_address_0                  : in    std_logic_vector(24 downto 0);
      amm_readdata_0                 : out   std_logic_vector(511 downto 0);
      amm_writedata_0                : in    std_logic_vector(511 downto 0);
      amm_burstcount_0               : in    std_logic_vector(6 downto 0);
      amm_byteenable_0               : in    std_logic_vector(63 downto 0);
      amm_beginbursttransfer_0       : in    std_logic;
      amm_readdatavalid_0            : out   std_logic;
      amm_ready_1                    : out   std_logic;
      amm_read_1                     : in    std_logic;
      amm_write_1                    : in    std_logic;
      amm_address_1                  : in    std_logic_vector(24 downto 0);
      amm_readdata_1                 : out   std_logic_vector(511 downto 0);
      amm_writedata_1                : in    std_logic_vector(511 downto 0);
      amm_burstcount_1               : in    std_logic_vector(6 downto 0);
      amm_byteenable_1               : in    std_logic_vector(63 downto 0);
      amm_beginbursttransfer_1       : in    std_logic;
      amm_readdatavalid_1            : out   std_logic;
      ctrl_user_priority_hi_0        : in    std_logic;
      ctrl_user_priority_hi_1        : in    std_logic;
      ctrl_auto_precharge_req_0      : in    std_logic;
      ctrl_auto_precharge_req_1      : in    std_logic;
      ctrl_user_refresh_req          : in    std_logic_vector(3 downto 0);
      ctrl_user_refresh_bank         : in    std_logic_vector(15 downto 0);
      ctrl_user_refresh_ack          : out   std_logic;
      ctrl_self_refresh_req          : in    std_logic_vector(3 downto 0);
      ctrl_self_refresh_ack          : out   std_logic;
      ctrl_will_refresh              : out   std_logic;
      ctrl_deep_power_down_req       : in    std_logic;
      ctrl_deep_power_down_ack       : out   std_logic;
      ctrl_power_down_ack            : out   std_logic;
      ctrl_zq_cal_long_req           : in    std_logic;
      ctrl_zq_cal_short_req          : in    std_logic;
      ctrl_zq_cal_ack                : out   std_logic;
      ctrl_ecc_write_info_0          : in    std_logic_vector(14 downto 0);
      ctrl_ecc_rdata_id_0            : out   std_logic_vector(12 downto 0);
      ctrl_ecc_read_info_0           : out   std_logic_vector(2 downto 0);
      ctrl_ecc_cmd_info_0            : out   std_logic_vector(2 downto 0);
      ctrl_ecc_idle_0                : out   std_logic;
      ctrl_ecc_wr_pointer_info_0     : out   std_logic_vector(11 downto 0);
      ctrl_ecc_write_info_1          : in    std_logic_vector(14 downto 0);
      ctrl_ecc_rdata_id_1            : out   std_logic_vector(12 downto 0);
      ctrl_ecc_read_info_1           : out   std_logic_vector(2 downto 0);
      ctrl_ecc_cmd_info_1            : out   std_logic_vector(2 downto 0);
      ctrl_ecc_idle_1                : out   std_logic;
      ctrl_ecc_wr_pointer_info_1     : out   std_logic_vector(11 downto 0);
      mmr_slave_waitrequest_0        : out   std_logic;
      mmr_slave_read_0               : in    std_logic;
      mmr_slave_write_0              : in    std_logic;
      mmr_slave_address_0            : in    std_logic_vector(9 downto 0);
      mmr_slave_readdata_0           : out   std_logic_vector(31 downto 0);
      mmr_slave_writedata_0          : in    std_logic_vector(31 downto 0);
      mmr_slave_burstcount_0         : in    std_logic_vector(1 downto 0);
      mmr_slave_beginbursttransfer_0 : in    std_logic;
      mmr_slave_readdatavalid_0      : out   std_logic;
      mmr_slave_waitrequest_1        : out   std_logic;
      mmr_slave_read_1               : in    std_logic;
      mmr_slave_write_1              : in    std_logic;
      mmr_slave_address_1            : in    std_logic_vector(9 downto 0);
      mmr_slave_readdata_1           : out   std_logic_vector(31 downto 0);
      mmr_slave_writedata_1          : in    std_logic_vector(31 downto 0);
      mmr_slave_burstcount_1         : in    std_logic_vector(1 downto 0);
      mmr_slave_beginbursttransfer_1 : in    std_logic;
      mmr_slave_readdatavalid_1      : out   std_logic;
      hps_to_emif                    : in    std_logic_vector(4095 downto 0);
      emif_to_hps                    : out   std_logic_vector(4095 downto 0);
      hps_to_emif_gp                 : in    std_logic_vector(1 downto 0);
      emif_to_hps_gp                 : out   std_logic_vector(0 downto 0);
      cal_debug_waitrequest          : out   std_logic;
      cal_debug_read                 : in    std_logic;
      cal_debug_write                : in    std_logic;
      cal_debug_addr                 : in    std_logic_vector(23 downto 0);
      cal_debug_read_data            : out   std_logic_vector(31 downto 0);
      cal_debug_write_data           : in    std_logic_vector(31 downto 0);
      cal_debug_byteenable           : in    std_logic_vector(3 downto 0);
      cal_debug_read_data_valid      : out   std_logic;
      cal_debug_out_waitrequest      : in    std_logic;
      cal_debug_out_read             : out   std_logic;
      cal_debug_out_write            : out   std_logic;
      cal_debug_out_addr             : out   std_logic_vector(23 downto 0);
      cal_debug_out_read_data        : in    std_logic_vector(31 downto 0);
      cal_debug_out_write_data       : out   std_logic_vector(31 downto 0);
      cal_debug_out_byteenable       : out   std_logic_vector(3 downto 0);
      cal_debug_out_read_data_valid  : in    std_logic;
      cal_master_waitrequest         : in    std_logic;
      cal_master_read                : out   std_logic;
      cal_master_write               : out   std_logic;
      cal_master_addr                : out   std_logic_vector(15 downto 0);
      cal_master_read_data           : in    std_logic_vector(31 downto 0);
      cal_master_write_data          : out   std_logic_vector(31 downto 0);
      cal_master_byteenable          : out   std_logic_vector(3 downto 0);
      cal_master_read_data_valid     : in    std_logic;
      cal_master_burstcount          : out   std_logic;
      cal_master_debugaccess         : out   std_logic;
      ioaux_pio_in                   : in    std_logic_vector(7 downto 0);
      ioaux_pio_out                  : out   std_logic_vector(7 downto 0);
      pa_dprio_clk                   : in    std_logic;
      pa_dprio_read                  : in    std_logic;
      pa_dprio_reg_addr              : in    std_logic_vector(8 downto 0);
      pa_dprio_rst_n                 : in    std_logic;
      pa_dprio_write                 : in    std_logic;
      pa_dprio_writedata             : in    std_logic_vector(7 downto 0);
      pa_dprio_block_select          : out   std_logic;
      pa_dprio_readdata              : out   std_logic_vector(7 downto 0);
      pll_phase_en                   : in    std_logic;
      pll_up_dn                      : in    std_logic;
      pll_cnt_sel                    : in    std_logic_vector(3 downto 0);
      pll_num_phase_shifts           : in    std_logic_vector(2 downto 0);
      pll_phase_done                 : out   std_logic;
      pll_core_refclk                : in    std_logic_vector(0 downto 0);
      dft_core_clk_buf_out           : out   std_logic_vector(1 downto 0);
      dft_core_clk_locked            : out   std_logic_vector(1 downto 0)
   );
end entity SDRAM_altera_emif_arch_nf_191_65stwui;

architecture rtl of SDRAM_altera_emif_arch_nf_191_65stwui is
   component SDRAM_altera_emif_arch_nf_191_65stwui_top is
      generic (
         PROTOCOL_ENUM                                      : string                                   := "";
         PHY_TARGET_IS_ES                                   : boolean                                  := false;
         PHY_TARGET_IS_ES2                                  : boolean                                  := false;
         PHY_TARGET_IS_PRODUCTION                           : boolean                                  := false;
         PHY_CONFIG_ENUM                                    : string                                   := "";
         PHY_PING_PONG_EN                                   : boolean                                  := false;
         PHY_DLL_CORE_UPDN_EN                               : boolean                                  := false;
         PHY_CORE_CLKS_SHARING_ENUM                         : string                                   := "";
         PHY_CALIBRATED_OCT                                 : boolean                                  := false;
         PHY_AC_CALIBRATED_OCT                              : boolean                                  := false;
         PHY_CK_CALIBRATED_OCT                              : boolean                                  := false;
         PHY_DATA_CALIBRATED_OCT                            : boolean                                  := false;
         PHY_HPS_ENABLE_EARLY_RELEASE                       : boolean                                  := false;
         PLL_NUM_OF_EXTRA_CLKS                              : integer                                  := 0;
         MEM_FORMAT_ENUM                                    : string                                   := "";
         MEM_BURST_LENGTH                                   : integer                                  := 0;
         MEM_DATA_MASK_EN                                   : boolean                                  := false;
         MEM_TTL_DATA_WIDTH                                 : integer                                  := 0;
         MEM_TTL_NUM_OF_READ_GROUPS                         : integer                                  := 0;
         MEM_TTL_NUM_OF_WRITE_GROUPS                        : integer                                  := 0;
         DIAG_SIM_REGTEST_MODE                              : boolean                                  := false;
         DIAG_SYNTH_FOR_SIM                                 : boolean                                  := false;
         DIAG_ECLIPSE_DEBUG                                 : boolean                                  := false;
         DIAG_EXPORT_VJI                                    : boolean                                  := false;
         DIAG_INTERFACE_ID                                  : integer                                  := 0;
         DIAG_USE_ABSTRACT_PHY                              : boolean                                  := false;
         DIAG_SIM_VERBOSE_LEVEL                             : integer                                  := 0;
         DIAG_FAST_SIM                                      : boolean                                  := false;
         SILICON_REV                                        : string                                   := "";
         IS_HPS                                             : boolean                                  := false;
         IS_VID                                             : boolean                                  := false;
         USER_CLK_RATIO                                     : integer                                  := 0;
         C2P_P2C_CLK_RATIO                                  : integer                                  := 0;
         PHY_HMC_CLK_RATIO                                  : integer                                  := 0;
         DIAG_ABSTRACT_PHY_WLAT                             : integer                                  := 0;
         DIAG_ABSTRACT_PHY_RLAT                             : integer                                  := 0;
         DIAG_CPA_OUT_1_EN                                  : boolean                                  := false;
         DIAG_USE_CPA_LOCK                                  : boolean                                  := false;
         DQS_BUS_MODE_ENUM                                  : string                                   := "";
         AC_PIN_MAP_SCHEME                                  : string                                   := "";
         NUM_OF_HMC_PORTS                                   : integer                                  := 0;
         HMC_AVL_PROTOCOL_ENUM                              : string                                   := "";
         HMC_CTRL_DIMM_TYPE                                 : string                                   := "";
         REGISTER_AFI                                       : boolean                                  := false;
         SEQ_SYNTH_CPU_CLK_DIVIDE                           : integer                                  := 0;
         SEQ_SYNTH_CAL_CLK_DIVIDE                           : integer                                  := 0;
         SEQ_SIM_CPU_CLK_DIVIDE                             : integer                                  := 0;
         SEQ_SIM_CAL_CLK_DIVIDE                             : integer                                  := 0;
         SEQ_SYNTH_OSC_FREQ_MHZ                             : integer                                  := 0;
         SEQ_SIM_OSC_FREQ_MHZ                               : integer                                  := 0;
         NUM_OF_RTL_TILES                                   : integer                                  := 0;
         PRI_RDATA_TILE_INDEX                               : integer                                  := 0;
         PRI_RDATA_LANE_INDEX                               : integer                                  := 0;
         PRI_WDATA_TILE_INDEX                               : integer                                  := 0;
         PRI_WDATA_LANE_INDEX                               : integer                                  := 0;
         PRI_AC_TILE_INDEX                                  : integer                                  := 0;
         SEC_RDATA_TILE_INDEX                               : integer                                  := 0;
         SEC_RDATA_LANE_INDEX                               : integer                                  := 0;
         SEC_WDATA_TILE_INDEX                               : integer                                  := 0;
         SEC_WDATA_LANE_INDEX                               : integer                                  := 0;
         SEC_AC_TILE_INDEX                                  : integer                                  := 0;
         LANES_USAGE_0                                      : integer                                  := 0;
         LANES_USAGE_1                                      : integer                                  := 0;
         LANES_USAGE_2                                      : integer                                  := 0;
         LANES_USAGE_3                                      : integer                                  := 0;
         LANES_USAGE_AUTOGEN_WCNT                           : integer                                  := 0;
         PINS_USAGE_0                                       : integer                                  := 0;
         PINS_USAGE_1                                       : integer                                  := 0;
         PINS_USAGE_2                                       : integer                                  := 0;
         PINS_USAGE_3                                       : integer                                  := 0;
         PINS_USAGE_4                                       : integer                                  := 0;
         PINS_USAGE_5                                       : integer                                  := 0;
         PINS_USAGE_6                                       : integer                                  := 0;
         PINS_USAGE_7                                       : integer                                  := 0;
         PINS_USAGE_8                                       : integer                                  := 0;
         PINS_USAGE_9                                       : integer                                  := 0;
         PINS_USAGE_10                                      : integer                                  := 0;
         PINS_USAGE_11                                      : integer                                  := 0;
         PINS_USAGE_12                                      : integer                                  := 0;
         PINS_USAGE_AUTOGEN_WCNT                            : integer                                  := 0;
         PINS_RATE_0                                        : integer                                  := 0;
         PINS_RATE_1                                        : integer                                  := 0;
         PINS_RATE_2                                        : integer                                  := 0;
         PINS_RATE_3                                        : integer                                  := 0;
         PINS_RATE_4                                        : integer                                  := 0;
         PINS_RATE_5                                        : integer                                  := 0;
         PINS_RATE_6                                        : integer                                  := 0;
         PINS_RATE_7                                        : integer                                  := 0;
         PINS_RATE_8                                        : integer                                  := 0;
         PINS_RATE_9                                        : integer                                  := 0;
         PINS_RATE_10                                       : integer                                  := 0;
         PINS_RATE_11                                       : integer                                  := 0;
         PINS_RATE_12                                       : integer                                  := 0;
         PINS_RATE_AUTOGEN_WCNT                             : integer                                  := 0;
         PINS_WDB_0                                         : integer                                  := 0;
         PINS_WDB_1                                         : integer                                  := 0;
         PINS_WDB_2                                         : integer                                  := 0;
         PINS_WDB_3                                         : integer                                  := 0;
         PINS_WDB_4                                         : integer                                  := 0;
         PINS_WDB_5                                         : integer                                  := 0;
         PINS_WDB_6                                         : integer                                  := 0;
         PINS_WDB_7                                         : integer                                  := 0;
         PINS_WDB_8                                         : integer                                  := 0;
         PINS_WDB_9                                         : integer                                  := 0;
         PINS_WDB_10                                        : integer                                  := 0;
         PINS_WDB_11                                        : integer                                  := 0;
         PINS_WDB_12                                        : integer                                  := 0;
         PINS_WDB_13                                        : integer                                  := 0;
         PINS_WDB_14                                        : integer                                  := 0;
         PINS_WDB_15                                        : integer                                  := 0;
         PINS_WDB_16                                        : integer                                  := 0;
         PINS_WDB_17                                        : integer                                  := 0;
         PINS_WDB_18                                        : integer                                  := 0;
         PINS_WDB_19                                        : integer                                  := 0;
         PINS_WDB_20                                        : integer                                  := 0;
         PINS_WDB_21                                        : integer                                  := 0;
         PINS_WDB_22                                        : integer                                  := 0;
         PINS_WDB_23                                        : integer                                  := 0;
         PINS_WDB_24                                        : integer                                  := 0;
         PINS_WDB_25                                        : integer                                  := 0;
         PINS_WDB_26                                        : integer                                  := 0;
         PINS_WDB_27                                        : integer                                  := 0;
         PINS_WDB_28                                        : integer                                  := 0;
         PINS_WDB_29                                        : integer                                  := 0;
         PINS_WDB_30                                        : integer                                  := 0;
         PINS_WDB_31                                        : integer                                  := 0;
         PINS_WDB_32                                        : integer                                  := 0;
         PINS_WDB_33                                        : integer                                  := 0;
         PINS_WDB_34                                        : integer                                  := 0;
         PINS_WDB_35                                        : integer                                  := 0;
         PINS_WDB_36                                        : integer                                  := 0;
         PINS_WDB_37                                        : integer                                  := 0;
         PINS_WDB_38                                        : integer                                  := 0;
         PINS_WDB_AUTOGEN_WCNT                              : integer                                  := 0;
         PINS_DATA_IN_MODE_0                                : integer                                  := 0;
         PINS_DATA_IN_MODE_1                                : integer                                  := 0;
         PINS_DATA_IN_MODE_2                                : integer                                  := 0;
         PINS_DATA_IN_MODE_3                                : integer                                  := 0;
         PINS_DATA_IN_MODE_4                                : integer                                  := 0;
         PINS_DATA_IN_MODE_5                                : integer                                  := 0;
         PINS_DATA_IN_MODE_6                                : integer                                  := 0;
         PINS_DATA_IN_MODE_7                                : integer                                  := 0;
         PINS_DATA_IN_MODE_8                                : integer                                  := 0;
         PINS_DATA_IN_MODE_9                                : integer                                  := 0;
         PINS_DATA_IN_MODE_10                               : integer                                  := 0;
         PINS_DATA_IN_MODE_11                               : integer                                  := 0;
         PINS_DATA_IN_MODE_12                               : integer                                  := 0;
         PINS_DATA_IN_MODE_13                               : integer                                  := 0;
         PINS_DATA_IN_MODE_14                               : integer                                  := 0;
         PINS_DATA_IN_MODE_15                               : integer                                  := 0;
         PINS_DATA_IN_MODE_16                               : integer                                  := 0;
         PINS_DATA_IN_MODE_17                               : integer                                  := 0;
         PINS_DATA_IN_MODE_18                               : integer                                  := 0;
         PINS_DATA_IN_MODE_19                               : integer                                  := 0;
         PINS_DATA_IN_MODE_20                               : integer                                  := 0;
         PINS_DATA_IN_MODE_21                               : integer                                  := 0;
         PINS_DATA_IN_MODE_22                               : integer                                  := 0;
         PINS_DATA_IN_MODE_23                               : integer                                  := 0;
         PINS_DATA_IN_MODE_24                               : integer                                  := 0;
         PINS_DATA_IN_MODE_25                               : integer                                  := 0;
         PINS_DATA_IN_MODE_26                               : integer                                  := 0;
         PINS_DATA_IN_MODE_27                               : integer                                  := 0;
         PINS_DATA_IN_MODE_28                               : integer                                  := 0;
         PINS_DATA_IN_MODE_29                               : integer                                  := 0;
         PINS_DATA_IN_MODE_30                               : integer                                  := 0;
         PINS_DATA_IN_MODE_31                               : integer                                  := 0;
         PINS_DATA_IN_MODE_32                               : integer                                  := 0;
         PINS_DATA_IN_MODE_33                               : integer                                  := 0;
         PINS_DATA_IN_MODE_34                               : integer                                  := 0;
         PINS_DATA_IN_MODE_35                               : integer                                  := 0;
         PINS_DATA_IN_MODE_36                               : integer                                  := 0;
         PINS_DATA_IN_MODE_37                               : integer                                  := 0;
         PINS_DATA_IN_MODE_38                               : integer                                  := 0;
         PINS_DATA_IN_MODE_AUTOGEN_WCNT                     : integer                                  := 0;
         PINS_C2L_DRIVEN_0                                  : integer                                  := 0;
         PINS_C2L_DRIVEN_1                                  : integer                                  := 0;
         PINS_C2L_DRIVEN_2                                  : integer                                  := 0;
         PINS_C2L_DRIVEN_3                                  : integer                                  := 0;
         PINS_C2L_DRIVEN_4                                  : integer                                  := 0;
         PINS_C2L_DRIVEN_5                                  : integer                                  := 0;
         PINS_C2L_DRIVEN_6                                  : integer                                  := 0;
         PINS_C2L_DRIVEN_7                                  : integer                                  := 0;
         PINS_C2L_DRIVEN_8                                  : integer                                  := 0;
         PINS_C2L_DRIVEN_9                                  : integer                                  := 0;
         PINS_C2L_DRIVEN_10                                 : integer                                  := 0;
         PINS_C2L_DRIVEN_11                                 : integer                                  := 0;
         PINS_C2L_DRIVEN_12                                 : integer                                  := 0;
         PINS_C2L_DRIVEN_AUTOGEN_WCNT                       : integer                                  := 0;
         PINS_DB_IN_BYPASS_0                                : integer                                  := 0;
         PINS_DB_IN_BYPASS_1                                : integer                                  := 0;
         PINS_DB_IN_BYPASS_2                                : integer                                  := 0;
         PINS_DB_IN_BYPASS_3                                : integer                                  := 0;
         PINS_DB_IN_BYPASS_4                                : integer                                  := 0;
         PINS_DB_IN_BYPASS_5                                : integer                                  := 0;
         PINS_DB_IN_BYPASS_6                                : integer                                  := 0;
         PINS_DB_IN_BYPASS_7                                : integer                                  := 0;
         PINS_DB_IN_BYPASS_8                                : integer                                  := 0;
         PINS_DB_IN_BYPASS_9                                : integer                                  := 0;
         PINS_DB_IN_BYPASS_10                               : integer                                  := 0;
         PINS_DB_IN_BYPASS_11                               : integer                                  := 0;
         PINS_DB_IN_BYPASS_12                               : integer                                  := 0;
         PINS_DB_IN_BYPASS_AUTOGEN_WCNT                     : integer                                  := 0;
         PINS_DB_OUT_BYPASS_0                               : integer                                  := 0;
         PINS_DB_OUT_BYPASS_1                               : integer                                  := 0;
         PINS_DB_OUT_BYPASS_2                               : integer                                  := 0;
         PINS_DB_OUT_BYPASS_3                               : integer                                  := 0;
         PINS_DB_OUT_BYPASS_4                               : integer                                  := 0;
         PINS_DB_OUT_BYPASS_5                               : integer                                  := 0;
         PINS_DB_OUT_BYPASS_6                               : integer                                  := 0;
         PINS_DB_OUT_BYPASS_7                               : integer                                  := 0;
         PINS_DB_OUT_BYPASS_8                               : integer                                  := 0;
         PINS_DB_OUT_BYPASS_9                               : integer                                  := 0;
         PINS_DB_OUT_BYPASS_10                              : integer                                  := 0;
         PINS_DB_OUT_BYPASS_11                              : integer                                  := 0;
         PINS_DB_OUT_BYPASS_12                              : integer                                  := 0;
         PINS_DB_OUT_BYPASS_AUTOGEN_WCNT                    : integer                                  := 0;
         PINS_DB_OE_BYPASS_0                                : integer                                  := 0;
         PINS_DB_OE_BYPASS_1                                : integer                                  := 0;
         PINS_DB_OE_BYPASS_2                                : integer                                  := 0;
         PINS_DB_OE_BYPASS_3                                : integer                                  := 0;
         PINS_DB_OE_BYPASS_4                                : integer                                  := 0;
         PINS_DB_OE_BYPASS_5                                : integer                                  := 0;
         PINS_DB_OE_BYPASS_6                                : integer                                  := 0;
         PINS_DB_OE_BYPASS_7                                : integer                                  := 0;
         PINS_DB_OE_BYPASS_8                                : integer                                  := 0;
         PINS_DB_OE_BYPASS_9                                : integer                                  := 0;
         PINS_DB_OE_BYPASS_10                               : integer                                  := 0;
         PINS_DB_OE_BYPASS_11                               : integer                                  := 0;
         PINS_DB_OE_BYPASS_12                               : integer                                  := 0;
         PINS_DB_OE_BYPASS_AUTOGEN_WCNT                     : integer                                  := 0;
         PINS_INVERT_WR_0                                   : integer                                  := 0;
         PINS_INVERT_WR_1                                   : integer                                  := 0;
         PINS_INVERT_WR_2                                   : integer                                  := 0;
         PINS_INVERT_WR_3                                   : integer                                  := 0;
         PINS_INVERT_WR_4                                   : integer                                  := 0;
         PINS_INVERT_WR_5                                   : integer                                  := 0;
         PINS_INVERT_WR_6                                   : integer                                  := 0;
         PINS_INVERT_WR_7                                   : integer                                  := 0;
         PINS_INVERT_WR_8                                   : integer                                  := 0;
         PINS_INVERT_WR_9                                   : integer                                  := 0;
         PINS_INVERT_WR_10                                  : integer                                  := 0;
         PINS_INVERT_WR_11                                  : integer                                  := 0;
         PINS_INVERT_WR_12                                  : integer                                  := 0;
         PINS_INVERT_WR_AUTOGEN_WCNT                        : integer                                  := 0;
         PINS_INVERT_OE_0                                   : integer                                  := 0;
         PINS_INVERT_OE_1                                   : integer                                  := 0;
         PINS_INVERT_OE_2                                   : integer                                  := 0;
         PINS_INVERT_OE_3                                   : integer                                  := 0;
         PINS_INVERT_OE_4                                   : integer                                  := 0;
         PINS_INVERT_OE_5                                   : integer                                  := 0;
         PINS_INVERT_OE_6                                   : integer                                  := 0;
         PINS_INVERT_OE_7                                   : integer                                  := 0;
         PINS_INVERT_OE_8                                   : integer                                  := 0;
         PINS_INVERT_OE_9                                   : integer                                  := 0;
         PINS_INVERT_OE_10                                  : integer                                  := 0;
         PINS_INVERT_OE_11                                  : integer                                  := 0;
         PINS_INVERT_OE_12                                  : integer                                  := 0;
         PINS_INVERT_OE_AUTOGEN_WCNT                        : integer                                  := 0;
         PINS_AC_HMC_DATA_OVERRIDE_ENA_0                    : integer                                  := 0;
         PINS_AC_HMC_DATA_OVERRIDE_ENA_1                    : integer                                  := 0;
         PINS_AC_HMC_DATA_OVERRIDE_ENA_2                    : integer                                  := 0;
         PINS_AC_HMC_DATA_OVERRIDE_ENA_3                    : integer                                  := 0;
         PINS_AC_HMC_DATA_OVERRIDE_ENA_4                    : integer                                  := 0;
         PINS_AC_HMC_DATA_OVERRIDE_ENA_5                    : integer                                  := 0;
         PINS_AC_HMC_DATA_OVERRIDE_ENA_6                    : integer                                  := 0;
         PINS_AC_HMC_DATA_OVERRIDE_ENA_7                    : integer                                  := 0;
         PINS_AC_HMC_DATA_OVERRIDE_ENA_8                    : integer                                  := 0;
         PINS_AC_HMC_DATA_OVERRIDE_ENA_9                    : integer                                  := 0;
         PINS_AC_HMC_DATA_OVERRIDE_ENA_10                   : integer                                  := 0;
         PINS_AC_HMC_DATA_OVERRIDE_ENA_11                   : integer                                  := 0;
         PINS_AC_HMC_DATA_OVERRIDE_ENA_12                   : integer                                  := 0;
         PINS_AC_HMC_DATA_OVERRIDE_ENA_AUTOGEN_WCNT         : integer                                  := 0;
         PINS_OCT_MODE_0                                    : integer                                  := 0;
         PINS_OCT_MODE_1                                    : integer                                  := 0;
         PINS_OCT_MODE_2                                    : integer                                  := 0;
         PINS_OCT_MODE_3                                    : integer                                  := 0;
         PINS_OCT_MODE_4                                    : integer                                  := 0;
         PINS_OCT_MODE_5                                    : integer                                  := 0;
         PINS_OCT_MODE_6                                    : integer                                  := 0;
         PINS_OCT_MODE_7                                    : integer                                  := 0;
         PINS_OCT_MODE_8                                    : integer                                  := 0;
         PINS_OCT_MODE_9                                    : integer                                  := 0;
         PINS_OCT_MODE_10                                   : integer                                  := 0;
         PINS_OCT_MODE_11                                   : integer                                  := 0;
         PINS_OCT_MODE_12                                   : integer                                  := 0;
         PINS_OCT_MODE_AUTOGEN_WCNT                         : integer                                  := 0;
         PINS_GPIO_MODE_0                                   : integer                                  := 0;
         PINS_GPIO_MODE_1                                   : integer                                  := 0;
         PINS_GPIO_MODE_2                                   : integer                                  := 0;
         PINS_GPIO_MODE_3                                   : integer                                  := 0;
         PINS_GPIO_MODE_4                                   : integer                                  := 0;
         PINS_GPIO_MODE_5                                   : integer                                  := 0;
         PINS_GPIO_MODE_6                                   : integer                                  := 0;
         PINS_GPIO_MODE_7                                   : integer                                  := 0;
         PINS_GPIO_MODE_8                                   : integer                                  := 0;
         PINS_GPIO_MODE_9                                   : integer                                  := 0;
         PINS_GPIO_MODE_10                                  : integer                                  := 0;
         PINS_GPIO_MODE_11                                  : integer                                  := 0;
         PINS_GPIO_MODE_12                                  : integer                                  := 0;
         PINS_GPIO_MODE_AUTOGEN_WCNT                        : integer                                  := 0;
         UNUSED_MEM_PINS_PINLOC_0                           : integer                                  := 0;
         UNUSED_MEM_PINS_PINLOC_1                           : integer                                  := 0;
         UNUSED_MEM_PINS_PINLOC_2                           : integer                                  := 0;
         UNUSED_MEM_PINS_PINLOC_3                           : integer                                  := 0;
         UNUSED_MEM_PINS_PINLOC_4                           : integer                                  := 0;
         UNUSED_MEM_PINS_PINLOC_5                           : integer                                  := 0;
         UNUSED_MEM_PINS_PINLOC_6                           : integer                                  := 0;
         UNUSED_MEM_PINS_PINLOC_7                           : integer                                  := 0;
         UNUSED_MEM_PINS_PINLOC_8                           : integer                                  := 0;
         UNUSED_MEM_PINS_PINLOC_9                           : integer                                  := 0;
         UNUSED_MEM_PINS_PINLOC_10                          : integer                                  := 0;
         UNUSED_MEM_PINS_PINLOC_11                          : integer                                  := 0;
         UNUSED_MEM_PINS_PINLOC_12                          : integer                                  := 0;
         UNUSED_MEM_PINS_PINLOC_13                          : integer                                  := 0;
         UNUSED_MEM_PINS_PINLOC_14                          : integer                                  := 0;
         UNUSED_MEM_PINS_PINLOC_15                          : integer                                  := 0;
         UNUSED_MEM_PINS_PINLOC_16                          : integer                                  := 0;
         UNUSED_MEM_PINS_PINLOC_17                          : integer                                  := 0;
         UNUSED_MEM_PINS_PINLOC_18                          : integer                                  := 0;
         UNUSED_MEM_PINS_PINLOC_19                          : integer                                  := 0;
         UNUSED_MEM_PINS_PINLOC_20                          : integer                                  := 0;
         UNUSED_MEM_PINS_PINLOC_21                          : integer                                  := 0;
         UNUSED_MEM_PINS_PINLOC_22                          : integer                                  := 0;
         UNUSED_MEM_PINS_PINLOC_23                          : integer                                  := 0;
         UNUSED_MEM_PINS_PINLOC_24                          : integer                                  := 0;
         UNUSED_MEM_PINS_PINLOC_25                          : integer                                  := 0;
         UNUSED_MEM_PINS_PINLOC_26                          : integer                                  := 0;
         UNUSED_MEM_PINS_PINLOC_27                          : integer                                  := 0;
         UNUSED_MEM_PINS_PINLOC_28                          : integer                                  := 0;
         UNUSED_MEM_PINS_PINLOC_29                          : integer                                  := 0;
         UNUSED_MEM_PINS_PINLOC_30                          : integer                                  := 0;
         UNUSED_MEM_PINS_PINLOC_31                          : integer                                  := 0;
         UNUSED_MEM_PINS_PINLOC_32                          : integer                                  := 0;
         UNUSED_MEM_PINS_PINLOC_33                          : integer                                  := 0;
         UNUSED_MEM_PINS_PINLOC_34                          : integer                                  := 0;
         UNUSED_MEM_PINS_PINLOC_35                          : integer                                  := 0;
         UNUSED_MEM_PINS_PINLOC_36                          : integer                                  := 0;
         UNUSED_MEM_PINS_PINLOC_37                          : integer                                  := 0;
         UNUSED_MEM_PINS_PINLOC_38                          : integer                                  := 0;
         UNUSED_MEM_PINS_PINLOC_39                          : integer                                  := 0;
         UNUSED_MEM_PINS_PINLOC_40                          : integer                                  := 0;
         UNUSED_MEM_PINS_PINLOC_41                          : integer                                  := 0;
         UNUSED_MEM_PINS_PINLOC_42                          : integer                                  := 0;
         UNUSED_MEM_PINS_PINLOC_43                          : integer                                  := 0;
         UNUSED_MEM_PINS_PINLOC_44                          : integer                                  := 0;
         UNUSED_MEM_PINS_PINLOC_45                          : integer                                  := 0;
         UNUSED_MEM_PINS_PINLOC_46                          : integer                                  := 0;
         UNUSED_MEM_PINS_PINLOC_47                          : integer                                  := 0;
         UNUSED_MEM_PINS_PINLOC_48                          : integer                                  := 0;
         UNUSED_MEM_PINS_PINLOC_49                          : integer                                  := 0;
         UNUSED_MEM_PINS_PINLOC_50                          : integer                                  := 0;
         UNUSED_MEM_PINS_PINLOC_51                          : integer                                  := 0;
         UNUSED_MEM_PINS_PINLOC_52                          : integer                                  := 0;
         UNUSED_MEM_PINS_PINLOC_53                          : integer                                  := 0;
         UNUSED_MEM_PINS_PINLOC_54                          : integer                                  := 0;
         UNUSED_MEM_PINS_PINLOC_55                          : integer                                  := 0;
         UNUSED_MEM_PINS_PINLOC_56                          : integer                                  := 0;
         UNUSED_MEM_PINS_PINLOC_57                          : integer                                  := 0;
         UNUSED_MEM_PINS_PINLOC_58                          : integer                                  := 0;
         UNUSED_MEM_PINS_PINLOC_59                          : integer                                  := 0;
         UNUSED_MEM_PINS_PINLOC_60                          : integer                                  := 0;
         UNUSED_MEM_PINS_PINLOC_61                          : integer                                  := 0;
         UNUSED_MEM_PINS_PINLOC_62                          : integer                                  := 0;
         UNUSED_MEM_PINS_PINLOC_63                          : integer                                  := 0;
         UNUSED_MEM_PINS_PINLOC_64                          : integer                                  := 0;
         UNUSED_MEM_PINS_PINLOC_65                          : integer                                  := 0;
         UNUSED_MEM_PINS_PINLOC_66                          : integer                                  := 0;
         UNUSED_MEM_PINS_PINLOC_67                          : integer                                  := 0;
         UNUSED_MEM_PINS_PINLOC_68                          : integer                                  := 0;
         UNUSED_MEM_PINS_PINLOC_69                          : integer                                  := 0;
         UNUSED_MEM_PINS_PINLOC_70                          : integer                                  := 0;
         UNUSED_MEM_PINS_PINLOC_71                          : integer                                  := 0;
         UNUSED_MEM_PINS_PINLOC_72                          : integer                                  := 0;
         UNUSED_MEM_PINS_PINLOC_73                          : integer                                  := 0;
         UNUSED_MEM_PINS_PINLOC_74                          : integer                                  := 0;
         UNUSED_MEM_PINS_PINLOC_75                          : integer                                  := 0;
         UNUSED_MEM_PINS_PINLOC_76                          : integer                                  := 0;
         UNUSED_MEM_PINS_PINLOC_77                          : integer                                  := 0;
         UNUSED_MEM_PINS_PINLOC_78                          : integer                                  := 0;
         UNUSED_MEM_PINS_PINLOC_79                          : integer                                  := 0;
         UNUSED_MEM_PINS_PINLOC_80                          : integer                                  := 0;
         UNUSED_MEM_PINS_PINLOC_81                          : integer                                  := 0;
         UNUSED_MEM_PINS_PINLOC_82                          : integer                                  := 0;
         UNUSED_MEM_PINS_PINLOC_83                          : integer                                  := 0;
         UNUSED_MEM_PINS_PINLOC_84                          : integer                                  := 0;
         UNUSED_MEM_PINS_PINLOC_85                          : integer                                  := 0;
         UNUSED_MEM_PINS_PINLOC_86                          : integer                                  := 0;
         UNUSED_MEM_PINS_PINLOC_87                          : integer                                  := 0;
         UNUSED_MEM_PINS_PINLOC_88                          : integer                                  := 0;
         UNUSED_MEM_PINS_PINLOC_89                          : integer                                  := 0;
         UNUSED_MEM_PINS_PINLOC_90                          : integer                                  := 0;
         UNUSED_MEM_PINS_PINLOC_91                          : integer                                  := 0;
         UNUSED_MEM_PINS_PINLOC_92                          : integer                                  := 0;
         UNUSED_MEM_PINS_PINLOC_93                          : integer                                  := 0;
         UNUSED_MEM_PINS_PINLOC_94                          : integer                                  := 0;
         UNUSED_MEM_PINS_PINLOC_95                          : integer                                  := 0;
         UNUSED_MEM_PINS_PINLOC_96                          : integer                                  := 0;
         UNUSED_MEM_PINS_PINLOC_97                          : integer                                  := 0;
         UNUSED_MEM_PINS_PINLOC_98                          : integer                                  := 0;
         UNUSED_MEM_PINS_PINLOC_99                          : integer                                  := 0;
         UNUSED_MEM_PINS_PINLOC_100                         : integer                                  := 0;
         UNUSED_MEM_PINS_PINLOC_101                         : integer                                  := 0;
         UNUSED_MEM_PINS_PINLOC_102                         : integer                                  := 0;
         UNUSED_MEM_PINS_PINLOC_103                         : integer                                  := 0;
         UNUSED_MEM_PINS_PINLOC_104                         : integer                                  := 0;
         UNUSED_MEM_PINS_PINLOC_105                         : integer                                  := 0;
         UNUSED_MEM_PINS_PINLOC_106                         : integer                                  := 0;
         UNUSED_MEM_PINS_PINLOC_107                         : integer                                  := 0;
         UNUSED_MEM_PINS_PINLOC_108                         : integer                                  := 0;
         UNUSED_MEM_PINS_PINLOC_109                         : integer                                  := 0;
         UNUSED_MEM_PINS_PINLOC_110                         : integer                                  := 0;
         UNUSED_MEM_PINS_PINLOC_111                         : integer                                  := 0;
         UNUSED_MEM_PINS_PINLOC_112                         : integer                                  := 0;
         UNUSED_MEM_PINS_PINLOC_113                         : integer                                  := 0;
         UNUSED_MEM_PINS_PINLOC_114                         : integer                                  := 0;
         UNUSED_MEM_PINS_PINLOC_115                         : integer                                  := 0;
         UNUSED_MEM_PINS_PINLOC_116                         : integer                                  := 0;
         UNUSED_MEM_PINS_PINLOC_117                         : integer                                  := 0;
         UNUSED_MEM_PINS_PINLOC_118                         : integer                                  := 0;
         UNUSED_MEM_PINS_PINLOC_119                         : integer                                  := 0;
         UNUSED_MEM_PINS_PINLOC_120                         : integer                                  := 0;
         UNUSED_MEM_PINS_PINLOC_121                         : integer                                  := 0;
         UNUSED_MEM_PINS_PINLOC_122                         : integer                                  := 0;
         UNUSED_MEM_PINS_PINLOC_123                         : integer                                  := 0;
         UNUSED_MEM_PINS_PINLOC_124                         : integer                                  := 0;
         UNUSED_MEM_PINS_PINLOC_125                         : integer                                  := 0;
         UNUSED_MEM_PINS_PINLOC_126                         : integer                                  := 0;
         UNUSED_MEM_PINS_PINLOC_127                         : integer                                  := 0;
         UNUSED_MEM_PINS_PINLOC_128                         : integer                                  := 0;
         UNUSED_MEM_PINS_PINLOC_AUTOGEN_WCNT                : integer                                  := 0;
         UNUSED_DQS_BUSES_LANELOC_0                         : integer                                  := 0;
         UNUSED_DQS_BUSES_LANELOC_1                         : integer                                  := 0;
         UNUSED_DQS_BUSES_LANELOC_2                         : integer                                  := 0;
         UNUSED_DQS_BUSES_LANELOC_3                         : integer                                  := 0;
         UNUSED_DQS_BUSES_LANELOC_4                         : integer                                  := 0;
         UNUSED_DQS_BUSES_LANELOC_5                         : integer                                  := 0;
         UNUSED_DQS_BUSES_LANELOC_6                         : integer                                  := 0;
         UNUSED_DQS_BUSES_LANELOC_7                         : integer                                  := 0;
         UNUSED_DQS_BUSES_LANELOC_8                         : integer                                  := 0;
         UNUSED_DQS_BUSES_LANELOC_9                         : integer                                  := 0;
         UNUSED_DQS_BUSES_LANELOC_10                        : integer                                  := 0;
         UNUSED_DQS_BUSES_LANELOC_AUTOGEN_WCNT              : integer                                  := 0;
         CENTER_TIDS_0                                      : integer                                  := 0;
         CENTER_TIDS_1                                      : integer                                  := 0;
         CENTER_TIDS_2                                      : integer                                  := 0;
         CENTER_TIDS_AUTOGEN_WCNT                           : integer                                  := 0;
         HMC_TIDS_0                                         : integer                                  := 0;
         HMC_TIDS_1                                         : integer                                  := 0;
         HMC_TIDS_2                                         : integer                                  := 0;
         HMC_TIDS_AUTOGEN_WCNT                              : integer                                  := 0;
         LANE_TIDS_0                                        : integer                                  := 0;
         LANE_TIDS_1                                        : integer                                  := 0;
         LANE_TIDS_2                                        : integer                                  := 0;
         LANE_TIDS_3                                        : integer                                  := 0;
         LANE_TIDS_4                                        : integer                                  := 0;
         LANE_TIDS_5                                        : integer                                  := 0;
         LANE_TIDS_6                                        : integer                                  := 0;
         LANE_TIDS_7                                        : integer                                  := 0;
         LANE_TIDS_8                                        : integer                                  := 0;
         LANE_TIDS_9                                        : integer                                  := 0;
         LANE_TIDS_AUTOGEN_WCNT                             : integer                                  := 0;
         PREAMBLE_MODE                                      : string                                   := "";
         DBI_WR_ENABLE                                      : string                                   := "";
         DBI_RD_ENABLE                                      : string                                   := "";
         CRC_EN                                             : string                                   := "";
         SWAP_DQS_A_B                                       : string                                   := "";
         DQS_PACK_MODE                                      : string                                   := "";
         OCT_SIZE                                           : integer                                  := 0;
         DBC_WB_RESERVED_ENTRY                              : integer                                  := 0;
         DLL_MODE                                           : string                                   := "";
         DLL_CODEWORD                                       : integer                                  := 0;
         ABPHY_WRITE_PROTOCOL                               : integer                                  := 0;
         PHY_USERMODE_OCT                                   : boolean                                  := false;
         PHY_PERIODIC_OCT_RECAL                             : boolean                                  := false;
         PHY_HAS_DCC                                        : boolean                                  := false;
         PRI_HMC_CFG_ENABLE_ECC                             : string                                   := "";
         PRI_HMC_CFG_REORDER_DATA                           : string                                   := "";
         PRI_HMC_CFG_REORDER_READ                           : string                                   := "";
         PRI_HMC_CFG_REORDER_RDATA                          : string                                   := "";
         PRI_HMC_CFG_STARVE_LIMIT                           : integer                                  := 0;
         PRI_HMC_CFG_DQS_TRACKING_EN                        : string                                   := "";
         PRI_HMC_CFG_ARBITER_TYPE                           : string                                   := "";
         PRI_HMC_CFG_OPEN_PAGE_EN                           : string                                   := "";
         PRI_HMC_CFG_GEAR_DOWN_EN                           : string                                   := "";
         PRI_HMC_CFG_RLD3_MULTIBANK_MODE                    : string                                   := "";
         PRI_HMC_CFG_PING_PONG_MODE                         : string                                   := "";
         PRI_HMC_CFG_SLOT_ROTATE_EN                         : integer                                  := 0;
         PRI_HMC_CFG_SLOT_OFFSET                            : integer                                  := 0;
         PRI_HMC_CFG_COL_CMD_SLOT                           : integer                                  := 0;
         PRI_HMC_CFG_ROW_CMD_SLOT                           : integer                                  := 0;
         PRI_HMC_CFG_ENABLE_RC                              : string                                   := "";
         PRI_HMC_CFG_CS_TO_CHIP_MAPPING                     : integer                                  := 0;
         PRI_HMC_CFG_RB_RESERVED_ENTRY                      : integer                                  := 0;
         PRI_HMC_CFG_WB_RESERVED_ENTRY                      : integer                                  := 0;
         PRI_HMC_CFG_TCL                                    : integer                                  := 0;
         PRI_HMC_CFG_POWER_SAVING_EXIT_CYC                  : integer                                  := 0;
         PRI_HMC_CFG_MEM_CLK_DISABLE_ENTRY_CYC              : integer                                  := 0;
         PRI_HMC_CFG_WRITE_ODT_CHIP                         : integer                                  := 0;
         PRI_HMC_CFG_READ_ODT_CHIP                          : integer                                  := 0;
         PRI_HMC_CFG_WR_ODT_ON                              : integer                                  := 0;
         PRI_HMC_CFG_RD_ODT_ON                              : integer                                  := 0;
         PRI_HMC_CFG_WR_ODT_PERIOD                          : integer                                  := 0;
         PRI_HMC_CFG_RD_ODT_PERIOD                          : integer                                  := 0;
         PRI_HMC_CFG_RLD3_REFRESH_SEQ0                      : integer                                  := 0;
         PRI_HMC_CFG_RLD3_REFRESH_SEQ1                      : integer                                  := 0;
         PRI_HMC_CFG_RLD3_REFRESH_SEQ2                      : integer                                  := 0;
         PRI_HMC_CFG_RLD3_REFRESH_SEQ3                      : integer                                  := 0;
         PRI_HMC_CFG_SRF_ZQCAL_DISABLE                      : string                                   := "";
         PRI_HMC_CFG_MPS_ZQCAL_DISABLE                      : string                                   := "";
         PRI_HMC_CFG_MPS_DQSTRK_DISABLE                     : string                                   := "";
         PRI_HMC_CFG_SHORT_DQSTRK_CTRL_EN                   : string                                   := "";
         PRI_HMC_CFG_PERIOD_DQSTRK_CTRL_EN                  : string                                   := "";
         PRI_HMC_CFG_PERIOD_DQSTRK_INTERVAL                 : integer                                  := 0;
         PRI_HMC_CFG_DQSTRK_TO_VALID_LAST                   : integer                                  := 0;
         PRI_HMC_CFG_DQSTRK_TO_VALID                        : integer                                  := 0;
         PRI_HMC_CFG_RFSH_WARN_THRESHOLD                    : integer                                  := 0;
         PRI_HMC_CFG_SB_CG_DISABLE                          : string                                   := "";
         PRI_HMC_CFG_USER_RFSH_EN                           : string                                   := "";
         PRI_HMC_CFG_SRF_AUTOEXIT_EN                        : string                                   := "";
         PRI_HMC_CFG_SRF_ENTRY_EXIT_BLOCK                   : string                                   := "";
         PRI_HMC_CFG_SB_DDR4_MR3                            : integer                                  := 0;
         PRI_HMC_CFG_SB_DDR4_MR4                            : integer                                  := 0;
         PRI_HMC_CFG_SB_DDR4_MR5                            : integer                                  := 0;
         PRI_HMC_CFG_DDR4_MPS_ADDR_MIRROR                   : integer                                  := 0;
         PRI_HMC_CFG_MEM_IF_COLADDR_WIDTH                   : string                                   := "";
         PRI_HMC_CFG_MEM_IF_ROWADDR_WIDTH                   : string                                   := "";
         PRI_HMC_CFG_MEM_IF_BANKADDR_WIDTH                  : string                                   := "";
         PRI_HMC_CFG_MEM_IF_BGADDR_WIDTH                    : string                                   := "";
         PRI_HMC_CFG_LOCAL_IF_CS_WIDTH                      : string                                   := "";
         PRI_HMC_CFG_ADDR_ORDER                             : string                                   := "";
         PRI_HMC_CFG_ACT_TO_RDWR                            : integer                                  := 0;
         PRI_HMC_CFG_ACT_TO_PCH                             : integer                                  := 0;
         PRI_HMC_CFG_ACT_TO_ACT                             : integer                                  := 0;
         PRI_HMC_CFG_ACT_TO_ACT_DIFF_BANK                   : integer                                  := 0;
         PRI_HMC_CFG_ACT_TO_ACT_DIFF_BG                     : integer                                  := 0;
         PRI_HMC_CFG_RD_TO_RD                               : integer                                  := 0;
         PRI_HMC_CFG_RD_TO_RD_DIFF_CHIP                     : integer                                  := 0;
         PRI_HMC_CFG_RD_TO_RD_DIFF_BG                       : integer                                  := 0;
         PRI_HMC_CFG_RD_TO_WR                               : integer                                  := 0;
         PRI_HMC_CFG_RD_TO_WR_DIFF_CHIP                     : integer                                  := 0;
         PRI_HMC_CFG_RD_TO_WR_DIFF_BG                       : integer                                  := 0;
         PRI_HMC_CFG_RD_TO_PCH                              : integer                                  := 0;
         PRI_HMC_CFG_RD_AP_TO_VALID                         : integer                                  := 0;
         PRI_HMC_CFG_WR_TO_WR                               : integer                                  := 0;
         PRI_HMC_CFG_WR_TO_WR_DIFF_CHIP                     : integer                                  := 0;
         PRI_HMC_CFG_WR_TO_WR_DIFF_BG                       : integer                                  := 0;
         PRI_HMC_CFG_WR_TO_RD                               : integer                                  := 0;
         PRI_HMC_CFG_WR_TO_RD_DIFF_CHIP                     : integer                                  := 0;
         PRI_HMC_CFG_WR_TO_RD_DIFF_BG                       : integer                                  := 0;
         PRI_HMC_CFG_WR_TO_PCH                              : integer                                  := 0;
         PRI_HMC_CFG_WR_AP_TO_VALID                         : integer                                  := 0;
         PRI_HMC_CFG_PCH_TO_VALID                           : integer                                  := 0;
         PRI_HMC_CFG_PCH_ALL_TO_VALID                       : integer                                  := 0;
         PRI_HMC_CFG_ARF_TO_VALID                           : integer                                  := 0;
         PRI_HMC_CFG_PDN_TO_VALID                           : integer                                  := 0;
         PRI_HMC_CFG_SRF_TO_VALID                           : integer                                  := 0;
         PRI_HMC_CFG_SRF_TO_ZQ_CAL                          : integer                                  := 0;
         PRI_HMC_CFG_ARF_PERIOD                             : integer                                  := 0;
         PRI_HMC_CFG_PDN_PERIOD                             : integer                                  := 0;
         PRI_HMC_CFG_ZQCL_TO_VALID                          : integer                                  := 0;
         PRI_HMC_CFG_ZQCS_TO_VALID                          : integer                                  := 0;
         PRI_HMC_CFG_MRS_TO_VALID                           : integer                                  := 0;
         PRI_HMC_CFG_MPS_TO_VALID                           : integer                                  := 0;
         PRI_HMC_CFG_MRR_TO_VALID                           : integer                                  := 0;
         PRI_HMC_CFG_MPR_TO_VALID                           : integer                                  := 0;
         PRI_HMC_CFG_MPS_EXIT_CS_TO_CKE                     : integer                                  := 0;
         PRI_HMC_CFG_MPS_EXIT_CKE_TO_CS                     : integer                                  := 0;
         PRI_HMC_CFG_RLD3_MULTIBANK_REF_DELAY               : integer                                  := 0;
         PRI_HMC_CFG_MMR_CMD_TO_VALID                       : integer                                  := 0;
         PRI_HMC_CFG_4_ACT_TO_ACT                           : integer                                  := 0;
         PRI_HMC_CFG_16_ACT_TO_ACT                          : integer                                  := 0;
         SEC_HMC_CFG_ENABLE_ECC                             : string                                   := "";
         SEC_HMC_CFG_REORDER_DATA                           : string                                   := "";
         SEC_HMC_CFG_REORDER_READ                           : string                                   := "";
         SEC_HMC_CFG_REORDER_RDATA                          : string                                   := "";
         SEC_HMC_CFG_STARVE_LIMIT                           : integer                                  := 0;
         SEC_HMC_CFG_DQS_TRACKING_EN                        : string                                   := "";
         SEC_HMC_CFG_ARBITER_TYPE                           : string                                   := "";
         SEC_HMC_CFG_OPEN_PAGE_EN                           : string                                   := "";
         SEC_HMC_CFG_GEAR_DOWN_EN                           : string                                   := "";
         SEC_HMC_CFG_RLD3_MULTIBANK_MODE                    : string                                   := "";
         SEC_HMC_CFG_PING_PONG_MODE                         : string                                   := "";
         SEC_HMC_CFG_SLOT_ROTATE_EN                         : integer                                  := 0;
         SEC_HMC_CFG_SLOT_OFFSET                            : integer                                  := 0;
         SEC_HMC_CFG_COL_CMD_SLOT                           : integer                                  := 0;
         SEC_HMC_CFG_ROW_CMD_SLOT                           : integer                                  := 0;
         SEC_HMC_CFG_ENABLE_RC                              : string                                   := "";
         SEC_HMC_CFG_CS_TO_CHIP_MAPPING                     : integer                                  := 0;
         SEC_HMC_CFG_RB_RESERVED_ENTRY                      : integer                                  := 0;
         SEC_HMC_CFG_WB_RESERVED_ENTRY                      : integer                                  := 0;
         SEC_HMC_CFG_TCL                                    : integer                                  := 0;
         SEC_HMC_CFG_POWER_SAVING_EXIT_CYC                  : integer                                  := 0;
         SEC_HMC_CFG_MEM_CLK_DISABLE_ENTRY_CYC              : integer                                  := 0;
         SEC_HMC_CFG_WRITE_ODT_CHIP                         : integer                                  := 0;
         SEC_HMC_CFG_READ_ODT_CHIP                          : integer                                  := 0;
         SEC_HMC_CFG_WR_ODT_ON                              : integer                                  := 0;
         SEC_HMC_CFG_RD_ODT_ON                              : integer                                  := 0;
         SEC_HMC_CFG_WR_ODT_PERIOD                          : integer                                  := 0;
         SEC_HMC_CFG_RD_ODT_PERIOD                          : integer                                  := 0;
         SEC_HMC_CFG_RLD3_REFRESH_SEQ0                      : integer                                  := 0;
         SEC_HMC_CFG_RLD3_REFRESH_SEQ1                      : integer                                  := 0;
         SEC_HMC_CFG_RLD3_REFRESH_SEQ2                      : integer                                  := 0;
         SEC_HMC_CFG_RLD3_REFRESH_SEQ3                      : integer                                  := 0;
         SEC_HMC_CFG_SRF_ZQCAL_DISABLE                      : string                                   := "";
         SEC_HMC_CFG_MPS_ZQCAL_DISABLE                      : string                                   := "";
         SEC_HMC_CFG_MPS_DQSTRK_DISABLE                     : string                                   := "";
         SEC_HMC_CFG_SHORT_DQSTRK_CTRL_EN                   : string                                   := "";
         SEC_HMC_CFG_PERIOD_DQSTRK_CTRL_EN                  : string                                   := "";
         SEC_HMC_CFG_PERIOD_DQSTRK_INTERVAL                 : integer                                  := 0;
         SEC_HMC_CFG_DQSTRK_TO_VALID_LAST                   : integer                                  := 0;
         SEC_HMC_CFG_DQSTRK_TO_VALID                        : integer                                  := 0;
         SEC_HMC_CFG_RFSH_WARN_THRESHOLD                    : integer                                  := 0;
         SEC_HMC_CFG_SB_CG_DISABLE                          : string                                   := "";
         SEC_HMC_CFG_USER_RFSH_EN                           : string                                   := "";
         SEC_HMC_CFG_SRF_AUTOEXIT_EN                        : string                                   := "";
         SEC_HMC_CFG_SRF_ENTRY_EXIT_BLOCK                   : string                                   := "";
         SEC_HMC_CFG_SB_DDR4_MR3                            : integer                                  := 0;
         SEC_HMC_CFG_SB_DDR4_MR4                            : integer                                  := 0;
         SEC_HMC_CFG_SB_DDR4_MR5                            : integer                                  := 0;
         SEC_HMC_CFG_DDR4_MPS_ADDR_MIRROR                   : integer                                  := 0;
         SEC_HMC_CFG_MEM_IF_COLADDR_WIDTH                   : string                                   := "";
         SEC_HMC_CFG_MEM_IF_ROWADDR_WIDTH                   : string                                   := "";
         SEC_HMC_CFG_MEM_IF_BANKADDR_WIDTH                  : string                                   := "";
         SEC_HMC_CFG_MEM_IF_BGADDR_WIDTH                    : string                                   := "";
         SEC_HMC_CFG_LOCAL_IF_CS_WIDTH                      : string                                   := "";
         SEC_HMC_CFG_ADDR_ORDER                             : string                                   := "";
         SEC_HMC_CFG_ACT_TO_RDWR                            : integer                                  := 0;
         SEC_HMC_CFG_ACT_TO_PCH                             : integer                                  := 0;
         SEC_HMC_CFG_ACT_TO_ACT                             : integer                                  := 0;
         SEC_HMC_CFG_ACT_TO_ACT_DIFF_BANK                   : integer                                  := 0;
         SEC_HMC_CFG_ACT_TO_ACT_DIFF_BG                     : integer                                  := 0;
         SEC_HMC_CFG_RD_TO_RD                               : integer                                  := 0;
         SEC_HMC_CFG_RD_TO_RD_DIFF_CHIP                     : integer                                  := 0;
         SEC_HMC_CFG_RD_TO_RD_DIFF_BG                       : integer                                  := 0;
         SEC_HMC_CFG_RD_TO_WR                               : integer                                  := 0;
         SEC_HMC_CFG_RD_TO_WR_DIFF_CHIP                     : integer                                  := 0;
         SEC_HMC_CFG_RD_TO_WR_DIFF_BG                       : integer                                  := 0;
         SEC_HMC_CFG_RD_TO_PCH                              : integer                                  := 0;
         SEC_HMC_CFG_RD_AP_TO_VALID                         : integer                                  := 0;
         SEC_HMC_CFG_WR_TO_WR                               : integer                                  := 0;
         SEC_HMC_CFG_WR_TO_WR_DIFF_CHIP                     : integer                                  := 0;
         SEC_HMC_CFG_WR_TO_WR_DIFF_BG                       : integer                                  := 0;
         SEC_HMC_CFG_WR_TO_RD                               : integer                                  := 0;
         SEC_HMC_CFG_WR_TO_RD_DIFF_CHIP                     : integer                                  := 0;
         SEC_HMC_CFG_WR_TO_RD_DIFF_BG                       : integer                                  := 0;
         SEC_HMC_CFG_WR_TO_PCH                              : integer                                  := 0;
         SEC_HMC_CFG_WR_AP_TO_VALID                         : integer                                  := 0;
         SEC_HMC_CFG_PCH_TO_VALID                           : integer                                  := 0;
         SEC_HMC_CFG_PCH_ALL_TO_VALID                       : integer                                  := 0;
         SEC_HMC_CFG_ARF_TO_VALID                           : integer                                  := 0;
         SEC_HMC_CFG_PDN_TO_VALID                           : integer                                  := 0;
         SEC_HMC_CFG_SRF_TO_VALID                           : integer                                  := 0;
         SEC_HMC_CFG_SRF_TO_ZQ_CAL                          : integer                                  := 0;
         SEC_HMC_CFG_ARF_PERIOD                             : integer                                  := 0;
         SEC_HMC_CFG_PDN_PERIOD                             : integer                                  := 0;
         SEC_HMC_CFG_ZQCL_TO_VALID                          : integer                                  := 0;
         SEC_HMC_CFG_ZQCS_TO_VALID                          : integer                                  := 0;
         SEC_HMC_CFG_MRS_TO_VALID                           : integer                                  := 0;
         SEC_HMC_CFG_MPS_TO_VALID                           : integer                                  := 0;
         SEC_HMC_CFG_MRR_TO_VALID                           : integer                                  := 0;
         SEC_HMC_CFG_MPR_TO_VALID                           : integer                                  := 0;
         SEC_HMC_CFG_MPS_EXIT_CS_TO_CKE                     : integer                                  := 0;
         SEC_HMC_CFG_MPS_EXIT_CKE_TO_CS                     : integer                                  := 0;
         SEC_HMC_CFG_RLD3_MULTIBANK_REF_DELAY               : integer                                  := 0;
         SEC_HMC_CFG_MMR_CMD_TO_VALID                       : integer                                  := 0;
         SEC_HMC_CFG_4_ACT_TO_ACT                           : integer                                  := 0;
         SEC_HMC_CFG_16_ACT_TO_ACT                          : integer                                  := 0;
         PINS_PER_LANE                                      : integer                                  := 0;
         LANES_PER_TILE                                     : integer                                  := 0;
         OCT_CONTROL_WIDTH                                  : integer                                  := 0;
         PORT_MEM_CK_WIDTH                                  : integer                                  := 0;
         PORT_MEM_CK_PINLOC_0                               : integer                                  := 0;
         PORT_MEM_CK_PINLOC_1                               : integer                                  := 0;
         PORT_MEM_CK_PINLOC_2                               : integer                                  := 0;
         PORT_MEM_CK_PINLOC_3                               : integer                                  := 0;
         PORT_MEM_CK_PINLOC_4                               : integer                                  := 0;
         PORT_MEM_CK_PINLOC_5                               : integer                                  := 0;
         PORT_MEM_CK_PINLOC_AUTOGEN_WCNT                    : integer                                  := 0;
         PORT_MEM_CK_N_WIDTH                                : integer                                  := 0;
         PORT_MEM_CK_N_PINLOC_0                             : integer                                  := 0;
         PORT_MEM_CK_N_PINLOC_1                             : integer                                  := 0;
         PORT_MEM_CK_N_PINLOC_2                             : integer                                  := 0;
         PORT_MEM_CK_N_PINLOC_3                             : integer                                  := 0;
         PORT_MEM_CK_N_PINLOC_4                             : integer                                  := 0;
         PORT_MEM_CK_N_PINLOC_5                             : integer                                  := 0;
         PORT_MEM_CK_N_PINLOC_AUTOGEN_WCNT                  : integer                                  := 0;
         PORT_MEM_CK_BIDIR_WIDTH                            : integer                                  := 0;
         PORT_MEM_CK_BIDIR_PINLOC_0                         : integer                                  := 0;
         PORT_MEM_CK_BIDIR_PINLOC_1                         : integer                                  := 0;
         PORT_MEM_CK_BIDIR_PINLOC_2                         : integer                                  := 0;
         PORT_MEM_CK_BIDIR_PINLOC_3                         : integer                                  := 0;
         PORT_MEM_CK_BIDIR_PINLOC_4                         : integer                                  := 0;
         PORT_MEM_CK_BIDIR_PINLOC_5                         : integer                                  := 0;
         PORT_MEM_CK_BIDIR_PINLOC_AUTOGEN_WCNT              : integer                                  := 0;
         PORT_MEM_CK_BIDIR_N_WIDTH                          : integer                                  := 0;
         PORT_MEM_CK_BIDIR_N_PINLOC_0                       : integer                                  := 0;
         PORT_MEM_CK_BIDIR_N_PINLOC_1                       : integer                                  := 0;
         PORT_MEM_CK_BIDIR_N_PINLOC_2                       : integer                                  := 0;
         PORT_MEM_CK_BIDIR_N_PINLOC_3                       : integer                                  := 0;
         PORT_MEM_CK_BIDIR_N_PINLOC_4                       : integer                                  := 0;
         PORT_MEM_CK_BIDIR_N_PINLOC_5                       : integer                                  := 0;
         PORT_MEM_CK_BIDIR_N_PINLOC_AUTOGEN_WCNT            : integer                                  := 0;
         PORT_MEM_DK_WIDTH                                  : integer                                  := 0;
         PORT_MEM_DK_PINLOC_0                               : integer                                  := 0;
         PORT_MEM_DK_PINLOC_1                               : integer                                  := 0;
         PORT_MEM_DK_PINLOC_2                               : integer                                  := 0;
         PORT_MEM_DK_PINLOC_3                               : integer                                  := 0;
         PORT_MEM_DK_PINLOC_4                               : integer                                  := 0;
         PORT_MEM_DK_PINLOC_5                               : integer                                  := 0;
         PORT_MEM_DK_PINLOC_AUTOGEN_WCNT                    : integer                                  := 0;
         PORT_MEM_DK_N_WIDTH                                : integer                                  := 0;
         PORT_MEM_DK_N_PINLOC_0                             : integer                                  := 0;
         PORT_MEM_DK_N_PINLOC_1                             : integer                                  := 0;
         PORT_MEM_DK_N_PINLOC_2                             : integer                                  := 0;
         PORT_MEM_DK_N_PINLOC_3                             : integer                                  := 0;
         PORT_MEM_DK_N_PINLOC_4                             : integer                                  := 0;
         PORT_MEM_DK_N_PINLOC_5                             : integer                                  := 0;
         PORT_MEM_DK_N_PINLOC_AUTOGEN_WCNT                  : integer                                  := 0;
         PORT_MEM_DKA_WIDTH                                 : integer                                  := 0;
         PORT_MEM_DKA_PINLOC_0                              : integer                                  := 0;
         PORT_MEM_DKA_PINLOC_1                              : integer                                  := 0;
         PORT_MEM_DKA_PINLOC_2                              : integer                                  := 0;
         PORT_MEM_DKA_PINLOC_3                              : integer                                  := 0;
         PORT_MEM_DKA_PINLOC_4                              : integer                                  := 0;
         PORT_MEM_DKA_PINLOC_5                              : integer                                  := 0;
         PORT_MEM_DKA_PINLOC_AUTOGEN_WCNT                   : integer                                  := 0;
         PORT_MEM_DKA_N_WIDTH                               : integer                                  := 0;
         PORT_MEM_DKA_N_PINLOC_0                            : integer                                  := 0;
         PORT_MEM_DKA_N_PINLOC_1                            : integer                                  := 0;
         PORT_MEM_DKA_N_PINLOC_2                            : integer                                  := 0;
         PORT_MEM_DKA_N_PINLOC_3                            : integer                                  := 0;
         PORT_MEM_DKA_N_PINLOC_4                            : integer                                  := 0;
         PORT_MEM_DKA_N_PINLOC_5                            : integer                                  := 0;
         PORT_MEM_DKA_N_PINLOC_AUTOGEN_WCNT                 : integer                                  := 0;
         PORT_MEM_DKB_WIDTH                                 : integer                                  := 0;
         PORT_MEM_DKB_PINLOC_0                              : integer                                  := 0;
         PORT_MEM_DKB_PINLOC_1                              : integer                                  := 0;
         PORT_MEM_DKB_PINLOC_2                              : integer                                  := 0;
         PORT_MEM_DKB_PINLOC_3                              : integer                                  := 0;
         PORT_MEM_DKB_PINLOC_4                              : integer                                  := 0;
         PORT_MEM_DKB_PINLOC_5                              : integer                                  := 0;
         PORT_MEM_DKB_PINLOC_AUTOGEN_WCNT                   : integer                                  := 0;
         PORT_MEM_DKB_N_WIDTH                               : integer                                  := 0;
         PORT_MEM_DKB_N_PINLOC_0                            : integer                                  := 0;
         PORT_MEM_DKB_N_PINLOC_1                            : integer                                  := 0;
         PORT_MEM_DKB_N_PINLOC_2                            : integer                                  := 0;
         PORT_MEM_DKB_N_PINLOC_3                            : integer                                  := 0;
         PORT_MEM_DKB_N_PINLOC_4                            : integer                                  := 0;
         PORT_MEM_DKB_N_PINLOC_5                            : integer                                  := 0;
         PORT_MEM_DKB_N_PINLOC_AUTOGEN_WCNT                 : integer                                  := 0;
         PORT_MEM_K_WIDTH                                   : integer                                  := 0;
         PORT_MEM_K_PINLOC_0                                : integer                                  := 0;
         PORT_MEM_K_PINLOC_1                                : integer                                  := 0;
         PORT_MEM_K_PINLOC_2                                : integer                                  := 0;
         PORT_MEM_K_PINLOC_3                                : integer                                  := 0;
         PORT_MEM_K_PINLOC_4                                : integer                                  := 0;
         PORT_MEM_K_PINLOC_5                                : integer                                  := 0;
         PORT_MEM_K_PINLOC_AUTOGEN_WCNT                     : integer                                  := 0;
         PORT_MEM_K_N_WIDTH                                 : integer                                  := 0;
         PORT_MEM_K_N_PINLOC_0                              : integer                                  := 0;
         PORT_MEM_K_N_PINLOC_1                              : integer                                  := 0;
         PORT_MEM_K_N_PINLOC_2                              : integer                                  := 0;
         PORT_MEM_K_N_PINLOC_3                              : integer                                  := 0;
         PORT_MEM_K_N_PINLOC_4                              : integer                                  := 0;
         PORT_MEM_K_N_PINLOC_5                              : integer                                  := 0;
         PORT_MEM_K_N_PINLOC_AUTOGEN_WCNT                   : integer                                  := 0;
         PORT_MEM_A_WIDTH                                   : integer                                  := 0;
         PORT_MEM_A_PINLOC_0                                : integer                                  := 0;
         PORT_MEM_A_PINLOC_1                                : integer                                  := 0;
         PORT_MEM_A_PINLOC_2                                : integer                                  := 0;
         PORT_MEM_A_PINLOC_3                                : integer                                  := 0;
         PORT_MEM_A_PINLOC_4                                : integer                                  := 0;
         PORT_MEM_A_PINLOC_5                                : integer                                  := 0;
         PORT_MEM_A_PINLOC_6                                : integer                                  := 0;
         PORT_MEM_A_PINLOC_7                                : integer                                  := 0;
         PORT_MEM_A_PINLOC_8                                : integer                                  := 0;
         PORT_MEM_A_PINLOC_9                                : integer                                  := 0;
         PORT_MEM_A_PINLOC_10                               : integer                                  := 0;
         PORT_MEM_A_PINLOC_11                               : integer                                  := 0;
         PORT_MEM_A_PINLOC_12                               : integer                                  := 0;
         PORT_MEM_A_PINLOC_13                               : integer                                  := 0;
         PORT_MEM_A_PINLOC_14                               : integer                                  := 0;
         PORT_MEM_A_PINLOC_15                               : integer                                  := 0;
         PORT_MEM_A_PINLOC_16                               : integer                                  := 0;
         PORT_MEM_A_PINLOC_AUTOGEN_WCNT                     : integer                                  := 0;
         PORT_MEM_BA_WIDTH                                  : integer                                  := 0;
         PORT_MEM_BA_PINLOC_0                               : integer                                  := 0;
         PORT_MEM_BA_PINLOC_1                               : integer                                  := 0;
         PORT_MEM_BA_PINLOC_2                               : integer                                  := 0;
         PORT_MEM_BA_PINLOC_3                               : integer                                  := 0;
         PORT_MEM_BA_PINLOC_4                               : integer                                  := 0;
         PORT_MEM_BA_PINLOC_5                               : integer                                  := 0;
         PORT_MEM_BA_PINLOC_AUTOGEN_WCNT                    : integer                                  := 0;
         PORT_MEM_BG_WIDTH                                  : integer                                  := 0;
         PORT_MEM_BG_PINLOC_0                               : integer                                  := 0;
         PORT_MEM_BG_PINLOC_1                               : integer                                  := 0;
         PORT_MEM_BG_PINLOC_2                               : integer                                  := 0;
         PORT_MEM_BG_PINLOC_3                               : integer                                  := 0;
         PORT_MEM_BG_PINLOC_4                               : integer                                  := 0;
         PORT_MEM_BG_PINLOC_5                               : integer                                  := 0;
         PORT_MEM_BG_PINLOC_AUTOGEN_WCNT                    : integer                                  := 0;
         PORT_MEM_C_WIDTH                                   : integer                                  := 0;
         PORT_MEM_C_PINLOC_0                                : integer                                  := 0;
         PORT_MEM_C_PINLOC_1                                : integer                                  := 0;
         PORT_MEM_C_PINLOC_2                                : integer                                  := 0;
         PORT_MEM_C_PINLOC_3                                : integer                                  := 0;
         PORT_MEM_C_PINLOC_4                                : integer                                  := 0;
         PORT_MEM_C_PINLOC_5                                : integer                                  := 0;
         PORT_MEM_C_PINLOC_AUTOGEN_WCNT                     : integer                                  := 0;
         PORT_MEM_CKE_WIDTH                                 : integer                                  := 0;
         PORT_MEM_CKE_PINLOC_0                              : integer                                  := 0;
         PORT_MEM_CKE_PINLOC_1                              : integer                                  := 0;
         PORT_MEM_CKE_PINLOC_2                              : integer                                  := 0;
         PORT_MEM_CKE_PINLOC_3                              : integer                                  := 0;
         PORT_MEM_CKE_PINLOC_4                              : integer                                  := 0;
         PORT_MEM_CKE_PINLOC_5                              : integer                                  := 0;
         PORT_MEM_CKE_PINLOC_AUTOGEN_WCNT                   : integer                                  := 0;
         PORT_MEM_CS_N_WIDTH                                : integer                                  := 0;
         PORT_MEM_CS_N_PINLOC_0                             : integer                                  := 0;
         PORT_MEM_CS_N_PINLOC_1                             : integer                                  := 0;
         PORT_MEM_CS_N_PINLOC_2                             : integer                                  := 0;
         PORT_MEM_CS_N_PINLOC_3                             : integer                                  := 0;
         PORT_MEM_CS_N_PINLOC_4                             : integer                                  := 0;
         PORT_MEM_CS_N_PINLOC_5                             : integer                                  := 0;
         PORT_MEM_CS_N_PINLOC_AUTOGEN_WCNT                  : integer                                  := 0;
         PORT_MEM_RM_WIDTH                                  : integer                                  := 0;
         PORT_MEM_RM_PINLOC_0                               : integer                                  := 0;
         PORT_MEM_RM_PINLOC_1                               : integer                                  := 0;
         PORT_MEM_RM_PINLOC_2                               : integer                                  := 0;
         PORT_MEM_RM_PINLOC_3                               : integer                                  := 0;
         PORT_MEM_RM_PINLOC_4                               : integer                                  := 0;
         PORT_MEM_RM_PINLOC_5                               : integer                                  := 0;
         PORT_MEM_RM_PINLOC_AUTOGEN_WCNT                    : integer                                  := 0;
         PORT_MEM_ODT_WIDTH                                 : integer                                  := 0;
         PORT_MEM_ODT_PINLOC_0                              : integer                                  := 0;
         PORT_MEM_ODT_PINLOC_1                              : integer                                  := 0;
         PORT_MEM_ODT_PINLOC_2                              : integer                                  := 0;
         PORT_MEM_ODT_PINLOC_3                              : integer                                  := 0;
         PORT_MEM_ODT_PINLOC_4                              : integer                                  := 0;
         PORT_MEM_ODT_PINLOC_5                              : integer                                  := 0;
         PORT_MEM_ODT_PINLOC_AUTOGEN_WCNT                   : integer                                  := 0;
         PORT_MEM_REQ_N_WIDTH                               : integer                                  := 0;
         PORT_MEM_REQ_N_PINLOC_0                            : integer                                  := 0;
         PORT_MEM_REQ_N_PINLOC_1                            : integer                                  := 0;
         PORT_MEM_REQ_N_PINLOC_2                            : integer                                  := 0;
         PORT_MEM_REQ_N_PINLOC_3                            : integer                                  := 0;
         PORT_MEM_REQ_N_PINLOC_4                            : integer                                  := 0;
         PORT_MEM_REQ_N_PINLOC_5                            : integer                                  := 0;
         PORT_MEM_REQ_N_PINLOC_AUTOGEN_WCNT                 : integer                                  := 0;
         PORT_MEM_GNT_N_WIDTH                               : integer                                  := 0;
         PORT_MEM_GNT_N_PINLOC_0                            : integer                                  := 0;
         PORT_MEM_GNT_N_PINLOC_1                            : integer                                  := 0;
         PORT_MEM_GNT_N_PINLOC_2                            : integer                                  := 0;
         PORT_MEM_GNT_N_PINLOC_3                            : integer                                  := 0;
         PORT_MEM_GNT_N_PINLOC_4                            : integer                                  := 0;
         PORT_MEM_GNT_N_PINLOC_5                            : integer                                  := 0;
         PORT_MEM_GNT_N_PINLOC_AUTOGEN_WCNT                 : integer                                  := 0;
         PORT_MEM_ERR_N_WIDTH                               : integer                                  := 0;
         PORT_MEM_ERR_N_PINLOC_0                            : integer                                  := 0;
         PORT_MEM_ERR_N_PINLOC_1                            : integer                                  := 0;
         PORT_MEM_ERR_N_PINLOC_2                            : integer                                  := 0;
         PORT_MEM_ERR_N_PINLOC_3                            : integer                                  := 0;
         PORT_MEM_ERR_N_PINLOC_4                            : integer                                  := 0;
         PORT_MEM_ERR_N_PINLOC_5                            : integer                                  := 0;
         PORT_MEM_ERR_N_PINLOC_AUTOGEN_WCNT                 : integer                                  := 0;
         PORT_MEM_RAS_N_WIDTH                               : integer                                  := 0;
         PORT_MEM_RAS_N_PINLOC_0                            : integer                                  := 0;
         PORT_MEM_RAS_N_PINLOC_1                            : integer                                  := 0;
         PORT_MEM_RAS_N_PINLOC_AUTOGEN_WCNT                 : integer                                  := 0;
         PORT_MEM_CAS_N_WIDTH                               : integer                                  := 0;
         PORT_MEM_CAS_N_PINLOC_0                            : integer                                  := 0;
         PORT_MEM_CAS_N_PINLOC_1                            : integer                                  := 0;
         PORT_MEM_CAS_N_PINLOC_AUTOGEN_WCNT                 : integer                                  := 0;
         PORT_MEM_WE_N_WIDTH                                : integer                                  := 0;
         PORT_MEM_WE_N_PINLOC_0                             : integer                                  := 0;
         PORT_MEM_WE_N_PINLOC_1                             : integer                                  := 0;
         PORT_MEM_WE_N_PINLOC_AUTOGEN_WCNT                  : integer                                  := 0;
         PORT_MEM_RESET_N_WIDTH                             : integer                                  := 0;
         PORT_MEM_RESET_N_PINLOC_0                          : integer                                  := 0;
         PORT_MEM_RESET_N_PINLOC_1                          : integer                                  := 0;
         PORT_MEM_RESET_N_PINLOC_AUTOGEN_WCNT               : integer                                  := 0;
         PORT_MEM_ACT_N_WIDTH                               : integer                                  := 0;
         PORT_MEM_ACT_N_PINLOC_0                            : integer                                  := 0;
         PORT_MEM_ACT_N_PINLOC_1                            : integer                                  := 0;
         PORT_MEM_ACT_N_PINLOC_AUTOGEN_WCNT                 : integer                                  := 0;
         PORT_MEM_PAR_WIDTH                                 : integer                                  := 0;
         PORT_MEM_PAR_PINLOC_0                              : integer                                  := 0;
         PORT_MEM_PAR_PINLOC_1                              : integer                                  := 0;
         PORT_MEM_PAR_PINLOC_AUTOGEN_WCNT                   : integer                                  := 0;
         PORT_MEM_CA_WIDTH                                  : integer                                  := 0;
         PORT_MEM_CA_PINLOC_0                               : integer                                  := 0;
         PORT_MEM_CA_PINLOC_1                               : integer                                  := 0;
         PORT_MEM_CA_PINLOC_2                               : integer                                  := 0;
         PORT_MEM_CA_PINLOC_3                               : integer                                  := 0;
         PORT_MEM_CA_PINLOC_4                               : integer                                  := 0;
         PORT_MEM_CA_PINLOC_5                               : integer                                  := 0;
         PORT_MEM_CA_PINLOC_6                               : integer                                  := 0;
         PORT_MEM_CA_PINLOC_7                               : integer                                  := 0;
         PORT_MEM_CA_PINLOC_8                               : integer                                  := 0;
         PORT_MEM_CA_PINLOC_9                               : integer                                  := 0;
         PORT_MEM_CA_PINLOC_10                              : integer                                  := 0;
         PORT_MEM_CA_PINLOC_11                              : integer                                  := 0;
         PORT_MEM_CA_PINLOC_12                              : integer                                  := 0;
         PORT_MEM_CA_PINLOC_13                              : integer                                  := 0;
         PORT_MEM_CA_PINLOC_14                              : integer                                  := 0;
         PORT_MEM_CA_PINLOC_15                              : integer                                  := 0;
         PORT_MEM_CA_PINLOC_16                              : integer                                  := 0;
         PORT_MEM_CA_PINLOC_AUTOGEN_WCNT                    : integer                                  := 0;
         PORT_MEM_REF_N_WIDTH                               : integer                                  := 0;
         PORT_MEM_REF_N_PINLOC_0                            : integer                                  := 0;
         PORT_MEM_REF_N_PINLOC_AUTOGEN_WCNT                 : integer                                  := 0;
         PORT_MEM_WPS_N_WIDTH                               : integer                                  := 0;
         PORT_MEM_WPS_N_PINLOC_0                            : integer                                  := 0;
         PORT_MEM_WPS_N_PINLOC_AUTOGEN_WCNT                 : integer                                  := 0;
         PORT_MEM_RPS_N_WIDTH                               : integer                                  := 0;
         PORT_MEM_RPS_N_PINLOC_0                            : integer                                  := 0;
         PORT_MEM_RPS_N_PINLOC_AUTOGEN_WCNT                 : integer                                  := 0;
         PORT_MEM_DOFF_N_WIDTH                              : integer                                  := 0;
         PORT_MEM_DOFF_N_PINLOC_0                           : integer                                  := 0;
         PORT_MEM_DOFF_N_PINLOC_AUTOGEN_WCNT                : integer                                  := 0;
         PORT_MEM_LDA_N_WIDTH                               : integer                                  := 0;
         PORT_MEM_LDA_N_PINLOC_0                            : integer                                  := 0;
         PORT_MEM_LDA_N_PINLOC_AUTOGEN_WCNT                 : integer                                  := 0;
         PORT_MEM_LDB_N_WIDTH                               : integer                                  := 0;
         PORT_MEM_LDB_N_PINLOC_0                            : integer                                  := 0;
         PORT_MEM_LDB_N_PINLOC_AUTOGEN_WCNT                 : integer                                  := 0;
         PORT_MEM_RWA_N_WIDTH                               : integer                                  := 0;
         PORT_MEM_RWA_N_PINLOC_0                            : integer                                  := 0;
         PORT_MEM_RWA_N_PINLOC_AUTOGEN_WCNT                 : integer                                  := 0;
         PORT_MEM_RWB_N_WIDTH                               : integer                                  := 0;
         PORT_MEM_RWB_N_PINLOC_0                            : integer                                  := 0;
         PORT_MEM_RWB_N_PINLOC_AUTOGEN_WCNT                 : integer                                  := 0;
         PORT_MEM_LBK0_N_WIDTH                              : integer                                  := 0;
         PORT_MEM_LBK0_N_PINLOC_0                           : integer                                  := 0;
         PORT_MEM_LBK0_N_PINLOC_AUTOGEN_WCNT                : integer                                  := 0;
         PORT_MEM_LBK1_N_WIDTH                              : integer                                  := 0;
         PORT_MEM_LBK1_N_PINLOC_0                           : integer                                  := 0;
         PORT_MEM_LBK1_N_PINLOC_AUTOGEN_WCNT                : integer                                  := 0;
         PORT_MEM_CFG_N_WIDTH                               : integer                                  := 0;
         PORT_MEM_CFG_N_PINLOC_0                            : integer                                  := 0;
         PORT_MEM_CFG_N_PINLOC_AUTOGEN_WCNT                 : integer                                  := 0;
         PORT_MEM_AP_WIDTH                                  : integer                                  := 0;
         PORT_MEM_AP_PINLOC_0                               : integer                                  := 0;
         PORT_MEM_AP_PINLOC_AUTOGEN_WCNT                    : integer                                  := 0;
         PORT_MEM_AINV_WIDTH                                : integer                                  := 0;
         PORT_MEM_AINV_PINLOC_0                             : integer                                  := 0;
         PORT_MEM_AINV_PINLOC_AUTOGEN_WCNT                  : integer                                  := 0;
         PORT_MEM_DM_WIDTH                                  : integer                                  := 0;
         PORT_MEM_DM_PINLOC_0                               : integer                                  := 0;
         PORT_MEM_DM_PINLOC_1                               : integer                                  := 0;
         PORT_MEM_DM_PINLOC_2                               : integer                                  := 0;
         PORT_MEM_DM_PINLOC_3                               : integer                                  := 0;
         PORT_MEM_DM_PINLOC_4                               : integer                                  := 0;
         PORT_MEM_DM_PINLOC_5                               : integer                                  := 0;
         PORT_MEM_DM_PINLOC_6                               : integer                                  := 0;
         PORT_MEM_DM_PINLOC_7                               : integer                                  := 0;
         PORT_MEM_DM_PINLOC_8                               : integer                                  := 0;
         PORT_MEM_DM_PINLOC_9                               : integer                                  := 0;
         PORT_MEM_DM_PINLOC_10                              : integer                                  := 0;
         PORT_MEM_DM_PINLOC_11                              : integer                                  := 0;
         PORT_MEM_DM_PINLOC_12                              : integer                                  := 0;
         PORT_MEM_DM_PINLOC_AUTOGEN_WCNT                    : integer                                  := 0;
         PORT_MEM_BWS_N_WIDTH                               : integer                                  := 0;
         PORT_MEM_BWS_N_PINLOC_0                            : integer                                  := 0;
         PORT_MEM_BWS_N_PINLOC_1                            : integer                                  := 0;
         PORT_MEM_BWS_N_PINLOC_2                            : integer                                  := 0;
         PORT_MEM_BWS_N_PINLOC_AUTOGEN_WCNT                 : integer                                  := 0;
         PORT_MEM_D_WIDTH                                   : integer                                  := 0;
         PORT_MEM_D_PINLOC_0                                : integer                                  := 0;
         PORT_MEM_D_PINLOC_1                                : integer                                  := 0;
         PORT_MEM_D_PINLOC_2                                : integer                                  := 0;
         PORT_MEM_D_PINLOC_3                                : integer                                  := 0;
         PORT_MEM_D_PINLOC_4                                : integer                                  := 0;
         PORT_MEM_D_PINLOC_5                                : integer                                  := 0;
         PORT_MEM_D_PINLOC_6                                : integer                                  := 0;
         PORT_MEM_D_PINLOC_7                                : integer                                  := 0;
         PORT_MEM_D_PINLOC_8                                : integer                                  := 0;
         PORT_MEM_D_PINLOC_9                                : integer                                  := 0;
         PORT_MEM_D_PINLOC_10                               : integer                                  := 0;
         PORT_MEM_D_PINLOC_11                               : integer                                  := 0;
         PORT_MEM_D_PINLOC_12                               : integer                                  := 0;
         PORT_MEM_D_PINLOC_13                               : integer                                  := 0;
         PORT_MEM_D_PINLOC_14                               : integer                                  := 0;
         PORT_MEM_D_PINLOC_15                               : integer                                  := 0;
         PORT_MEM_D_PINLOC_16                               : integer                                  := 0;
         PORT_MEM_D_PINLOC_17                               : integer                                  := 0;
         PORT_MEM_D_PINLOC_18                               : integer                                  := 0;
         PORT_MEM_D_PINLOC_19                               : integer                                  := 0;
         PORT_MEM_D_PINLOC_20                               : integer                                  := 0;
         PORT_MEM_D_PINLOC_21                               : integer                                  := 0;
         PORT_MEM_D_PINLOC_22                               : integer                                  := 0;
         PORT_MEM_D_PINLOC_23                               : integer                                  := 0;
         PORT_MEM_D_PINLOC_24                               : integer                                  := 0;
         PORT_MEM_D_PINLOC_25                               : integer                                  := 0;
         PORT_MEM_D_PINLOC_26                               : integer                                  := 0;
         PORT_MEM_D_PINLOC_27                               : integer                                  := 0;
         PORT_MEM_D_PINLOC_28                               : integer                                  := 0;
         PORT_MEM_D_PINLOC_29                               : integer                                  := 0;
         PORT_MEM_D_PINLOC_30                               : integer                                  := 0;
         PORT_MEM_D_PINLOC_31                               : integer                                  := 0;
         PORT_MEM_D_PINLOC_32                               : integer                                  := 0;
         PORT_MEM_D_PINLOC_33                               : integer                                  := 0;
         PORT_MEM_D_PINLOC_34                               : integer                                  := 0;
         PORT_MEM_D_PINLOC_35                               : integer                                  := 0;
         PORT_MEM_D_PINLOC_36                               : integer                                  := 0;
         PORT_MEM_D_PINLOC_37                               : integer                                  := 0;
         PORT_MEM_D_PINLOC_38                               : integer                                  := 0;
         PORT_MEM_D_PINLOC_39                               : integer                                  := 0;
         PORT_MEM_D_PINLOC_40                               : integer                                  := 0;
         PORT_MEM_D_PINLOC_41                               : integer                                  := 0;
         PORT_MEM_D_PINLOC_42                               : integer                                  := 0;
         PORT_MEM_D_PINLOC_43                               : integer                                  := 0;
         PORT_MEM_D_PINLOC_44                               : integer                                  := 0;
         PORT_MEM_D_PINLOC_45                               : integer                                  := 0;
         PORT_MEM_D_PINLOC_46                               : integer                                  := 0;
         PORT_MEM_D_PINLOC_47                               : integer                                  := 0;
         PORT_MEM_D_PINLOC_48                               : integer                                  := 0;
         PORT_MEM_D_PINLOC_AUTOGEN_WCNT                     : integer                                  := 0;
         PORT_MEM_DQ_WIDTH                                  : integer                                  := 0;
         PORT_MEM_DQ_PINLOC_0                               : integer                                  := 0;
         PORT_MEM_DQ_PINLOC_1                               : integer                                  := 0;
         PORT_MEM_DQ_PINLOC_2                               : integer                                  := 0;
         PORT_MEM_DQ_PINLOC_3                               : integer                                  := 0;
         PORT_MEM_DQ_PINLOC_4                               : integer                                  := 0;
         PORT_MEM_DQ_PINLOC_5                               : integer                                  := 0;
         PORT_MEM_DQ_PINLOC_6                               : integer                                  := 0;
         PORT_MEM_DQ_PINLOC_7                               : integer                                  := 0;
         PORT_MEM_DQ_PINLOC_8                               : integer                                  := 0;
         PORT_MEM_DQ_PINLOC_9                               : integer                                  := 0;
         PORT_MEM_DQ_PINLOC_10                              : integer                                  := 0;
         PORT_MEM_DQ_PINLOC_11                              : integer                                  := 0;
         PORT_MEM_DQ_PINLOC_12                              : integer                                  := 0;
         PORT_MEM_DQ_PINLOC_13                              : integer                                  := 0;
         PORT_MEM_DQ_PINLOC_14                              : integer                                  := 0;
         PORT_MEM_DQ_PINLOC_15                              : integer                                  := 0;
         PORT_MEM_DQ_PINLOC_16                              : integer                                  := 0;
         PORT_MEM_DQ_PINLOC_17                              : integer                                  := 0;
         PORT_MEM_DQ_PINLOC_18                              : integer                                  := 0;
         PORT_MEM_DQ_PINLOC_19                              : integer                                  := 0;
         PORT_MEM_DQ_PINLOC_20                              : integer                                  := 0;
         PORT_MEM_DQ_PINLOC_21                              : integer                                  := 0;
         PORT_MEM_DQ_PINLOC_22                              : integer                                  := 0;
         PORT_MEM_DQ_PINLOC_23                              : integer                                  := 0;
         PORT_MEM_DQ_PINLOC_24                              : integer                                  := 0;
         PORT_MEM_DQ_PINLOC_25                              : integer                                  := 0;
         PORT_MEM_DQ_PINLOC_26                              : integer                                  := 0;
         PORT_MEM_DQ_PINLOC_27                              : integer                                  := 0;
         PORT_MEM_DQ_PINLOC_28                              : integer                                  := 0;
         PORT_MEM_DQ_PINLOC_29                              : integer                                  := 0;
         PORT_MEM_DQ_PINLOC_30                              : integer                                  := 0;
         PORT_MEM_DQ_PINLOC_31                              : integer                                  := 0;
         PORT_MEM_DQ_PINLOC_32                              : integer                                  := 0;
         PORT_MEM_DQ_PINLOC_33                              : integer                                  := 0;
         PORT_MEM_DQ_PINLOC_34                              : integer                                  := 0;
         PORT_MEM_DQ_PINLOC_35                              : integer                                  := 0;
         PORT_MEM_DQ_PINLOC_36                              : integer                                  := 0;
         PORT_MEM_DQ_PINLOC_37                              : integer                                  := 0;
         PORT_MEM_DQ_PINLOC_38                              : integer                                  := 0;
         PORT_MEM_DQ_PINLOC_39                              : integer                                  := 0;
         PORT_MEM_DQ_PINLOC_40                              : integer                                  := 0;
         PORT_MEM_DQ_PINLOC_41                              : integer                                  := 0;
         PORT_MEM_DQ_PINLOC_42                              : integer                                  := 0;
         PORT_MEM_DQ_PINLOC_43                              : integer                                  := 0;
         PORT_MEM_DQ_PINLOC_44                              : integer                                  := 0;
         PORT_MEM_DQ_PINLOC_45                              : integer                                  := 0;
         PORT_MEM_DQ_PINLOC_46                              : integer                                  := 0;
         PORT_MEM_DQ_PINLOC_47                              : integer                                  := 0;
         PORT_MEM_DQ_PINLOC_48                              : integer                                  := 0;
         PORT_MEM_DQ_PINLOC_AUTOGEN_WCNT                    : integer                                  := 0;
         PORT_MEM_DBI_N_WIDTH                               : integer                                  := 0;
         PORT_MEM_DBI_N_PINLOC_0                            : integer                                  := 0;
         PORT_MEM_DBI_N_PINLOC_1                            : integer                                  := 0;
         PORT_MEM_DBI_N_PINLOC_2                            : integer                                  := 0;
         PORT_MEM_DBI_N_PINLOC_3                            : integer                                  := 0;
         PORT_MEM_DBI_N_PINLOC_4                            : integer                                  := 0;
         PORT_MEM_DBI_N_PINLOC_5                            : integer                                  := 0;
         PORT_MEM_DBI_N_PINLOC_6                            : integer                                  := 0;
         PORT_MEM_DBI_N_PINLOC_AUTOGEN_WCNT                 : integer                                  := 0;
         PORT_MEM_DQA_WIDTH                                 : integer                                  := 0;
         PORT_MEM_DQA_PINLOC_0                              : integer                                  := 0;
         PORT_MEM_DQA_PINLOC_1                              : integer                                  := 0;
         PORT_MEM_DQA_PINLOC_2                              : integer                                  := 0;
         PORT_MEM_DQA_PINLOC_3                              : integer                                  := 0;
         PORT_MEM_DQA_PINLOC_4                              : integer                                  := 0;
         PORT_MEM_DQA_PINLOC_5                              : integer                                  := 0;
         PORT_MEM_DQA_PINLOC_6                              : integer                                  := 0;
         PORT_MEM_DQA_PINLOC_7                              : integer                                  := 0;
         PORT_MEM_DQA_PINLOC_8                              : integer                                  := 0;
         PORT_MEM_DQA_PINLOC_9                              : integer                                  := 0;
         PORT_MEM_DQA_PINLOC_10                             : integer                                  := 0;
         PORT_MEM_DQA_PINLOC_11                             : integer                                  := 0;
         PORT_MEM_DQA_PINLOC_12                             : integer                                  := 0;
         PORT_MEM_DQA_PINLOC_13                             : integer                                  := 0;
         PORT_MEM_DQA_PINLOC_14                             : integer                                  := 0;
         PORT_MEM_DQA_PINLOC_15                             : integer                                  := 0;
         PORT_MEM_DQA_PINLOC_16                             : integer                                  := 0;
         PORT_MEM_DQA_PINLOC_17                             : integer                                  := 0;
         PORT_MEM_DQA_PINLOC_18                             : integer                                  := 0;
         PORT_MEM_DQA_PINLOC_19                             : integer                                  := 0;
         PORT_MEM_DQA_PINLOC_20                             : integer                                  := 0;
         PORT_MEM_DQA_PINLOC_21                             : integer                                  := 0;
         PORT_MEM_DQA_PINLOC_22                             : integer                                  := 0;
         PORT_MEM_DQA_PINLOC_23                             : integer                                  := 0;
         PORT_MEM_DQA_PINLOC_24                             : integer                                  := 0;
         PORT_MEM_DQA_PINLOC_25                             : integer                                  := 0;
         PORT_MEM_DQA_PINLOC_26                             : integer                                  := 0;
         PORT_MEM_DQA_PINLOC_27                             : integer                                  := 0;
         PORT_MEM_DQA_PINLOC_28                             : integer                                  := 0;
         PORT_MEM_DQA_PINLOC_29                             : integer                                  := 0;
         PORT_MEM_DQA_PINLOC_30                             : integer                                  := 0;
         PORT_MEM_DQA_PINLOC_31                             : integer                                  := 0;
         PORT_MEM_DQA_PINLOC_32                             : integer                                  := 0;
         PORT_MEM_DQA_PINLOC_33                             : integer                                  := 0;
         PORT_MEM_DQA_PINLOC_34                             : integer                                  := 0;
         PORT_MEM_DQA_PINLOC_35                             : integer                                  := 0;
         PORT_MEM_DQA_PINLOC_36                             : integer                                  := 0;
         PORT_MEM_DQA_PINLOC_37                             : integer                                  := 0;
         PORT_MEM_DQA_PINLOC_38                             : integer                                  := 0;
         PORT_MEM_DQA_PINLOC_39                             : integer                                  := 0;
         PORT_MEM_DQA_PINLOC_40                             : integer                                  := 0;
         PORT_MEM_DQA_PINLOC_41                             : integer                                  := 0;
         PORT_MEM_DQA_PINLOC_42                             : integer                                  := 0;
         PORT_MEM_DQA_PINLOC_43                             : integer                                  := 0;
         PORT_MEM_DQA_PINLOC_44                             : integer                                  := 0;
         PORT_MEM_DQA_PINLOC_45                             : integer                                  := 0;
         PORT_MEM_DQA_PINLOC_46                             : integer                                  := 0;
         PORT_MEM_DQA_PINLOC_47                             : integer                                  := 0;
         PORT_MEM_DQA_PINLOC_48                             : integer                                  := 0;
         PORT_MEM_DQA_PINLOC_AUTOGEN_WCNT                   : integer                                  := 0;
         PORT_MEM_DQB_WIDTH                                 : integer                                  := 0;
         PORT_MEM_DQB_PINLOC_0                              : integer                                  := 0;
         PORT_MEM_DQB_PINLOC_1                              : integer                                  := 0;
         PORT_MEM_DQB_PINLOC_2                              : integer                                  := 0;
         PORT_MEM_DQB_PINLOC_3                              : integer                                  := 0;
         PORT_MEM_DQB_PINLOC_4                              : integer                                  := 0;
         PORT_MEM_DQB_PINLOC_5                              : integer                                  := 0;
         PORT_MEM_DQB_PINLOC_6                              : integer                                  := 0;
         PORT_MEM_DQB_PINLOC_7                              : integer                                  := 0;
         PORT_MEM_DQB_PINLOC_8                              : integer                                  := 0;
         PORT_MEM_DQB_PINLOC_9                              : integer                                  := 0;
         PORT_MEM_DQB_PINLOC_10                             : integer                                  := 0;
         PORT_MEM_DQB_PINLOC_11                             : integer                                  := 0;
         PORT_MEM_DQB_PINLOC_12                             : integer                                  := 0;
         PORT_MEM_DQB_PINLOC_13                             : integer                                  := 0;
         PORT_MEM_DQB_PINLOC_14                             : integer                                  := 0;
         PORT_MEM_DQB_PINLOC_15                             : integer                                  := 0;
         PORT_MEM_DQB_PINLOC_16                             : integer                                  := 0;
         PORT_MEM_DQB_PINLOC_17                             : integer                                  := 0;
         PORT_MEM_DQB_PINLOC_18                             : integer                                  := 0;
         PORT_MEM_DQB_PINLOC_19                             : integer                                  := 0;
         PORT_MEM_DQB_PINLOC_20                             : integer                                  := 0;
         PORT_MEM_DQB_PINLOC_21                             : integer                                  := 0;
         PORT_MEM_DQB_PINLOC_22                             : integer                                  := 0;
         PORT_MEM_DQB_PINLOC_23                             : integer                                  := 0;
         PORT_MEM_DQB_PINLOC_24                             : integer                                  := 0;
         PORT_MEM_DQB_PINLOC_25                             : integer                                  := 0;
         PORT_MEM_DQB_PINLOC_26                             : integer                                  := 0;
         PORT_MEM_DQB_PINLOC_27                             : integer                                  := 0;
         PORT_MEM_DQB_PINLOC_28                             : integer                                  := 0;
         PORT_MEM_DQB_PINLOC_29                             : integer                                  := 0;
         PORT_MEM_DQB_PINLOC_30                             : integer                                  := 0;
         PORT_MEM_DQB_PINLOC_31                             : integer                                  := 0;
         PORT_MEM_DQB_PINLOC_32                             : integer                                  := 0;
         PORT_MEM_DQB_PINLOC_33                             : integer                                  := 0;
         PORT_MEM_DQB_PINLOC_34                             : integer                                  := 0;
         PORT_MEM_DQB_PINLOC_35                             : integer                                  := 0;
         PORT_MEM_DQB_PINLOC_36                             : integer                                  := 0;
         PORT_MEM_DQB_PINLOC_37                             : integer                                  := 0;
         PORT_MEM_DQB_PINLOC_38                             : integer                                  := 0;
         PORT_MEM_DQB_PINLOC_39                             : integer                                  := 0;
         PORT_MEM_DQB_PINLOC_40                             : integer                                  := 0;
         PORT_MEM_DQB_PINLOC_41                             : integer                                  := 0;
         PORT_MEM_DQB_PINLOC_42                             : integer                                  := 0;
         PORT_MEM_DQB_PINLOC_43                             : integer                                  := 0;
         PORT_MEM_DQB_PINLOC_44                             : integer                                  := 0;
         PORT_MEM_DQB_PINLOC_45                             : integer                                  := 0;
         PORT_MEM_DQB_PINLOC_46                             : integer                                  := 0;
         PORT_MEM_DQB_PINLOC_47                             : integer                                  := 0;
         PORT_MEM_DQB_PINLOC_48                             : integer                                  := 0;
         PORT_MEM_DQB_PINLOC_AUTOGEN_WCNT                   : integer                                  := 0;
         PORT_MEM_DINVA_WIDTH                               : integer                                  := 0;
         PORT_MEM_DINVA_PINLOC_0                            : integer                                  := 0;
         PORT_MEM_DINVA_PINLOC_1                            : integer                                  := 0;
         PORT_MEM_DINVA_PINLOC_2                            : integer                                  := 0;
         PORT_MEM_DINVA_PINLOC_AUTOGEN_WCNT                 : integer                                  := 0;
         PORT_MEM_DINVB_WIDTH                               : integer                                  := 0;
         PORT_MEM_DINVB_PINLOC_0                            : integer                                  := 0;
         PORT_MEM_DINVB_PINLOC_1                            : integer                                  := 0;
         PORT_MEM_DINVB_PINLOC_2                            : integer                                  := 0;
         PORT_MEM_DINVB_PINLOC_AUTOGEN_WCNT                 : integer                                  := 0;
         PORT_MEM_Q_WIDTH                                   : integer                                  := 0;
         PORT_MEM_Q_PINLOC_0                                : integer                                  := 0;
         PORT_MEM_Q_PINLOC_1                                : integer                                  := 0;
         PORT_MEM_Q_PINLOC_2                                : integer                                  := 0;
         PORT_MEM_Q_PINLOC_3                                : integer                                  := 0;
         PORT_MEM_Q_PINLOC_4                                : integer                                  := 0;
         PORT_MEM_Q_PINLOC_5                                : integer                                  := 0;
         PORT_MEM_Q_PINLOC_6                                : integer                                  := 0;
         PORT_MEM_Q_PINLOC_7                                : integer                                  := 0;
         PORT_MEM_Q_PINLOC_8                                : integer                                  := 0;
         PORT_MEM_Q_PINLOC_9                                : integer                                  := 0;
         PORT_MEM_Q_PINLOC_10                               : integer                                  := 0;
         PORT_MEM_Q_PINLOC_11                               : integer                                  := 0;
         PORT_MEM_Q_PINLOC_12                               : integer                                  := 0;
         PORT_MEM_Q_PINLOC_13                               : integer                                  := 0;
         PORT_MEM_Q_PINLOC_14                               : integer                                  := 0;
         PORT_MEM_Q_PINLOC_15                               : integer                                  := 0;
         PORT_MEM_Q_PINLOC_16                               : integer                                  := 0;
         PORT_MEM_Q_PINLOC_17                               : integer                                  := 0;
         PORT_MEM_Q_PINLOC_18                               : integer                                  := 0;
         PORT_MEM_Q_PINLOC_19                               : integer                                  := 0;
         PORT_MEM_Q_PINLOC_20                               : integer                                  := 0;
         PORT_MEM_Q_PINLOC_21                               : integer                                  := 0;
         PORT_MEM_Q_PINLOC_22                               : integer                                  := 0;
         PORT_MEM_Q_PINLOC_23                               : integer                                  := 0;
         PORT_MEM_Q_PINLOC_24                               : integer                                  := 0;
         PORT_MEM_Q_PINLOC_25                               : integer                                  := 0;
         PORT_MEM_Q_PINLOC_26                               : integer                                  := 0;
         PORT_MEM_Q_PINLOC_27                               : integer                                  := 0;
         PORT_MEM_Q_PINLOC_28                               : integer                                  := 0;
         PORT_MEM_Q_PINLOC_29                               : integer                                  := 0;
         PORT_MEM_Q_PINLOC_30                               : integer                                  := 0;
         PORT_MEM_Q_PINLOC_31                               : integer                                  := 0;
         PORT_MEM_Q_PINLOC_32                               : integer                                  := 0;
         PORT_MEM_Q_PINLOC_33                               : integer                                  := 0;
         PORT_MEM_Q_PINLOC_34                               : integer                                  := 0;
         PORT_MEM_Q_PINLOC_35                               : integer                                  := 0;
         PORT_MEM_Q_PINLOC_36                               : integer                                  := 0;
         PORT_MEM_Q_PINLOC_37                               : integer                                  := 0;
         PORT_MEM_Q_PINLOC_38                               : integer                                  := 0;
         PORT_MEM_Q_PINLOC_39                               : integer                                  := 0;
         PORT_MEM_Q_PINLOC_40                               : integer                                  := 0;
         PORT_MEM_Q_PINLOC_41                               : integer                                  := 0;
         PORT_MEM_Q_PINLOC_42                               : integer                                  := 0;
         PORT_MEM_Q_PINLOC_43                               : integer                                  := 0;
         PORT_MEM_Q_PINLOC_44                               : integer                                  := 0;
         PORT_MEM_Q_PINLOC_45                               : integer                                  := 0;
         PORT_MEM_Q_PINLOC_46                               : integer                                  := 0;
         PORT_MEM_Q_PINLOC_47                               : integer                                  := 0;
         PORT_MEM_Q_PINLOC_48                               : integer                                  := 0;
         PORT_MEM_Q_PINLOC_AUTOGEN_WCNT                     : integer                                  := 0;
         PORT_MEM_DQS_WIDTH                                 : integer                                  := 0;
         PORT_MEM_DQS_PINLOC_0                              : integer                                  := 0;
         PORT_MEM_DQS_PINLOC_1                              : integer                                  := 0;
         PORT_MEM_DQS_PINLOC_2                              : integer                                  := 0;
         PORT_MEM_DQS_PINLOC_3                              : integer                                  := 0;
         PORT_MEM_DQS_PINLOC_4                              : integer                                  := 0;
         PORT_MEM_DQS_PINLOC_5                              : integer                                  := 0;
         PORT_MEM_DQS_PINLOC_6                              : integer                                  := 0;
         PORT_MEM_DQS_PINLOC_7                              : integer                                  := 0;
         PORT_MEM_DQS_PINLOC_8                              : integer                                  := 0;
         PORT_MEM_DQS_PINLOC_9                              : integer                                  := 0;
         PORT_MEM_DQS_PINLOC_10                             : integer                                  := 0;
         PORT_MEM_DQS_PINLOC_11                             : integer                                  := 0;
         PORT_MEM_DQS_PINLOC_12                             : integer                                  := 0;
         PORT_MEM_DQS_PINLOC_AUTOGEN_WCNT                   : integer                                  := 0;
         PORT_MEM_DQS_N_WIDTH                               : integer                                  := 0;
         PORT_MEM_DQS_N_PINLOC_0                            : integer                                  := 0;
         PORT_MEM_DQS_N_PINLOC_1                            : integer                                  := 0;
         PORT_MEM_DQS_N_PINLOC_2                            : integer                                  := 0;
         PORT_MEM_DQS_N_PINLOC_3                            : integer                                  := 0;
         PORT_MEM_DQS_N_PINLOC_4                            : integer                                  := 0;
         PORT_MEM_DQS_N_PINLOC_5                            : integer                                  := 0;
         PORT_MEM_DQS_N_PINLOC_6                            : integer                                  := 0;
         PORT_MEM_DQS_N_PINLOC_7                            : integer                                  := 0;
         PORT_MEM_DQS_N_PINLOC_8                            : integer                                  := 0;
         PORT_MEM_DQS_N_PINLOC_9                            : integer                                  := 0;
         PORT_MEM_DQS_N_PINLOC_10                           : integer                                  := 0;
         PORT_MEM_DQS_N_PINLOC_11                           : integer                                  := 0;
         PORT_MEM_DQS_N_PINLOC_12                           : integer                                  := 0;
         PORT_MEM_DQS_N_PINLOC_AUTOGEN_WCNT                 : integer                                  := 0;
         PORT_MEM_QK_WIDTH                                  : integer                                  := 0;
         PORT_MEM_QK_PINLOC_0                               : integer                                  := 0;
         PORT_MEM_QK_PINLOC_1                               : integer                                  := 0;
         PORT_MEM_QK_PINLOC_2                               : integer                                  := 0;
         PORT_MEM_QK_PINLOC_3                               : integer                                  := 0;
         PORT_MEM_QK_PINLOC_4                               : integer                                  := 0;
         PORT_MEM_QK_PINLOC_5                               : integer                                  := 0;
         PORT_MEM_QK_PINLOC_AUTOGEN_WCNT                    : integer                                  := 0;
         PORT_MEM_QK_N_WIDTH                                : integer                                  := 0;
         PORT_MEM_QK_N_PINLOC_0                             : integer                                  := 0;
         PORT_MEM_QK_N_PINLOC_1                             : integer                                  := 0;
         PORT_MEM_QK_N_PINLOC_2                             : integer                                  := 0;
         PORT_MEM_QK_N_PINLOC_3                             : integer                                  := 0;
         PORT_MEM_QK_N_PINLOC_4                             : integer                                  := 0;
         PORT_MEM_QK_N_PINLOC_5                             : integer                                  := 0;
         PORT_MEM_QK_N_PINLOC_AUTOGEN_WCNT                  : integer                                  := 0;
         PORT_MEM_QKA_WIDTH                                 : integer                                  := 0;
         PORT_MEM_QKA_PINLOC_0                              : integer                                  := 0;
         PORT_MEM_QKA_PINLOC_1                              : integer                                  := 0;
         PORT_MEM_QKA_PINLOC_2                              : integer                                  := 0;
         PORT_MEM_QKA_PINLOC_3                              : integer                                  := 0;
         PORT_MEM_QKA_PINLOC_4                              : integer                                  := 0;
         PORT_MEM_QKA_PINLOC_5                              : integer                                  := 0;
         PORT_MEM_QKA_PINLOC_AUTOGEN_WCNT                   : integer                                  := 0;
         PORT_MEM_QKA_N_WIDTH                               : integer                                  := 0;
         PORT_MEM_QKA_N_PINLOC_0                            : integer                                  := 0;
         PORT_MEM_QKA_N_PINLOC_1                            : integer                                  := 0;
         PORT_MEM_QKA_N_PINLOC_2                            : integer                                  := 0;
         PORT_MEM_QKA_N_PINLOC_3                            : integer                                  := 0;
         PORT_MEM_QKA_N_PINLOC_4                            : integer                                  := 0;
         PORT_MEM_QKA_N_PINLOC_5                            : integer                                  := 0;
         PORT_MEM_QKA_N_PINLOC_AUTOGEN_WCNT                 : integer                                  := 0;
         PORT_MEM_QKB_WIDTH                                 : integer                                  := 0;
         PORT_MEM_QKB_PINLOC_0                              : integer                                  := 0;
         PORT_MEM_QKB_PINLOC_1                              : integer                                  := 0;
         PORT_MEM_QKB_PINLOC_2                              : integer                                  := 0;
         PORT_MEM_QKB_PINLOC_3                              : integer                                  := 0;
         PORT_MEM_QKB_PINLOC_4                              : integer                                  := 0;
         PORT_MEM_QKB_PINLOC_5                              : integer                                  := 0;
         PORT_MEM_QKB_PINLOC_AUTOGEN_WCNT                   : integer                                  := 0;
         PORT_MEM_QKB_N_WIDTH                               : integer                                  := 0;
         PORT_MEM_QKB_N_PINLOC_0                            : integer                                  := 0;
         PORT_MEM_QKB_N_PINLOC_1                            : integer                                  := 0;
         PORT_MEM_QKB_N_PINLOC_2                            : integer                                  := 0;
         PORT_MEM_QKB_N_PINLOC_3                            : integer                                  := 0;
         PORT_MEM_QKB_N_PINLOC_4                            : integer                                  := 0;
         PORT_MEM_QKB_N_PINLOC_5                            : integer                                  := 0;
         PORT_MEM_QKB_N_PINLOC_AUTOGEN_WCNT                 : integer                                  := 0;
         PORT_MEM_CQ_WIDTH                                  : integer                                  := 0;
         PORT_MEM_CQ_PINLOC_0                               : integer                                  := 0;
         PORT_MEM_CQ_PINLOC_1                               : integer                                  := 0;
         PORT_MEM_CQ_PINLOC_AUTOGEN_WCNT                    : integer                                  := 0;
         PORT_MEM_CQ_N_WIDTH                                : integer                                  := 0;
         PORT_MEM_CQ_N_PINLOC_0                             : integer                                  := 0;
         PORT_MEM_CQ_N_PINLOC_1                             : integer                                  := 0;
         PORT_MEM_CQ_N_PINLOC_AUTOGEN_WCNT                  : integer                                  := 0;
         PORT_MEM_ALERT_N_WIDTH                             : integer                                  := 0;
         PORT_MEM_ALERT_N_PINLOC_0                          : integer                                  := 0;
         PORT_MEM_ALERT_N_PINLOC_1                          : integer                                  := 0;
         PORT_MEM_ALERT_N_PINLOC_AUTOGEN_WCNT               : integer                                  := 0;
         PORT_MEM_PE_N_WIDTH                                : integer                                  := 0;
         PORT_MEM_PE_N_PINLOC_0                             : integer                                  := 0;
         PORT_MEM_PE_N_PINLOC_1                             : integer                                  := 0;
         PORT_MEM_PE_N_PINLOC_AUTOGEN_WCNT                  : integer                                  := 0;
         PORT_CLKS_SHARING_MASTER_OUT_WIDTH                 : integer                                  := 0;
         PORT_CLKS_SHARING_SLAVE_IN_WIDTH                   : integer                                  := 0;
         PORT_CLKS_SHARING_SLAVE_OUT_WIDTH                  : integer                                  := 0;
         PORT_AFI_RLAT_WIDTH                                : integer                                  := 0;
         PORT_AFI_WLAT_WIDTH                                : integer                                  := 0;
         PORT_AFI_SEQ_BUSY_WIDTH                            : integer                                  := 0;
         PORT_AFI_ADDR_WIDTH                                : integer                                  := 0;
         PORT_AFI_BA_WIDTH                                  : integer                                  := 0;
         PORT_AFI_BG_WIDTH                                  : integer                                  := 0;
         PORT_AFI_C_WIDTH                                   : integer                                  := 0;
         PORT_AFI_CKE_WIDTH                                 : integer                                  := 0;
         PORT_AFI_CS_N_WIDTH                                : integer                                  := 0;
         PORT_AFI_RM_WIDTH                                  : integer                                  := 0;
         PORT_AFI_ODT_WIDTH                                 : integer                                  := 0;
         PORT_AFI_RAS_N_WIDTH                               : integer                                  := 0;
         PORT_AFI_CAS_N_WIDTH                               : integer                                  := 0;
         PORT_AFI_WE_N_WIDTH                                : integer                                  := 0;
         PORT_AFI_RST_N_WIDTH                               : integer                                  := 0;
         PORT_AFI_ACT_N_WIDTH                               : integer                                  := 0;
         PORT_AFI_REQ_N_WIDTH                               : integer                                  := 0;
         PORT_AFI_GNT_N_WIDTH                               : integer                                  := 0;
         PORT_AFI_ERR_N_WIDTH                               : integer                                  := 0;
         PORT_AFI_PAR_WIDTH                                 : integer                                  := 0;
         PORT_AFI_CA_WIDTH                                  : integer                                  := 0;
         PORT_AFI_REF_N_WIDTH                               : integer                                  := 0;
         PORT_AFI_WPS_N_WIDTH                               : integer                                  := 0;
         PORT_AFI_RPS_N_WIDTH                               : integer                                  := 0;
         PORT_AFI_DOFF_N_WIDTH                              : integer                                  := 0;
         PORT_AFI_LD_N_WIDTH                                : integer                                  := 0;
         PORT_AFI_RW_N_WIDTH                                : integer                                  := 0;
         PORT_AFI_LBK0_N_WIDTH                              : integer                                  := 0;
         PORT_AFI_LBK1_N_WIDTH                              : integer                                  := 0;
         PORT_AFI_CFG_N_WIDTH                               : integer                                  := 0;
         PORT_AFI_AP_WIDTH                                  : integer                                  := 0;
         PORT_AFI_AINV_WIDTH                                : integer                                  := 0;
         PORT_AFI_DM_WIDTH                                  : integer                                  := 0;
         PORT_AFI_DM_N_WIDTH                                : integer                                  := 0;
         PORT_AFI_BWS_N_WIDTH                               : integer                                  := 0;
         PORT_AFI_RDATA_DBI_N_WIDTH                         : integer                                  := 0;
         PORT_AFI_WDATA_DBI_N_WIDTH                         : integer                                  := 0;
         PORT_AFI_RDATA_DINV_WIDTH                          : integer                                  := 0;
         PORT_AFI_WDATA_DINV_WIDTH                          : integer                                  := 0;
         PORT_AFI_DQS_BURST_WIDTH                           : integer                                  := 0;
         PORT_AFI_WDATA_VALID_WIDTH                         : integer                                  := 0;
         PORT_AFI_WDATA_WIDTH                               : integer                                  := 0;
         PORT_AFI_RDATA_EN_FULL_WIDTH                       : integer                                  := 0;
         PORT_AFI_RDATA_WIDTH                               : integer                                  := 0;
         PORT_AFI_RDATA_VALID_WIDTH                         : integer                                  := 0;
         PORT_AFI_RRANK_WIDTH                               : integer                                  := 0;
         PORT_AFI_WRANK_WIDTH                               : integer                                  := 0;
         PORT_AFI_ALERT_N_WIDTH                             : integer                                  := 0;
         PORT_AFI_PE_N_WIDTH                                : integer                                  := 0;
         PORT_CTRL_AST_CMD_DATA_WIDTH                       : integer                                  := 0;
         PORT_CTRL_AST_WR_DATA_WIDTH                        : integer                                  := 0;
         PORT_CTRL_AST_RD_DATA_WIDTH                        : integer                                  := 0;
         PORT_CTRL_AMM_ADDRESS_WIDTH                        : integer                                  := 0;
         PORT_CTRL_AMM_RDATA_WIDTH                          : integer                                  := 0;
         PORT_CTRL_AMM_WDATA_WIDTH                          : integer                                  := 0;
         PORT_CTRL_AMM_BCOUNT_WIDTH                         : integer                                  := 0;
         PORT_CTRL_AMM_BYTEEN_WIDTH                         : integer                                  := 0;
         PORT_CTRL_USER_REFRESH_REQ_WIDTH                   : integer                                  := 0;
         PORT_CTRL_USER_REFRESH_BANK_WIDTH                  : integer                                  := 0;
         PORT_CTRL_SELF_REFRESH_REQ_WIDTH                   : integer                                  := 0;
         PORT_CTRL_ECC_WRITE_INFO_WIDTH                     : integer                                  := 0;
         PORT_CTRL_ECC_RDATA_ID_WIDTH                       : integer                                  := 0;
         PORT_CTRL_ECC_READ_INFO_WIDTH                      : integer                                  := 0;
         PORT_CTRL_ECC_CMD_INFO_WIDTH                       : integer                                  := 0;
         PORT_CTRL_ECC_WB_POINTER_WIDTH                     : integer                                  := 0;
         PORT_CTRL_MMR_SLAVE_ADDRESS_WIDTH                  : integer                                  := 0;
         PORT_CTRL_MMR_SLAVE_RDATA_WIDTH                    : integer                                  := 0;
         PORT_CTRL_MMR_SLAVE_WDATA_WIDTH                    : integer                                  := 0;
         PORT_CTRL_MMR_SLAVE_BCOUNT_WIDTH                   : integer                                  := 0;
         PORT_HPS_EMIF_H2E_WIDTH                            : integer                                  := 0;
         PORT_HPS_EMIF_E2H_WIDTH                            : integer                                  := 0;
         PORT_HPS_EMIF_H2E_GP_WIDTH                         : integer                                  := 0;
         PORT_HPS_EMIF_E2H_GP_WIDTH                         : integer                                  := 0;
         PORT_CAL_DEBUG_ADDRESS_WIDTH                       : integer                                  := 0;
         PORT_CAL_DEBUG_RDATA_WIDTH                         : integer                                  := 0;
         PORT_CAL_DEBUG_WDATA_WIDTH                         : integer                                  := 0;
         PORT_CAL_DEBUG_BYTEEN_WIDTH                        : integer                                  := 0;
         PORT_CAL_DEBUG_OUT_ADDRESS_WIDTH                   : integer                                  := 0;
         PORT_CAL_DEBUG_OUT_RDATA_WIDTH                     : integer                                  := 0;
         PORT_CAL_DEBUG_OUT_WDATA_WIDTH                     : integer                                  := 0;
         PORT_CAL_DEBUG_OUT_BYTEEN_WIDTH                    : integer                                  := 0;
         PORT_CAL_MASTER_ADDRESS_WIDTH                      : integer                                  := 0;
         PORT_CAL_MASTER_RDATA_WIDTH                        : integer                                  := 0;
         PORT_CAL_MASTER_WDATA_WIDTH                        : integer                                  := 0;
         PORT_CAL_MASTER_BYTEEN_WIDTH                       : integer                                  := 0;
         PORT_DFT_NF_IOAUX_PIO_IN_WIDTH                     : integer                                  := 0;
         PORT_DFT_NF_IOAUX_PIO_OUT_WIDTH                    : integer                                  := 0;
         PORT_DFT_NF_PA_DPRIO_REG_ADDR_WIDTH                : integer                                  := 0;
         PORT_DFT_NF_PA_DPRIO_WRITEDATA_WIDTH               : integer                                  := 0;
         PORT_DFT_NF_PA_DPRIO_READDATA_WIDTH                : integer                                  := 0;
         PORT_DFT_NF_PLL_CNTSEL_WIDTH                       : integer                                  := 0;
         PORT_DFT_NF_PLL_NUM_SHIFT_WIDTH                    : integer                                  := 0;
         PORT_DFT_NF_PLL_CORE_REFCLK_WIDTH                  : integer                                  := 0;
         PORT_DFT_NF_CORE_CLK_BUF_OUT_WIDTH                 : integer                                  := 0;
         PORT_DFT_NF_CORE_CLK_LOCKED_WIDTH                  : integer                                  := 0;
         PLL_VCO_FREQ_MHZ_INT                               : integer                                  := 0;
         PLL_VCO_TO_MEM_CLK_FREQ_RATIO                      : integer                                  := 0;
         PLL_PHY_CLK_VCO_PHASE                              : integer                                  := 0;
         PLL_VCO_FREQ_PS_STR                                : string                                   := "";
         PLL_REF_CLK_FREQ_PS_STR                            : string                                   := "";
         PLL_REF_CLK_FREQ_PS                                : integer                                  := 0;
         PLL_SIM_VCO_FREQ_PS                                : integer                                  := 0;
         PLL_SIM_PHYCLK_0_FREQ_PS                           : integer                                  := 0;
         PLL_SIM_PHYCLK_1_FREQ_PS                           : integer                                  := 0;
         PLL_SIM_PHYCLK_FB_FREQ_PS                          : integer                                  := 0;
         PLL_SIM_PHY_CLK_VCO_PHASE_PS                       : integer                                  := 0;
         PLL_SIM_CAL_SLAVE_CLK_FREQ_PS                      : integer                                  := 0;
         PLL_SIM_CAL_MASTER_CLK_FREQ_PS                     : integer                                  := 0;
         PLL_M_CNT_HIGH                                     : integer                                  := 0;
         PLL_M_CNT_LOW                                      : integer                                  := 0;
         PLL_N_CNT_HIGH                                     : integer                                  := 0;
         PLL_N_CNT_LOW                                      : integer                                  := 0;
         PLL_M_CNT_BYPASS_EN                                : string                                   := "";
         PLL_N_CNT_BYPASS_EN                                : string                                   := "";
         PLL_M_CNT_EVEN_DUTY_EN                             : string                                   := "";
         PLL_N_CNT_EVEN_DUTY_EN                             : string                                   := "";
         PLL_FBCLK_MUX_1                                    : string                                   := "";
         PLL_FBCLK_MUX_2                                    : string                                   := "";
         PLL_M_CNT_IN_SRC                                   : string                                   := "";
         PLL_CP_SETTING                                     : string                                   := "";
         PLL_BW_CTRL                                        : string                                   := "";
         PLL_BW_SEL                                         : string                                   := "";
         PLL_C_CNT_HIGH_0                                   : integer                                  := 0;
         PLL_C_CNT_LOW_0                                    : integer                                  := 0;
         PLL_C_CNT_PRST_0                                   : integer                                  := 0;
         PLL_C_CNT_PH_MUX_PRST_0                            : integer                                  := 0;
         PLL_C_CNT_BYPASS_EN_0                              : string                                   := "";
         PLL_C_CNT_EVEN_DUTY_EN_0                           : string                                   := "";
         PLL_C_CNT_FREQ_PS_STR_0                            : string                                   := "";
         PLL_C_CNT_PHASE_PS_STR_0                           : string                                   := "";
         PLL_C_CNT_DUTY_CYCLE_0                             : integer                                  := 0;
         PLL_C_CNT_OUT_EN_0                                 : string                                   := "";
         PLL_C_CNT_HIGH_1                                   : integer                                  := 0;
         PLL_C_CNT_LOW_1                                    : integer                                  := 0;
         PLL_C_CNT_PRST_1                                   : integer                                  := 0;
         PLL_C_CNT_PH_MUX_PRST_1                            : integer                                  := 0;
         PLL_C_CNT_BYPASS_EN_1                              : string                                   := "";
         PLL_C_CNT_EVEN_DUTY_EN_1                           : string                                   := "";
         PLL_C_CNT_FREQ_PS_STR_1                            : string                                   := "";
         PLL_C_CNT_PHASE_PS_STR_1                           : string                                   := "";
         PLL_C_CNT_DUTY_CYCLE_1                             : integer                                  := 0;
         PLL_C_CNT_OUT_EN_1                                 : string                                   := "";
         PLL_C_CNT_HIGH_2                                   : integer                                  := 0;
         PLL_C_CNT_LOW_2                                    : integer                                  := 0;
         PLL_C_CNT_PRST_2                                   : integer                                  := 0;
         PLL_C_CNT_PH_MUX_PRST_2                            : integer                                  := 0;
         PLL_C_CNT_BYPASS_EN_2                              : string                                   := "";
         PLL_C_CNT_EVEN_DUTY_EN_2                           : string                                   := "";
         PLL_C_CNT_FREQ_PS_STR_2                            : string                                   := "";
         PLL_C_CNT_PHASE_PS_STR_2                           : string                                   := "";
         PLL_C_CNT_DUTY_CYCLE_2                             : integer                                  := 0;
         PLL_C_CNT_OUT_EN_2                                 : string                                   := "";
         PLL_C_CNT_HIGH_3                                   : integer                                  := 0;
         PLL_C_CNT_LOW_3                                    : integer                                  := 0;
         PLL_C_CNT_PRST_3                                   : integer                                  := 0;
         PLL_C_CNT_PH_MUX_PRST_3                            : integer                                  := 0;
         PLL_C_CNT_BYPASS_EN_3                              : string                                   := "";
         PLL_C_CNT_EVEN_DUTY_EN_3                           : string                                   := "";
         PLL_C_CNT_FREQ_PS_STR_3                            : string                                   := "";
         PLL_C_CNT_PHASE_PS_STR_3                           : string                                   := "";
         PLL_C_CNT_DUTY_CYCLE_3                             : integer                                  := 0;
         PLL_C_CNT_OUT_EN_3                                 : string                                   := "";
         PLL_C_CNT_HIGH_4                                   : integer                                  := 0;
         PLL_C_CNT_LOW_4                                    : integer                                  := 0;
         PLL_C_CNT_PRST_4                                   : integer                                  := 0;
         PLL_C_CNT_PH_MUX_PRST_4                            : integer                                  := 0;
         PLL_C_CNT_BYPASS_EN_4                              : string                                   := "";
         PLL_C_CNT_EVEN_DUTY_EN_4                           : string                                   := "";
         PLL_C_CNT_FREQ_PS_STR_4                            : string                                   := "";
         PLL_C_CNT_PHASE_PS_STR_4                           : string                                   := "";
         PLL_C_CNT_DUTY_CYCLE_4                             : integer                                  := 0;
         PLL_C_CNT_OUT_EN_4                                 : string                                   := "";
         PLL_C_CNT_HIGH_5                                   : integer                                  := 0;
         PLL_C_CNT_LOW_5                                    : integer                                  := 0;
         PLL_C_CNT_PRST_5                                   : integer                                  := 0;
         PLL_C_CNT_PH_MUX_PRST_5                            : integer                                  := 0;
         PLL_C_CNT_BYPASS_EN_5                              : string                                   := "";
         PLL_C_CNT_EVEN_DUTY_EN_5                           : string                                   := "";
         PLL_C_CNT_FREQ_PS_STR_5                            : string                                   := "";
         PLL_C_CNT_PHASE_PS_STR_5                           : string                                   := "";
         PLL_C_CNT_DUTY_CYCLE_5                             : integer                                  := 0;
         PLL_C_CNT_OUT_EN_5                                 : string                                   := "";
         PLL_C_CNT_HIGH_6                                   : integer                                  := 0;
         PLL_C_CNT_LOW_6                                    : integer                                  := 0;
         PLL_C_CNT_PRST_6                                   : integer                                  := 0;
         PLL_C_CNT_PH_MUX_PRST_6                            : integer                                  := 0;
         PLL_C_CNT_BYPASS_EN_6                              : string                                   := "";
         PLL_C_CNT_EVEN_DUTY_EN_6                           : string                                   := "";
         PLL_C_CNT_FREQ_PS_STR_6                            : string                                   := "";
         PLL_C_CNT_PHASE_PS_STR_6                           : string                                   := "";
         PLL_C_CNT_DUTY_CYCLE_6                             : integer                                  := 0;
         PLL_C_CNT_OUT_EN_6                                 : string                                   := "";
         PLL_C_CNT_HIGH_7                                   : integer                                  := 0;
         PLL_C_CNT_LOW_7                                    : integer                                  := 0;
         PLL_C_CNT_PRST_7                                   : integer                                  := 0;
         PLL_C_CNT_PH_MUX_PRST_7                            : integer                                  := 0;
         PLL_C_CNT_BYPASS_EN_7                              : string                                   := "";
         PLL_C_CNT_EVEN_DUTY_EN_7                           : string                                   := "";
         PLL_C_CNT_FREQ_PS_STR_7                            : string                                   := "";
         PLL_C_CNT_PHASE_PS_STR_7                           : string                                   := "";
         PLL_C_CNT_DUTY_CYCLE_7                             : integer                                  := 0;
         PLL_C_CNT_OUT_EN_7                                 : string                                   := "";
         PLL_C_CNT_HIGH_8                                   : integer                                  := 0;
         PLL_C_CNT_LOW_8                                    : integer                                  := 0;
         PLL_C_CNT_PRST_8                                   : integer                                  := 0;
         PLL_C_CNT_PH_MUX_PRST_8                            : integer                                  := 0;
         PLL_C_CNT_BYPASS_EN_8                              : string                                   := "";
         PLL_C_CNT_EVEN_DUTY_EN_8                           : string                                   := "";
         PLL_C_CNT_FREQ_PS_STR_8                            : string                                   := "";
         PLL_C_CNT_PHASE_PS_STR_8                           : string                                   := "";
         PLL_C_CNT_DUTY_CYCLE_8                             : integer                                  := 0;
         PLL_C_CNT_OUT_EN_8                                 : string                                   := "";
         SEQ_SYNTH_PARAMS_HEX_FILENAME                      : string                                   := "";
         SEQ_SIM_PARAMS_HEX_FILENAME                        : string                                   := "";
         SEQ_CODE_HEX_FILENAME                              : string                                   := ""
      );
      port (
         global_reset_n                 : in    std_logic;
         pll_ref_clk                    : in    std_logic;
         pll_locked                     : out   std_logic;
         pll_extra_clk_0                : out   std_logic;
         pll_extra_clk_1                : out   std_logic;
         pll_extra_clk_2                : out   std_logic;
         pll_extra_clk_3                : out   std_logic;
         oct_rzqin                      : in    std_logic;
         mem_ck                         : out   std_logic_vector(0 downto 0);
         mem_ck_n                       : out   std_logic_vector(0 downto 0);
         mem_a                          : out   std_logic_vector(16 downto 0);
         mem_act_n                      : out   std_logic_vector(0 downto 0);
         mem_ba                         : out   std_logic_vector(1 downto 0);
         mem_bg                         : out   std_logic_vector(0 downto 0);
         mem_c                          : out   std_logic_vector(0 downto 0);
         mem_cke                        : out   std_logic_vector(0 downto 0);
         mem_cs_n                       : out   std_logic_vector(0 downto 0);
         mem_rm                         : out   std_logic_vector(0 downto 0);
         mem_odt                        : out   std_logic_vector(0 downto 0);
         mem_reset_n                    : out   std_logic_vector(0 downto 0);
         mem_par                        : out   std_logic_vector(0 downto 0);
         mem_alert_n                    : in    std_logic_vector(0 downto 0);
         mem_dqs                        : inout std_logic_vector(7 downto 0);
         mem_dqs_n                      : inout std_logic_vector(7 downto 0);
         mem_dq                         : inout std_logic_vector(63 downto 0);
         mem_dbi_n                      : inout std_logic_vector(7 downto 0);
         mem_ck_bidir                   : inout std_logic_vector(0 downto 0);
         mem_ck_bidir_n                 : inout std_logic_vector(0 downto 0);
         mem_dk                         : out   std_logic_vector(0 downto 0);
         mem_dk_n                       : out   std_logic_vector(0 downto 0);
         mem_dka                        : out   std_logic_vector(0 downto 0);
         mem_dka_n                      : out   std_logic_vector(0 downto 0);
         mem_dkb                        : out   std_logic_vector(0 downto 0);
         mem_dkb_n                      : out   std_logic_vector(0 downto 0);
         mem_k                          : out   std_logic_vector(0 downto 0);
         mem_k_n                        : out   std_logic_vector(0 downto 0);
         mem_req_n                      : in    std_logic_vector(0 downto 0);
         mem_gnt_n                      : out   std_logic_vector(0 downto 0);
         mem_err_n                      : in    std_logic_vector(0 downto 0);
         mem_ras_n                      : out   std_logic_vector(0 downto 0);
         mem_cas_n                      : out   std_logic_vector(0 downto 0);
         mem_we_n                       : out   std_logic_vector(0 downto 0);
         mem_ca                         : out   std_logic_vector(0 downto 0);
         mem_ref_n                      : out   std_logic_vector(0 downto 0);
         mem_wps_n                      : out   std_logic_vector(0 downto 0);
         mem_rps_n                      : out   std_logic_vector(0 downto 0);
         mem_doff_n                     : out   std_logic_vector(0 downto 0);
         mem_lda_n                      : out   std_logic_vector(0 downto 0);
         mem_ldb_n                      : out   std_logic_vector(0 downto 0);
         mem_rwa_n                      : out   std_logic_vector(0 downto 0);
         mem_rwb_n                      : out   std_logic_vector(0 downto 0);
         mem_lbk0_n                     : out   std_logic_vector(0 downto 0);
         mem_lbk1_n                     : out   std_logic_vector(0 downto 0);
         mem_cfg_n                      : out   std_logic_vector(0 downto 0);
         mem_ap                         : out   std_logic_vector(0 downto 0);
         mem_ainv                       : out   std_logic_vector(0 downto 0);
         mem_dm                         : out   std_logic_vector(0 downto 0);
         mem_bws_n                      : out   std_logic_vector(0 downto 0);
         mem_d                          : out   std_logic_vector(0 downto 0);
         mem_dqa                        : inout std_logic_vector(0 downto 0);
         mem_dqb                        : inout std_logic_vector(0 downto 0);
         mem_dinva                      : inout std_logic_vector(0 downto 0);
         mem_dinvb                      : inout std_logic_vector(0 downto 0);
         mem_q                          : in    std_logic_vector(0 downto 0);
         mem_qk                         : in    std_logic_vector(0 downto 0);
         mem_qk_n                       : in    std_logic_vector(0 downto 0);
         mem_qka                        : in    std_logic_vector(0 downto 0);
         mem_qka_n                      : in    std_logic_vector(0 downto 0);
         mem_qkb                        : in    std_logic_vector(0 downto 0);
         mem_qkb_n                      : in    std_logic_vector(0 downto 0);
         mem_cq                         : in    std_logic_vector(0 downto 0);
         mem_cq_n                       : in    std_logic_vector(0 downto 0);
         mem_pe_n                       : in    std_logic_vector(0 downto 0);
         local_cal_success              : out   std_logic;
         local_cal_fail                 : out   std_logic;
         vid_cal_done_persist           : in    std_logic;
         afi_reset_n                    : out   std_logic;
         afi_clk                        : out   std_logic;
         afi_half_clk                   : out   std_logic;
         emif_usr_reset_n               : out   std_logic;
         emif_usr_clk                   : out   std_logic;
         emif_usr_half_clk              : out   std_logic;
         emif_usr_reset_n_sec           : out   std_logic;
         emif_usr_clk_sec               : out   std_logic;
         emif_usr_half_clk_sec          : out   std_logic;
         cal_master_reset_n             : out   std_logic;
         cal_master_clk                 : out   std_logic;
         cal_slave_reset_n              : out   std_logic;
         cal_slave_clk                  : out   std_logic;
         cal_slave_reset_n_in           : in    std_logic;
         cal_slave_clk_in               : in    std_logic;
         cal_debug_reset_n              : in    std_logic;
         cal_debug_clk                  : in    std_logic;
         cal_debug_out_reset_n          : out   std_logic;
         cal_debug_out_clk              : out   std_logic;
         clks_sharing_master_out        : out   std_logic_vector(31 downto 0);
         clks_sharing_slave_in          : in    std_logic_vector(31 downto 0);
         clks_sharing_slave_out         : out   std_logic_vector(31 downto 0);
         afi_cal_success                : out   std_logic;
         afi_cal_fail                   : out   std_logic;
         afi_cal_req                    : in    std_logic;
         afi_rlat                       : out   std_logic_vector(5 downto 0);
         afi_wlat                       : out   std_logic_vector(5 downto 0);
         afi_seq_busy                   : out   std_logic_vector(3 downto 0);
         afi_ctl_refresh_done           : in    std_logic;
         afi_ctl_long_idle              : in    std_logic;
         afi_mps_req                    : in    std_logic;
         afi_mps_ack                    : out   std_logic;
         afi_addr                       : in    std_logic_vector(0 downto 0);
         afi_ba                         : in    std_logic_vector(0 downto 0);
         afi_bg                         : in    std_logic_vector(0 downto 0);
         afi_c                          : in    std_logic_vector(0 downto 0);
         afi_cke                        : in    std_logic_vector(0 downto 0);
         afi_cs_n                       : in    std_logic_vector(0 downto 0);
         afi_rm                         : in    std_logic_vector(0 downto 0);
         afi_odt                        : in    std_logic_vector(0 downto 0);
         afi_ras_n                      : in    std_logic_vector(0 downto 0);
         afi_cas_n                      : in    std_logic_vector(0 downto 0);
         afi_we_n                       : in    std_logic_vector(0 downto 0);
         afi_rst_n                      : in    std_logic_vector(0 downto 0);
         afi_act_n                      : in    std_logic_vector(0 downto 0);
         afi_req_n                      : out   std_logic_vector(0 downto 0);
         afi_gnt_n                      : in    std_logic_vector(0 downto 0);
         afi_err_n                      : out   std_logic_vector(0 downto 0);
         afi_par                        : in    std_logic_vector(0 downto 0);
         afi_ca                         : in    std_logic_vector(0 downto 0);
         afi_ref_n                      : in    std_logic_vector(0 downto 0);
         afi_wps_n                      : in    std_logic_vector(0 downto 0);
         afi_rps_n                      : in    std_logic_vector(0 downto 0);
         afi_doff_n                     : in    std_logic_vector(0 downto 0);
         afi_ld_n                       : in    std_logic_vector(0 downto 0);
         afi_rw_n                       : in    std_logic_vector(0 downto 0);
         afi_lbk0_n                     : in    std_logic_vector(0 downto 0);
         afi_lbk1_n                     : in    std_logic_vector(0 downto 0);
         afi_cfg_n                      : in    std_logic_vector(0 downto 0);
         afi_ap                         : in    std_logic_vector(0 downto 0);
         afi_ainv                       : in    std_logic_vector(0 downto 0);
         afi_dm                         : in    std_logic_vector(0 downto 0);
         afi_dm_n                       : in    std_logic_vector(0 downto 0);
         afi_bws_n                      : in    std_logic_vector(0 downto 0);
         afi_rdata_dbi_n                : out   std_logic_vector(0 downto 0);
         afi_wdata_dbi_n                : in    std_logic_vector(0 downto 0);
         afi_rdata_dinv                 : out   std_logic_vector(0 downto 0);
         afi_wdata_dinv                 : in    std_logic_vector(0 downto 0);
         afi_dqs_burst                  : in    std_logic_vector(0 downto 0);
         afi_wdata_valid                : in    std_logic_vector(0 downto 0);
         afi_wdata                      : in    std_logic_vector(0 downto 0);
         afi_rdata_en_full              : in    std_logic_vector(0 downto 0);
         afi_rdata                      : out   std_logic_vector(0 downto 0);
         afi_rdata_valid                : out   std_logic_vector(0 downto 0);
         afi_rrank                      : in    std_logic_vector(0 downto 0);
         afi_wrank                      : in    std_logic_vector(0 downto 0);
         afi_alert_n                    : out   std_logic_vector(0 downto 0);
         afi_pe_n                       : out   std_logic_vector(0 downto 0);
         ast_cmd_data_0                 : in    std_logic_vector(0 downto 0);
         ast_cmd_valid_0                : in    std_logic;
         ast_cmd_ready_0                : out   std_logic;
         ast_cmd_data_1                 : in    std_logic_vector(0 downto 0);
         ast_cmd_valid_1                : in    std_logic;
         ast_cmd_ready_1                : out   std_logic;
         ast_wr_data_0                  : in    std_logic_vector(0 downto 0);
         ast_wr_valid_0                 : in    std_logic;
         ast_wr_ready_0                 : out   std_logic;
         ast_wr_data_1                  : in    std_logic_vector(0 downto 0);
         ast_wr_valid_1                 : in    std_logic;
         ast_wr_ready_1                 : out   std_logic;
         ast_rd_data_0                  : out   std_logic_vector(0 downto 0);
         ast_rd_valid_0                 : out   std_logic;
         ast_rd_ready_0                 : in    std_logic;
         ast_rd_data_1                  : out   std_logic_vector(0 downto 0);
         ast_rd_valid_1                 : out   std_logic;
         ast_rd_ready_1                 : in    std_logic;
         amm_ready_0                    : out   std_logic;
         amm_read_0                     : in    std_logic;
         amm_write_0                    : in    std_logic;
         amm_address_0                  : in    std_logic_vector(24 downto 0);
         amm_readdata_0                 : out   std_logic_vector(511 downto 0);
         amm_writedata_0                : in    std_logic_vector(511 downto 0);
         amm_burstcount_0               : in    std_logic_vector(6 downto 0);
         amm_byteenable_0               : in    std_logic_vector(63 downto 0);
         amm_beginbursttransfer_0       : in    std_logic;
         amm_readdatavalid_0            : out   std_logic;
         amm_ready_1                    : out   std_logic;
         amm_read_1                     : in    std_logic;
         amm_write_1                    : in    std_logic;
         amm_address_1                  : in    std_logic_vector(24 downto 0);
         amm_readdata_1                 : out   std_logic_vector(511 downto 0);
         amm_writedata_1                : in    std_logic_vector(511 downto 0);
         amm_burstcount_1               : in    std_logic_vector(6 downto 0);
         amm_byteenable_1               : in    std_logic_vector(63 downto 0);
         amm_beginbursttransfer_1       : in    std_logic;
         amm_readdatavalid_1            : out   std_logic;
         ctrl_user_priority_hi_0        : in    std_logic;
         ctrl_user_priority_hi_1        : in    std_logic;
         ctrl_auto_precharge_req_0      : in    std_logic;
         ctrl_auto_precharge_req_1      : in    std_logic;
         ctrl_user_refresh_req          : in    std_logic_vector(3 downto 0);
         ctrl_user_refresh_bank         : in    std_logic_vector(15 downto 0);
         ctrl_user_refresh_ack          : out   std_logic;
         ctrl_self_refresh_req          : in    std_logic_vector(3 downto 0);
         ctrl_self_refresh_ack          : out   std_logic;
         ctrl_will_refresh              : out   std_logic;
         ctrl_deep_power_down_req       : in    std_logic;
         ctrl_deep_power_down_ack       : out   std_logic;
         ctrl_power_down_ack            : out   std_logic;
         ctrl_zq_cal_long_req           : in    std_logic;
         ctrl_zq_cal_short_req          : in    std_logic;
         ctrl_zq_cal_ack                : out   std_logic;
         ctrl_ecc_write_info_0          : in    std_logic_vector(14 downto 0);
         ctrl_ecc_rdata_id_0            : out   std_logic_vector(12 downto 0);
         ctrl_ecc_read_info_0           : out   std_logic_vector(2 downto 0);
         ctrl_ecc_cmd_info_0            : out   std_logic_vector(2 downto 0);
         ctrl_ecc_idle_0                : out   std_logic;
         ctrl_ecc_wr_pointer_info_0     : out   std_logic_vector(11 downto 0);
         ctrl_ecc_write_info_1          : in    std_logic_vector(14 downto 0);
         ctrl_ecc_rdata_id_1            : out   std_logic_vector(12 downto 0);
         ctrl_ecc_read_info_1           : out   std_logic_vector(2 downto 0);
         ctrl_ecc_cmd_info_1            : out   std_logic_vector(2 downto 0);
         ctrl_ecc_idle_1                : out   std_logic;
         ctrl_ecc_wr_pointer_info_1     : out   std_logic_vector(11 downto 0);
         mmr_slave_waitrequest_0        : out   std_logic;
         mmr_slave_read_0               : in    std_logic;
         mmr_slave_write_0              : in    std_logic;
         mmr_slave_address_0            : in    std_logic_vector(9 downto 0);
         mmr_slave_readdata_0           : out   std_logic_vector(31 downto 0);
         mmr_slave_writedata_0          : in    std_logic_vector(31 downto 0);
         mmr_slave_burstcount_0         : in    std_logic_vector(1 downto 0);
         mmr_slave_beginbursttransfer_0 : in    std_logic;
         mmr_slave_readdatavalid_0      : out   std_logic;
         mmr_slave_waitrequest_1        : out   std_logic;
         mmr_slave_read_1               : in    std_logic;
         mmr_slave_write_1              : in    std_logic;
         mmr_slave_address_1            : in    std_logic_vector(9 downto 0);
         mmr_slave_readdata_1           : out   std_logic_vector(31 downto 0);
         mmr_slave_writedata_1          : in    std_logic_vector(31 downto 0);
         mmr_slave_burstcount_1         : in    std_logic_vector(1 downto 0);
         mmr_slave_beginbursttransfer_1 : in    std_logic;
         mmr_slave_readdatavalid_1      : out   std_logic;
         hps_to_emif                    : in    std_logic_vector(4095 downto 0);
         emif_to_hps                    : out   std_logic_vector(4095 downto 0);
         hps_to_emif_gp                 : in    std_logic_vector(1 downto 0);
         emif_to_hps_gp                 : out   std_logic_vector(0 downto 0);
         cal_debug_waitrequest          : out   std_logic;
         cal_debug_read                 : in    std_logic;
         cal_debug_write                : in    std_logic;
         cal_debug_addr                 : in    std_logic_vector(23 downto 0);
         cal_debug_read_data            : out   std_logic_vector(31 downto 0);
         cal_debug_write_data           : in    std_logic_vector(31 downto 0);
         cal_debug_byteenable           : in    std_logic_vector(3 downto 0);
         cal_debug_read_data_valid      : out   std_logic;
         cal_debug_out_waitrequest      : in    std_logic;
         cal_debug_out_read             : out   std_logic;
         cal_debug_out_write            : out   std_logic;
         cal_debug_out_addr             : out   std_logic_vector(23 downto 0);
         cal_debug_out_read_data        : in    std_logic_vector(31 downto 0);
         cal_debug_out_write_data       : out   std_logic_vector(31 downto 0);
         cal_debug_out_byteenable       : out   std_logic_vector(3 downto 0);
         cal_debug_out_read_data_valid  : in    std_logic;
         cal_master_waitrequest         : in    std_logic;
         cal_master_read                : out   std_logic;
         cal_master_write               : out   std_logic;
         cal_master_addr                : out   std_logic_vector(15 downto 0);
         cal_master_read_data           : in    std_logic_vector(31 downto 0);
         cal_master_write_data          : out   std_logic_vector(31 downto 0);
         cal_master_byteenable          : out   std_logic_vector(3 downto 0);
         cal_master_read_data_valid     : in    std_logic;
         cal_master_burstcount          : out   std_logic;
         cal_master_debugaccess         : out   std_logic;
         ioaux_pio_in                   : in    std_logic_vector(7 downto 0);
         ioaux_pio_out                  : out   std_logic_vector(7 downto 0);
         pa_dprio_clk                   : in    std_logic;
         pa_dprio_read                  : in    std_logic;
         pa_dprio_reg_addr              : in    std_logic_vector(8 downto 0);
         pa_dprio_rst_n                 : in    std_logic;
         pa_dprio_write                 : in    std_logic;
         pa_dprio_writedata             : in    std_logic_vector(7 downto 0);
         pa_dprio_block_select          : out   std_logic;
         pa_dprio_readdata              : out   std_logic_vector(7 downto 0);
         pll_phase_en                   : in    std_logic;
         pll_up_dn                      : in    std_logic;
         pll_cnt_sel                    : in    std_logic_vector(3 downto 0);
         pll_num_phase_shifts           : in    std_logic_vector(2 downto 0);
         pll_phase_done                 : out   std_logic;
         pll_core_refclk                : in    std_logic_vector(0 downto 0);
         dft_core_clk_buf_out           : out   std_logic_vector(1 downto 0);
         dft_core_clk_locked            : out   std_logic_vector(1 downto 0)
      );
   end component SDRAM_altera_emif_arch_nf_191_65stwui_top;

begin
   arch_inst : component SDRAM_altera_emif_arch_nf_191_65stwui_top
      generic map (
         PROTOCOL_ENUM => PROTOCOL_ENUM,
         PHY_TARGET_IS_ES => PHY_TARGET_IS_ES,
         PHY_TARGET_IS_ES2 => PHY_TARGET_IS_ES2,
         PHY_TARGET_IS_PRODUCTION => PHY_TARGET_IS_PRODUCTION,
         PHY_CONFIG_ENUM => PHY_CONFIG_ENUM,
         PHY_PING_PONG_EN => PHY_PING_PONG_EN,
         PHY_DLL_CORE_UPDN_EN => PHY_DLL_CORE_UPDN_EN,
         PHY_CORE_CLKS_SHARING_ENUM => PHY_CORE_CLKS_SHARING_ENUM,
         PHY_CALIBRATED_OCT => PHY_CALIBRATED_OCT,
         PHY_AC_CALIBRATED_OCT => PHY_AC_CALIBRATED_OCT,
         PHY_CK_CALIBRATED_OCT => PHY_CK_CALIBRATED_OCT,
         PHY_DATA_CALIBRATED_OCT => PHY_DATA_CALIBRATED_OCT,
         PHY_HPS_ENABLE_EARLY_RELEASE => PHY_HPS_ENABLE_EARLY_RELEASE,
         PLL_NUM_OF_EXTRA_CLKS => PLL_NUM_OF_EXTRA_CLKS,
         MEM_FORMAT_ENUM => MEM_FORMAT_ENUM,
         MEM_BURST_LENGTH => MEM_BURST_LENGTH,
         MEM_DATA_MASK_EN => MEM_DATA_MASK_EN,
         MEM_TTL_DATA_WIDTH => MEM_TTL_DATA_WIDTH,
         MEM_TTL_NUM_OF_READ_GROUPS => MEM_TTL_NUM_OF_READ_GROUPS,
         MEM_TTL_NUM_OF_WRITE_GROUPS => MEM_TTL_NUM_OF_WRITE_GROUPS,
         DIAG_SIM_REGTEST_MODE => DIAG_SIM_REGTEST_MODE,
         DIAG_SYNTH_FOR_SIM => DIAG_SYNTH_FOR_SIM,
         DIAG_ECLIPSE_DEBUG => DIAG_ECLIPSE_DEBUG,
         DIAG_EXPORT_VJI => DIAG_EXPORT_VJI,
         DIAG_INTERFACE_ID => DIAG_INTERFACE_ID,
         DIAG_USE_ABSTRACT_PHY => DIAG_USE_ABSTRACT_PHY,
         DIAG_SIM_VERBOSE_LEVEL => DIAG_SIM_VERBOSE_LEVEL,
         DIAG_FAST_SIM => DIAG_FAST_SIM,
         SILICON_REV => SILICON_REV,
         IS_HPS => IS_HPS,
         IS_VID => IS_VID,
         USER_CLK_RATIO => USER_CLK_RATIO,
         C2P_P2C_CLK_RATIO => C2P_P2C_CLK_RATIO,
         PHY_HMC_CLK_RATIO => PHY_HMC_CLK_RATIO,
         DIAG_ABSTRACT_PHY_WLAT => DIAG_ABSTRACT_PHY_WLAT,
         DIAG_ABSTRACT_PHY_RLAT => DIAG_ABSTRACT_PHY_RLAT,
         DIAG_CPA_OUT_1_EN => DIAG_CPA_OUT_1_EN,
         DIAG_USE_CPA_LOCK => DIAG_USE_CPA_LOCK,
         DQS_BUS_MODE_ENUM => DQS_BUS_MODE_ENUM,
         AC_PIN_MAP_SCHEME => AC_PIN_MAP_SCHEME,
         NUM_OF_HMC_PORTS => NUM_OF_HMC_PORTS,
         HMC_AVL_PROTOCOL_ENUM => HMC_AVL_PROTOCOL_ENUM,
         HMC_CTRL_DIMM_TYPE => HMC_CTRL_DIMM_TYPE,
         REGISTER_AFI => REGISTER_AFI,
         SEQ_SYNTH_CPU_CLK_DIVIDE => SEQ_SYNTH_CPU_CLK_DIVIDE,
         SEQ_SYNTH_CAL_CLK_DIVIDE => SEQ_SYNTH_CAL_CLK_DIVIDE,
         SEQ_SIM_CPU_CLK_DIVIDE => SEQ_SIM_CPU_CLK_DIVIDE,
         SEQ_SIM_CAL_CLK_DIVIDE => SEQ_SIM_CAL_CLK_DIVIDE,
         SEQ_SYNTH_OSC_FREQ_MHZ => SEQ_SYNTH_OSC_FREQ_MHZ,
         SEQ_SIM_OSC_FREQ_MHZ => SEQ_SIM_OSC_FREQ_MHZ,
         NUM_OF_RTL_TILES => NUM_OF_RTL_TILES,
         PRI_RDATA_TILE_INDEX => PRI_RDATA_TILE_INDEX,
         PRI_RDATA_LANE_INDEX => PRI_RDATA_LANE_INDEX,
         PRI_WDATA_TILE_INDEX => PRI_WDATA_TILE_INDEX,
         PRI_WDATA_LANE_INDEX => PRI_WDATA_LANE_INDEX,
         PRI_AC_TILE_INDEX => PRI_AC_TILE_INDEX,
         SEC_RDATA_TILE_INDEX => SEC_RDATA_TILE_INDEX,
         SEC_RDATA_LANE_INDEX => SEC_RDATA_LANE_INDEX,
         SEC_WDATA_TILE_INDEX => SEC_WDATA_TILE_INDEX,
         SEC_WDATA_LANE_INDEX => SEC_WDATA_LANE_INDEX,
         SEC_AC_TILE_INDEX => SEC_AC_TILE_INDEX,
         LANES_USAGE_0 => LANES_USAGE_0,
         LANES_USAGE_1 => LANES_USAGE_1,
         LANES_USAGE_2 => LANES_USAGE_2,
         LANES_USAGE_3 => LANES_USAGE_3,
         LANES_USAGE_AUTOGEN_WCNT => LANES_USAGE_AUTOGEN_WCNT,
         PINS_USAGE_0 => PINS_USAGE_0,
         PINS_USAGE_1 => PINS_USAGE_1,
         PINS_USAGE_2 => PINS_USAGE_2,
         PINS_USAGE_3 => PINS_USAGE_3,
         PINS_USAGE_4 => PINS_USAGE_4,
         PINS_USAGE_5 => PINS_USAGE_5,
         PINS_USAGE_6 => PINS_USAGE_6,
         PINS_USAGE_7 => PINS_USAGE_7,
         PINS_USAGE_8 => PINS_USAGE_8,
         PINS_USAGE_9 => PINS_USAGE_9,
         PINS_USAGE_10 => PINS_USAGE_10,
         PINS_USAGE_11 => PINS_USAGE_11,
         PINS_USAGE_12 => PINS_USAGE_12,
         PINS_USAGE_AUTOGEN_WCNT => PINS_USAGE_AUTOGEN_WCNT,
         PINS_RATE_0 => PINS_RATE_0,
         PINS_RATE_1 => PINS_RATE_1,
         PINS_RATE_2 => PINS_RATE_2,
         PINS_RATE_3 => PINS_RATE_3,
         PINS_RATE_4 => PINS_RATE_4,
         PINS_RATE_5 => PINS_RATE_5,
         PINS_RATE_6 => PINS_RATE_6,
         PINS_RATE_7 => PINS_RATE_7,
         PINS_RATE_8 => PINS_RATE_8,
         PINS_RATE_9 => PINS_RATE_9,
         PINS_RATE_10 => PINS_RATE_10,
         PINS_RATE_11 => PINS_RATE_11,
         PINS_RATE_12 => PINS_RATE_12,
         PINS_RATE_AUTOGEN_WCNT => PINS_RATE_AUTOGEN_WCNT,
         PINS_WDB_0 => PINS_WDB_0,
         PINS_WDB_1 => PINS_WDB_1,
         PINS_WDB_2 => PINS_WDB_2,
         PINS_WDB_3 => PINS_WDB_3,
         PINS_WDB_4 => PINS_WDB_4,
         PINS_WDB_5 => PINS_WDB_5,
         PINS_WDB_6 => PINS_WDB_6,
         PINS_WDB_7 => PINS_WDB_7,
         PINS_WDB_8 => PINS_WDB_8,
         PINS_WDB_9 => PINS_WDB_9,
         PINS_WDB_10 => PINS_WDB_10,
         PINS_WDB_11 => PINS_WDB_11,
         PINS_WDB_12 => PINS_WDB_12,
         PINS_WDB_13 => PINS_WDB_13,
         PINS_WDB_14 => PINS_WDB_14,
         PINS_WDB_15 => PINS_WDB_15,
         PINS_WDB_16 => PINS_WDB_16,
         PINS_WDB_17 => PINS_WDB_17,
         PINS_WDB_18 => PINS_WDB_18,
         PINS_WDB_19 => PINS_WDB_19,
         PINS_WDB_20 => PINS_WDB_20,
         PINS_WDB_21 => PINS_WDB_21,
         PINS_WDB_22 => PINS_WDB_22,
         PINS_WDB_23 => PINS_WDB_23,
         PINS_WDB_24 => PINS_WDB_24,
         PINS_WDB_25 => PINS_WDB_25,
         PINS_WDB_26 => PINS_WDB_26,
         PINS_WDB_27 => PINS_WDB_27,
         PINS_WDB_28 => PINS_WDB_28,
         PINS_WDB_29 => PINS_WDB_29,
         PINS_WDB_30 => PINS_WDB_30,
         PINS_WDB_31 => PINS_WDB_31,
         PINS_WDB_32 => PINS_WDB_32,
         PINS_WDB_33 => PINS_WDB_33,
         PINS_WDB_34 => PINS_WDB_34,
         PINS_WDB_35 => PINS_WDB_35,
         PINS_WDB_36 => PINS_WDB_36,
         PINS_WDB_37 => PINS_WDB_37,
         PINS_WDB_38 => PINS_WDB_38,
         PINS_WDB_AUTOGEN_WCNT => PINS_WDB_AUTOGEN_WCNT,
         PINS_DATA_IN_MODE_0 => PINS_DATA_IN_MODE_0,
         PINS_DATA_IN_MODE_1 => PINS_DATA_IN_MODE_1,
         PINS_DATA_IN_MODE_2 => PINS_DATA_IN_MODE_2,
         PINS_DATA_IN_MODE_3 => PINS_DATA_IN_MODE_3,
         PINS_DATA_IN_MODE_4 => PINS_DATA_IN_MODE_4,
         PINS_DATA_IN_MODE_5 => PINS_DATA_IN_MODE_5,
         PINS_DATA_IN_MODE_6 => PINS_DATA_IN_MODE_6,
         PINS_DATA_IN_MODE_7 => PINS_DATA_IN_MODE_7,
         PINS_DATA_IN_MODE_8 => PINS_DATA_IN_MODE_8,
         PINS_DATA_IN_MODE_9 => PINS_DATA_IN_MODE_9,
         PINS_DATA_IN_MODE_10 => PINS_DATA_IN_MODE_10,
         PINS_DATA_IN_MODE_11 => PINS_DATA_IN_MODE_11,
         PINS_DATA_IN_MODE_12 => PINS_DATA_IN_MODE_12,
         PINS_DATA_IN_MODE_13 => PINS_DATA_IN_MODE_13,
         PINS_DATA_IN_MODE_14 => PINS_DATA_IN_MODE_14,
         PINS_DATA_IN_MODE_15 => PINS_DATA_IN_MODE_15,
         PINS_DATA_IN_MODE_16 => PINS_DATA_IN_MODE_16,
         PINS_DATA_IN_MODE_17 => PINS_DATA_IN_MODE_17,
         PINS_DATA_IN_MODE_18 => PINS_DATA_IN_MODE_18,
         PINS_DATA_IN_MODE_19 => PINS_DATA_IN_MODE_19,
         PINS_DATA_IN_MODE_20 => PINS_DATA_IN_MODE_20,
         PINS_DATA_IN_MODE_21 => PINS_DATA_IN_MODE_21,
         PINS_DATA_IN_MODE_22 => PINS_DATA_IN_MODE_22,
         PINS_DATA_IN_MODE_23 => PINS_DATA_IN_MODE_23,
         PINS_DATA_IN_MODE_24 => PINS_DATA_IN_MODE_24,
         PINS_DATA_IN_MODE_25 => PINS_DATA_IN_MODE_25,
         PINS_DATA_IN_MODE_26 => PINS_DATA_IN_MODE_26,
         PINS_DATA_IN_MODE_27 => PINS_DATA_IN_MODE_27,
         PINS_DATA_IN_MODE_28 => PINS_DATA_IN_MODE_28,
         PINS_DATA_IN_MODE_29 => PINS_DATA_IN_MODE_29,
         PINS_DATA_IN_MODE_30 => PINS_DATA_IN_MODE_30,
         PINS_DATA_IN_MODE_31 => PINS_DATA_IN_MODE_31,
         PINS_DATA_IN_MODE_32 => PINS_DATA_IN_MODE_32,
         PINS_DATA_IN_MODE_33 => PINS_DATA_IN_MODE_33,
         PINS_DATA_IN_MODE_34 => PINS_DATA_IN_MODE_34,
         PINS_DATA_IN_MODE_35 => PINS_DATA_IN_MODE_35,
         PINS_DATA_IN_MODE_36 => PINS_DATA_IN_MODE_36,
         PINS_DATA_IN_MODE_37 => PINS_DATA_IN_MODE_37,
         PINS_DATA_IN_MODE_38 => PINS_DATA_IN_MODE_38,
         PINS_DATA_IN_MODE_AUTOGEN_WCNT => PINS_DATA_IN_MODE_AUTOGEN_WCNT,
         PINS_C2L_DRIVEN_0 => PINS_C2L_DRIVEN_0,
         PINS_C2L_DRIVEN_1 => PINS_C2L_DRIVEN_1,
         PINS_C2L_DRIVEN_2 => PINS_C2L_DRIVEN_2,
         PINS_C2L_DRIVEN_3 => PINS_C2L_DRIVEN_3,
         PINS_C2L_DRIVEN_4 => PINS_C2L_DRIVEN_4,
         PINS_C2L_DRIVEN_5 => PINS_C2L_DRIVEN_5,
         PINS_C2L_DRIVEN_6 => PINS_C2L_DRIVEN_6,
         PINS_C2L_DRIVEN_7 => PINS_C2L_DRIVEN_7,
         PINS_C2L_DRIVEN_8 => PINS_C2L_DRIVEN_8,
         PINS_C2L_DRIVEN_9 => PINS_C2L_DRIVEN_9,
         PINS_C2L_DRIVEN_10 => PINS_C2L_DRIVEN_10,
         PINS_C2L_DRIVEN_11 => PINS_C2L_DRIVEN_11,
         PINS_C2L_DRIVEN_12 => PINS_C2L_DRIVEN_12,
         PINS_C2L_DRIVEN_AUTOGEN_WCNT => PINS_C2L_DRIVEN_AUTOGEN_WCNT,
         PINS_DB_IN_BYPASS_0 => PINS_DB_IN_BYPASS_0,
         PINS_DB_IN_BYPASS_1 => PINS_DB_IN_BYPASS_1,
         PINS_DB_IN_BYPASS_2 => PINS_DB_IN_BYPASS_2,
         PINS_DB_IN_BYPASS_3 => PINS_DB_IN_BYPASS_3,
         PINS_DB_IN_BYPASS_4 => PINS_DB_IN_BYPASS_4,
         PINS_DB_IN_BYPASS_5 => PINS_DB_IN_BYPASS_5,
         PINS_DB_IN_BYPASS_6 => PINS_DB_IN_BYPASS_6,
         PINS_DB_IN_BYPASS_7 => PINS_DB_IN_BYPASS_7,
         PINS_DB_IN_BYPASS_8 => PINS_DB_IN_BYPASS_8,
         PINS_DB_IN_BYPASS_9 => PINS_DB_IN_BYPASS_9,
         PINS_DB_IN_BYPASS_10 => PINS_DB_IN_BYPASS_10,
         PINS_DB_IN_BYPASS_11 => PINS_DB_IN_BYPASS_11,
         PINS_DB_IN_BYPASS_12 => PINS_DB_IN_BYPASS_12,
         PINS_DB_IN_BYPASS_AUTOGEN_WCNT => PINS_DB_IN_BYPASS_AUTOGEN_WCNT,
         PINS_DB_OUT_BYPASS_0 => PINS_DB_OUT_BYPASS_0,
         PINS_DB_OUT_BYPASS_1 => PINS_DB_OUT_BYPASS_1,
         PINS_DB_OUT_BYPASS_2 => PINS_DB_OUT_BYPASS_2,
         PINS_DB_OUT_BYPASS_3 => PINS_DB_OUT_BYPASS_3,
         PINS_DB_OUT_BYPASS_4 => PINS_DB_OUT_BYPASS_4,
         PINS_DB_OUT_BYPASS_5 => PINS_DB_OUT_BYPASS_5,
         PINS_DB_OUT_BYPASS_6 => PINS_DB_OUT_BYPASS_6,
         PINS_DB_OUT_BYPASS_7 => PINS_DB_OUT_BYPASS_7,
         PINS_DB_OUT_BYPASS_8 => PINS_DB_OUT_BYPASS_8,
         PINS_DB_OUT_BYPASS_9 => PINS_DB_OUT_BYPASS_9,
         PINS_DB_OUT_BYPASS_10 => PINS_DB_OUT_BYPASS_10,
         PINS_DB_OUT_BYPASS_11 => PINS_DB_OUT_BYPASS_11,
         PINS_DB_OUT_BYPASS_12 => PINS_DB_OUT_BYPASS_12,
         PINS_DB_OUT_BYPASS_AUTOGEN_WCNT => PINS_DB_OUT_BYPASS_AUTOGEN_WCNT,
         PINS_DB_OE_BYPASS_0 => PINS_DB_OE_BYPASS_0,
         PINS_DB_OE_BYPASS_1 => PINS_DB_OE_BYPASS_1,
         PINS_DB_OE_BYPASS_2 => PINS_DB_OE_BYPASS_2,
         PINS_DB_OE_BYPASS_3 => PINS_DB_OE_BYPASS_3,
         PINS_DB_OE_BYPASS_4 => PINS_DB_OE_BYPASS_4,
         PINS_DB_OE_BYPASS_5 => PINS_DB_OE_BYPASS_5,
         PINS_DB_OE_BYPASS_6 => PINS_DB_OE_BYPASS_6,
         PINS_DB_OE_BYPASS_7 => PINS_DB_OE_BYPASS_7,
         PINS_DB_OE_BYPASS_8 => PINS_DB_OE_BYPASS_8,
         PINS_DB_OE_BYPASS_9 => PINS_DB_OE_BYPASS_9,
         PINS_DB_OE_BYPASS_10 => PINS_DB_OE_BYPASS_10,
         PINS_DB_OE_BYPASS_11 => PINS_DB_OE_BYPASS_11,
         PINS_DB_OE_BYPASS_12 => PINS_DB_OE_BYPASS_12,
         PINS_DB_OE_BYPASS_AUTOGEN_WCNT => PINS_DB_OE_BYPASS_AUTOGEN_WCNT,
         PINS_INVERT_WR_0 => PINS_INVERT_WR_0,
         PINS_INVERT_WR_1 => PINS_INVERT_WR_1,
         PINS_INVERT_WR_2 => PINS_INVERT_WR_2,
         PINS_INVERT_WR_3 => PINS_INVERT_WR_3,
         PINS_INVERT_WR_4 => PINS_INVERT_WR_4,
         PINS_INVERT_WR_5 => PINS_INVERT_WR_5,
         PINS_INVERT_WR_6 => PINS_INVERT_WR_6,
         PINS_INVERT_WR_7 => PINS_INVERT_WR_7,
         PINS_INVERT_WR_8 => PINS_INVERT_WR_8,
         PINS_INVERT_WR_9 => PINS_INVERT_WR_9,
         PINS_INVERT_WR_10 => PINS_INVERT_WR_10,
         PINS_INVERT_WR_11 => PINS_INVERT_WR_11,
         PINS_INVERT_WR_12 => PINS_INVERT_WR_12,
         PINS_INVERT_WR_AUTOGEN_WCNT => PINS_INVERT_WR_AUTOGEN_WCNT,
         PINS_INVERT_OE_0 => PINS_INVERT_OE_0,
         PINS_INVERT_OE_1 => PINS_INVERT_OE_1,
         PINS_INVERT_OE_2 => PINS_INVERT_OE_2,
         PINS_INVERT_OE_3 => PINS_INVERT_OE_3,
         PINS_INVERT_OE_4 => PINS_INVERT_OE_4,
         PINS_INVERT_OE_5 => PINS_INVERT_OE_5,
         PINS_INVERT_OE_6 => PINS_INVERT_OE_6,
         PINS_INVERT_OE_7 => PINS_INVERT_OE_7,
         PINS_INVERT_OE_8 => PINS_INVERT_OE_8,
         PINS_INVERT_OE_9 => PINS_INVERT_OE_9,
         PINS_INVERT_OE_10 => PINS_INVERT_OE_10,
         PINS_INVERT_OE_11 => PINS_INVERT_OE_11,
         PINS_INVERT_OE_12 => PINS_INVERT_OE_12,
         PINS_INVERT_OE_AUTOGEN_WCNT => PINS_INVERT_OE_AUTOGEN_WCNT,
         PINS_AC_HMC_DATA_OVERRIDE_ENA_0 => PINS_AC_HMC_DATA_OVERRIDE_ENA_0,
         PINS_AC_HMC_DATA_OVERRIDE_ENA_1 => PINS_AC_HMC_DATA_OVERRIDE_ENA_1,
         PINS_AC_HMC_DATA_OVERRIDE_ENA_2 => PINS_AC_HMC_DATA_OVERRIDE_ENA_2,
         PINS_AC_HMC_DATA_OVERRIDE_ENA_3 => PINS_AC_HMC_DATA_OVERRIDE_ENA_3,
         PINS_AC_HMC_DATA_OVERRIDE_ENA_4 => PINS_AC_HMC_DATA_OVERRIDE_ENA_4,
         PINS_AC_HMC_DATA_OVERRIDE_ENA_5 => PINS_AC_HMC_DATA_OVERRIDE_ENA_5,
         PINS_AC_HMC_DATA_OVERRIDE_ENA_6 => PINS_AC_HMC_DATA_OVERRIDE_ENA_6,
         PINS_AC_HMC_DATA_OVERRIDE_ENA_7 => PINS_AC_HMC_DATA_OVERRIDE_ENA_7,
         PINS_AC_HMC_DATA_OVERRIDE_ENA_8 => PINS_AC_HMC_DATA_OVERRIDE_ENA_8,
         PINS_AC_HMC_DATA_OVERRIDE_ENA_9 => PINS_AC_HMC_DATA_OVERRIDE_ENA_9,
         PINS_AC_HMC_DATA_OVERRIDE_ENA_10 => PINS_AC_HMC_DATA_OVERRIDE_ENA_10,
         PINS_AC_HMC_DATA_OVERRIDE_ENA_11 => PINS_AC_HMC_DATA_OVERRIDE_ENA_11,
         PINS_AC_HMC_DATA_OVERRIDE_ENA_12 => PINS_AC_HMC_DATA_OVERRIDE_ENA_12,
         PINS_AC_HMC_DATA_OVERRIDE_ENA_AUTOGEN_WCNT => PINS_AC_HMC_DATA_OVERRIDE_ENA_AUTOGEN_WCNT,
         PINS_OCT_MODE_0 => PINS_OCT_MODE_0,
         PINS_OCT_MODE_1 => PINS_OCT_MODE_1,
         PINS_OCT_MODE_2 => PINS_OCT_MODE_2,
         PINS_OCT_MODE_3 => PINS_OCT_MODE_3,
         PINS_OCT_MODE_4 => PINS_OCT_MODE_4,
         PINS_OCT_MODE_5 => PINS_OCT_MODE_5,
         PINS_OCT_MODE_6 => PINS_OCT_MODE_6,
         PINS_OCT_MODE_7 => PINS_OCT_MODE_7,
         PINS_OCT_MODE_8 => PINS_OCT_MODE_8,
         PINS_OCT_MODE_9 => PINS_OCT_MODE_9,
         PINS_OCT_MODE_10 => PINS_OCT_MODE_10,
         PINS_OCT_MODE_11 => PINS_OCT_MODE_11,
         PINS_OCT_MODE_12 => PINS_OCT_MODE_12,
         PINS_OCT_MODE_AUTOGEN_WCNT => PINS_OCT_MODE_AUTOGEN_WCNT,
         PINS_GPIO_MODE_0 => PINS_GPIO_MODE_0,
         PINS_GPIO_MODE_1 => PINS_GPIO_MODE_1,
         PINS_GPIO_MODE_2 => PINS_GPIO_MODE_2,
         PINS_GPIO_MODE_3 => PINS_GPIO_MODE_3,
         PINS_GPIO_MODE_4 => PINS_GPIO_MODE_4,
         PINS_GPIO_MODE_5 => PINS_GPIO_MODE_5,
         PINS_GPIO_MODE_6 => PINS_GPIO_MODE_6,
         PINS_GPIO_MODE_7 => PINS_GPIO_MODE_7,
         PINS_GPIO_MODE_8 => PINS_GPIO_MODE_8,
         PINS_GPIO_MODE_9 => PINS_GPIO_MODE_9,
         PINS_GPIO_MODE_10 => PINS_GPIO_MODE_10,
         PINS_GPIO_MODE_11 => PINS_GPIO_MODE_11,
         PINS_GPIO_MODE_12 => PINS_GPIO_MODE_12,
         PINS_GPIO_MODE_AUTOGEN_WCNT => PINS_GPIO_MODE_AUTOGEN_WCNT,
         UNUSED_MEM_PINS_PINLOC_0 => UNUSED_MEM_PINS_PINLOC_0,
         UNUSED_MEM_PINS_PINLOC_1 => UNUSED_MEM_PINS_PINLOC_1,
         UNUSED_MEM_PINS_PINLOC_2 => UNUSED_MEM_PINS_PINLOC_2,
         UNUSED_MEM_PINS_PINLOC_3 => UNUSED_MEM_PINS_PINLOC_3,
         UNUSED_MEM_PINS_PINLOC_4 => UNUSED_MEM_PINS_PINLOC_4,
         UNUSED_MEM_PINS_PINLOC_5 => UNUSED_MEM_PINS_PINLOC_5,
         UNUSED_MEM_PINS_PINLOC_6 => UNUSED_MEM_PINS_PINLOC_6,
         UNUSED_MEM_PINS_PINLOC_7 => UNUSED_MEM_PINS_PINLOC_7,
         UNUSED_MEM_PINS_PINLOC_8 => UNUSED_MEM_PINS_PINLOC_8,
         UNUSED_MEM_PINS_PINLOC_9 => UNUSED_MEM_PINS_PINLOC_9,
         UNUSED_MEM_PINS_PINLOC_10 => UNUSED_MEM_PINS_PINLOC_10,
         UNUSED_MEM_PINS_PINLOC_11 => UNUSED_MEM_PINS_PINLOC_11,
         UNUSED_MEM_PINS_PINLOC_12 => UNUSED_MEM_PINS_PINLOC_12,
         UNUSED_MEM_PINS_PINLOC_13 => UNUSED_MEM_PINS_PINLOC_13,
         UNUSED_MEM_PINS_PINLOC_14 => UNUSED_MEM_PINS_PINLOC_14,
         UNUSED_MEM_PINS_PINLOC_15 => UNUSED_MEM_PINS_PINLOC_15,
         UNUSED_MEM_PINS_PINLOC_16 => UNUSED_MEM_PINS_PINLOC_16,
         UNUSED_MEM_PINS_PINLOC_17 => UNUSED_MEM_PINS_PINLOC_17,
         UNUSED_MEM_PINS_PINLOC_18 => UNUSED_MEM_PINS_PINLOC_18,
         UNUSED_MEM_PINS_PINLOC_19 => UNUSED_MEM_PINS_PINLOC_19,
         UNUSED_MEM_PINS_PINLOC_20 => UNUSED_MEM_PINS_PINLOC_20,
         UNUSED_MEM_PINS_PINLOC_21 => UNUSED_MEM_PINS_PINLOC_21,
         UNUSED_MEM_PINS_PINLOC_22 => UNUSED_MEM_PINS_PINLOC_22,
         UNUSED_MEM_PINS_PINLOC_23 => UNUSED_MEM_PINS_PINLOC_23,
         UNUSED_MEM_PINS_PINLOC_24 => UNUSED_MEM_PINS_PINLOC_24,
         UNUSED_MEM_PINS_PINLOC_25 => UNUSED_MEM_PINS_PINLOC_25,
         UNUSED_MEM_PINS_PINLOC_26 => UNUSED_MEM_PINS_PINLOC_26,
         UNUSED_MEM_PINS_PINLOC_27 => UNUSED_MEM_PINS_PINLOC_27,
         UNUSED_MEM_PINS_PINLOC_28 => UNUSED_MEM_PINS_PINLOC_28,
         UNUSED_MEM_PINS_PINLOC_29 => UNUSED_MEM_PINS_PINLOC_29,
         UNUSED_MEM_PINS_PINLOC_30 => UNUSED_MEM_PINS_PINLOC_30,
         UNUSED_MEM_PINS_PINLOC_31 => UNUSED_MEM_PINS_PINLOC_31,
         UNUSED_MEM_PINS_PINLOC_32 => UNUSED_MEM_PINS_PINLOC_32,
         UNUSED_MEM_PINS_PINLOC_33 => UNUSED_MEM_PINS_PINLOC_33,
         UNUSED_MEM_PINS_PINLOC_34 => UNUSED_MEM_PINS_PINLOC_34,
         UNUSED_MEM_PINS_PINLOC_35 => UNUSED_MEM_PINS_PINLOC_35,
         UNUSED_MEM_PINS_PINLOC_36 => UNUSED_MEM_PINS_PINLOC_36,
         UNUSED_MEM_PINS_PINLOC_37 => UNUSED_MEM_PINS_PINLOC_37,
         UNUSED_MEM_PINS_PINLOC_38 => UNUSED_MEM_PINS_PINLOC_38,
         UNUSED_MEM_PINS_PINLOC_39 => UNUSED_MEM_PINS_PINLOC_39,
         UNUSED_MEM_PINS_PINLOC_40 => UNUSED_MEM_PINS_PINLOC_40,
         UNUSED_MEM_PINS_PINLOC_41 => UNUSED_MEM_PINS_PINLOC_41,
         UNUSED_MEM_PINS_PINLOC_42 => UNUSED_MEM_PINS_PINLOC_42,
         UNUSED_MEM_PINS_PINLOC_43 => UNUSED_MEM_PINS_PINLOC_43,
         UNUSED_MEM_PINS_PINLOC_44 => UNUSED_MEM_PINS_PINLOC_44,
         UNUSED_MEM_PINS_PINLOC_45 => UNUSED_MEM_PINS_PINLOC_45,
         UNUSED_MEM_PINS_PINLOC_46 => UNUSED_MEM_PINS_PINLOC_46,
         UNUSED_MEM_PINS_PINLOC_47 => UNUSED_MEM_PINS_PINLOC_47,
         UNUSED_MEM_PINS_PINLOC_48 => UNUSED_MEM_PINS_PINLOC_48,
         UNUSED_MEM_PINS_PINLOC_49 => UNUSED_MEM_PINS_PINLOC_49,
         UNUSED_MEM_PINS_PINLOC_50 => UNUSED_MEM_PINS_PINLOC_50,
         UNUSED_MEM_PINS_PINLOC_51 => UNUSED_MEM_PINS_PINLOC_51,
         UNUSED_MEM_PINS_PINLOC_52 => UNUSED_MEM_PINS_PINLOC_52,
         UNUSED_MEM_PINS_PINLOC_53 => UNUSED_MEM_PINS_PINLOC_53,
         UNUSED_MEM_PINS_PINLOC_54 => UNUSED_MEM_PINS_PINLOC_54,
         UNUSED_MEM_PINS_PINLOC_55 => UNUSED_MEM_PINS_PINLOC_55,
         UNUSED_MEM_PINS_PINLOC_56 => UNUSED_MEM_PINS_PINLOC_56,
         UNUSED_MEM_PINS_PINLOC_57 => UNUSED_MEM_PINS_PINLOC_57,
         UNUSED_MEM_PINS_PINLOC_58 => UNUSED_MEM_PINS_PINLOC_58,
         UNUSED_MEM_PINS_PINLOC_59 => UNUSED_MEM_PINS_PINLOC_59,
         UNUSED_MEM_PINS_PINLOC_60 => UNUSED_MEM_PINS_PINLOC_60,
         UNUSED_MEM_PINS_PINLOC_61 => UNUSED_MEM_PINS_PINLOC_61,
         UNUSED_MEM_PINS_PINLOC_62 => UNUSED_MEM_PINS_PINLOC_62,
         UNUSED_MEM_PINS_PINLOC_63 => UNUSED_MEM_PINS_PINLOC_63,
         UNUSED_MEM_PINS_PINLOC_64 => UNUSED_MEM_PINS_PINLOC_64,
         UNUSED_MEM_PINS_PINLOC_65 => UNUSED_MEM_PINS_PINLOC_65,
         UNUSED_MEM_PINS_PINLOC_66 => UNUSED_MEM_PINS_PINLOC_66,
         UNUSED_MEM_PINS_PINLOC_67 => UNUSED_MEM_PINS_PINLOC_67,
         UNUSED_MEM_PINS_PINLOC_68 => UNUSED_MEM_PINS_PINLOC_68,
         UNUSED_MEM_PINS_PINLOC_69 => UNUSED_MEM_PINS_PINLOC_69,
         UNUSED_MEM_PINS_PINLOC_70 => UNUSED_MEM_PINS_PINLOC_70,
         UNUSED_MEM_PINS_PINLOC_71 => UNUSED_MEM_PINS_PINLOC_71,
         UNUSED_MEM_PINS_PINLOC_72 => UNUSED_MEM_PINS_PINLOC_72,
         UNUSED_MEM_PINS_PINLOC_73 => UNUSED_MEM_PINS_PINLOC_73,
         UNUSED_MEM_PINS_PINLOC_74 => UNUSED_MEM_PINS_PINLOC_74,
         UNUSED_MEM_PINS_PINLOC_75 => UNUSED_MEM_PINS_PINLOC_75,
         UNUSED_MEM_PINS_PINLOC_76 => UNUSED_MEM_PINS_PINLOC_76,
         UNUSED_MEM_PINS_PINLOC_77 => UNUSED_MEM_PINS_PINLOC_77,
         UNUSED_MEM_PINS_PINLOC_78 => UNUSED_MEM_PINS_PINLOC_78,
         UNUSED_MEM_PINS_PINLOC_79 => UNUSED_MEM_PINS_PINLOC_79,
         UNUSED_MEM_PINS_PINLOC_80 => UNUSED_MEM_PINS_PINLOC_80,
         UNUSED_MEM_PINS_PINLOC_81 => UNUSED_MEM_PINS_PINLOC_81,
         UNUSED_MEM_PINS_PINLOC_82 => UNUSED_MEM_PINS_PINLOC_82,
         UNUSED_MEM_PINS_PINLOC_83 => UNUSED_MEM_PINS_PINLOC_83,
         UNUSED_MEM_PINS_PINLOC_84 => UNUSED_MEM_PINS_PINLOC_84,
         UNUSED_MEM_PINS_PINLOC_85 => UNUSED_MEM_PINS_PINLOC_85,
         UNUSED_MEM_PINS_PINLOC_86 => UNUSED_MEM_PINS_PINLOC_86,
         UNUSED_MEM_PINS_PINLOC_87 => UNUSED_MEM_PINS_PINLOC_87,
         UNUSED_MEM_PINS_PINLOC_88 => UNUSED_MEM_PINS_PINLOC_88,
         UNUSED_MEM_PINS_PINLOC_89 => UNUSED_MEM_PINS_PINLOC_89,
         UNUSED_MEM_PINS_PINLOC_90 => UNUSED_MEM_PINS_PINLOC_90,
         UNUSED_MEM_PINS_PINLOC_91 => UNUSED_MEM_PINS_PINLOC_91,
         UNUSED_MEM_PINS_PINLOC_92 => UNUSED_MEM_PINS_PINLOC_92,
         UNUSED_MEM_PINS_PINLOC_93 => UNUSED_MEM_PINS_PINLOC_93,
         UNUSED_MEM_PINS_PINLOC_94 => UNUSED_MEM_PINS_PINLOC_94,
         UNUSED_MEM_PINS_PINLOC_95 => UNUSED_MEM_PINS_PINLOC_95,
         UNUSED_MEM_PINS_PINLOC_96 => UNUSED_MEM_PINS_PINLOC_96,
         UNUSED_MEM_PINS_PINLOC_97 => UNUSED_MEM_PINS_PINLOC_97,
         UNUSED_MEM_PINS_PINLOC_98 => UNUSED_MEM_PINS_PINLOC_98,
         UNUSED_MEM_PINS_PINLOC_99 => UNUSED_MEM_PINS_PINLOC_99,
         UNUSED_MEM_PINS_PINLOC_100 => UNUSED_MEM_PINS_PINLOC_100,
         UNUSED_MEM_PINS_PINLOC_101 => UNUSED_MEM_PINS_PINLOC_101,
         UNUSED_MEM_PINS_PINLOC_102 => UNUSED_MEM_PINS_PINLOC_102,
         UNUSED_MEM_PINS_PINLOC_103 => UNUSED_MEM_PINS_PINLOC_103,
         UNUSED_MEM_PINS_PINLOC_104 => UNUSED_MEM_PINS_PINLOC_104,
         UNUSED_MEM_PINS_PINLOC_105 => UNUSED_MEM_PINS_PINLOC_105,
         UNUSED_MEM_PINS_PINLOC_106 => UNUSED_MEM_PINS_PINLOC_106,
         UNUSED_MEM_PINS_PINLOC_107 => UNUSED_MEM_PINS_PINLOC_107,
         UNUSED_MEM_PINS_PINLOC_108 => UNUSED_MEM_PINS_PINLOC_108,
         UNUSED_MEM_PINS_PINLOC_109 => UNUSED_MEM_PINS_PINLOC_109,
         UNUSED_MEM_PINS_PINLOC_110 => UNUSED_MEM_PINS_PINLOC_110,
         UNUSED_MEM_PINS_PINLOC_111 => UNUSED_MEM_PINS_PINLOC_111,
         UNUSED_MEM_PINS_PINLOC_112 => UNUSED_MEM_PINS_PINLOC_112,
         UNUSED_MEM_PINS_PINLOC_113 => UNUSED_MEM_PINS_PINLOC_113,
         UNUSED_MEM_PINS_PINLOC_114 => UNUSED_MEM_PINS_PINLOC_114,
         UNUSED_MEM_PINS_PINLOC_115 => UNUSED_MEM_PINS_PINLOC_115,
         UNUSED_MEM_PINS_PINLOC_116 => UNUSED_MEM_PINS_PINLOC_116,
         UNUSED_MEM_PINS_PINLOC_117 => UNUSED_MEM_PINS_PINLOC_117,
         UNUSED_MEM_PINS_PINLOC_118 => UNUSED_MEM_PINS_PINLOC_118,
         UNUSED_MEM_PINS_PINLOC_119 => UNUSED_MEM_PINS_PINLOC_119,
         UNUSED_MEM_PINS_PINLOC_120 => UNUSED_MEM_PINS_PINLOC_120,
         UNUSED_MEM_PINS_PINLOC_121 => UNUSED_MEM_PINS_PINLOC_121,
         UNUSED_MEM_PINS_PINLOC_122 => UNUSED_MEM_PINS_PINLOC_122,
         UNUSED_MEM_PINS_PINLOC_123 => UNUSED_MEM_PINS_PINLOC_123,
         UNUSED_MEM_PINS_PINLOC_124 => UNUSED_MEM_PINS_PINLOC_124,
         UNUSED_MEM_PINS_PINLOC_125 => UNUSED_MEM_PINS_PINLOC_125,
         UNUSED_MEM_PINS_PINLOC_126 => UNUSED_MEM_PINS_PINLOC_126,
         UNUSED_MEM_PINS_PINLOC_127 => UNUSED_MEM_PINS_PINLOC_127,
         UNUSED_MEM_PINS_PINLOC_128 => UNUSED_MEM_PINS_PINLOC_128,
         UNUSED_MEM_PINS_PINLOC_AUTOGEN_WCNT => UNUSED_MEM_PINS_PINLOC_AUTOGEN_WCNT,
         UNUSED_DQS_BUSES_LANELOC_0 => UNUSED_DQS_BUSES_LANELOC_0,
         UNUSED_DQS_BUSES_LANELOC_1 => UNUSED_DQS_BUSES_LANELOC_1,
         UNUSED_DQS_BUSES_LANELOC_2 => UNUSED_DQS_BUSES_LANELOC_2,
         UNUSED_DQS_BUSES_LANELOC_3 => UNUSED_DQS_BUSES_LANELOC_3,
         UNUSED_DQS_BUSES_LANELOC_4 => UNUSED_DQS_BUSES_LANELOC_4,
         UNUSED_DQS_BUSES_LANELOC_5 => UNUSED_DQS_BUSES_LANELOC_5,
         UNUSED_DQS_BUSES_LANELOC_6 => UNUSED_DQS_BUSES_LANELOC_6,
         UNUSED_DQS_BUSES_LANELOC_7 => UNUSED_DQS_BUSES_LANELOC_7,
         UNUSED_DQS_BUSES_LANELOC_8 => UNUSED_DQS_BUSES_LANELOC_8,
         UNUSED_DQS_BUSES_LANELOC_9 => UNUSED_DQS_BUSES_LANELOC_9,
         UNUSED_DQS_BUSES_LANELOC_10 => UNUSED_DQS_BUSES_LANELOC_10,
         UNUSED_DQS_BUSES_LANELOC_AUTOGEN_WCNT => UNUSED_DQS_BUSES_LANELOC_AUTOGEN_WCNT,
         CENTER_TIDS_0 => CENTER_TIDS_0,
         CENTER_TIDS_1 => CENTER_TIDS_1,
         CENTER_TIDS_2 => CENTER_TIDS_2,
         CENTER_TIDS_AUTOGEN_WCNT => CENTER_TIDS_AUTOGEN_WCNT,
         HMC_TIDS_0 => HMC_TIDS_0,
         HMC_TIDS_1 => HMC_TIDS_1,
         HMC_TIDS_2 => HMC_TIDS_2,
         HMC_TIDS_AUTOGEN_WCNT => HMC_TIDS_AUTOGEN_WCNT,
         LANE_TIDS_0 => LANE_TIDS_0,
         LANE_TIDS_1 => LANE_TIDS_1,
         LANE_TIDS_2 => LANE_TIDS_2,
         LANE_TIDS_3 => LANE_TIDS_3,
         LANE_TIDS_4 => LANE_TIDS_4,
         LANE_TIDS_5 => LANE_TIDS_5,
         LANE_TIDS_6 => LANE_TIDS_6,
         LANE_TIDS_7 => LANE_TIDS_7,
         LANE_TIDS_8 => LANE_TIDS_8,
         LANE_TIDS_9 => LANE_TIDS_9,
         LANE_TIDS_AUTOGEN_WCNT => LANE_TIDS_AUTOGEN_WCNT,
         PREAMBLE_MODE => PREAMBLE_MODE,
         DBI_WR_ENABLE => DBI_WR_ENABLE,
         DBI_RD_ENABLE => DBI_RD_ENABLE,
         CRC_EN => CRC_EN,
         SWAP_DQS_A_B => SWAP_DQS_A_B,
         DQS_PACK_MODE => DQS_PACK_MODE,
         OCT_SIZE => OCT_SIZE,
         DBC_WB_RESERVED_ENTRY => DBC_WB_RESERVED_ENTRY,
         DLL_MODE => DLL_MODE,
         DLL_CODEWORD => DLL_CODEWORD,
         ABPHY_WRITE_PROTOCOL => ABPHY_WRITE_PROTOCOL,
         PHY_USERMODE_OCT => PHY_USERMODE_OCT,
         PHY_PERIODIC_OCT_RECAL => PHY_PERIODIC_OCT_RECAL,
         PHY_HAS_DCC => PHY_HAS_DCC,
         PRI_HMC_CFG_ENABLE_ECC => PRI_HMC_CFG_ENABLE_ECC,
         PRI_HMC_CFG_REORDER_DATA => PRI_HMC_CFG_REORDER_DATA,
         PRI_HMC_CFG_REORDER_READ => PRI_HMC_CFG_REORDER_READ,
         PRI_HMC_CFG_REORDER_RDATA => PRI_HMC_CFG_REORDER_RDATA,
         PRI_HMC_CFG_STARVE_LIMIT => PRI_HMC_CFG_STARVE_LIMIT,
         PRI_HMC_CFG_DQS_TRACKING_EN => PRI_HMC_CFG_DQS_TRACKING_EN,
         PRI_HMC_CFG_ARBITER_TYPE => PRI_HMC_CFG_ARBITER_TYPE,
         PRI_HMC_CFG_OPEN_PAGE_EN => PRI_HMC_CFG_OPEN_PAGE_EN,
         PRI_HMC_CFG_GEAR_DOWN_EN => PRI_HMC_CFG_GEAR_DOWN_EN,
         PRI_HMC_CFG_RLD3_MULTIBANK_MODE => PRI_HMC_CFG_RLD3_MULTIBANK_MODE,
         PRI_HMC_CFG_PING_PONG_MODE => PRI_HMC_CFG_PING_PONG_MODE,
         PRI_HMC_CFG_SLOT_ROTATE_EN => PRI_HMC_CFG_SLOT_ROTATE_EN,
         PRI_HMC_CFG_SLOT_OFFSET => PRI_HMC_CFG_SLOT_OFFSET,
         PRI_HMC_CFG_COL_CMD_SLOT => PRI_HMC_CFG_COL_CMD_SLOT,
         PRI_HMC_CFG_ROW_CMD_SLOT => PRI_HMC_CFG_ROW_CMD_SLOT,
         PRI_HMC_CFG_ENABLE_RC => PRI_HMC_CFG_ENABLE_RC,
         PRI_HMC_CFG_CS_TO_CHIP_MAPPING => PRI_HMC_CFG_CS_TO_CHIP_MAPPING,
         PRI_HMC_CFG_RB_RESERVED_ENTRY => PRI_HMC_CFG_RB_RESERVED_ENTRY,
         PRI_HMC_CFG_WB_RESERVED_ENTRY => PRI_HMC_CFG_WB_RESERVED_ENTRY,
         PRI_HMC_CFG_TCL => PRI_HMC_CFG_TCL,
         PRI_HMC_CFG_POWER_SAVING_EXIT_CYC => PRI_HMC_CFG_POWER_SAVING_EXIT_CYC,
         PRI_HMC_CFG_MEM_CLK_DISABLE_ENTRY_CYC => PRI_HMC_CFG_MEM_CLK_DISABLE_ENTRY_CYC,
         PRI_HMC_CFG_WRITE_ODT_CHIP => PRI_HMC_CFG_WRITE_ODT_CHIP,
         PRI_HMC_CFG_READ_ODT_CHIP => PRI_HMC_CFG_READ_ODT_CHIP,
         PRI_HMC_CFG_WR_ODT_ON => PRI_HMC_CFG_WR_ODT_ON,
         PRI_HMC_CFG_RD_ODT_ON => PRI_HMC_CFG_RD_ODT_ON,
         PRI_HMC_CFG_WR_ODT_PERIOD => PRI_HMC_CFG_WR_ODT_PERIOD,
         PRI_HMC_CFG_RD_ODT_PERIOD => PRI_HMC_CFG_RD_ODT_PERIOD,
         PRI_HMC_CFG_RLD3_REFRESH_SEQ0 => PRI_HMC_CFG_RLD3_REFRESH_SEQ0,
         PRI_HMC_CFG_RLD3_REFRESH_SEQ1 => PRI_HMC_CFG_RLD3_REFRESH_SEQ1,
         PRI_HMC_CFG_RLD3_REFRESH_SEQ2 => PRI_HMC_CFG_RLD3_REFRESH_SEQ2,
         PRI_HMC_CFG_RLD3_REFRESH_SEQ3 => PRI_HMC_CFG_RLD3_REFRESH_SEQ3,
         PRI_HMC_CFG_SRF_ZQCAL_DISABLE => PRI_HMC_CFG_SRF_ZQCAL_DISABLE,
         PRI_HMC_CFG_MPS_ZQCAL_DISABLE => PRI_HMC_CFG_MPS_ZQCAL_DISABLE,
         PRI_HMC_CFG_MPS_DQSTRK_DISABLE => PRI_HMC_CFG_MPS_DQSTRK_DISABLE,
         PRI_HMC_CFG_SHORT_DQSTRK_CTRL_EN => PRI_HMC_CFG_SHORT_DQSTRK_CTRL_EN,
         PRI_HMC_CFG_PERIOD_DQSTRK_CTRL_EN => PRI_HMC_CFG_PERIOD_DQSTRK_CTRL_EN,
         PRI_HMC_CFG_PERIOD_DQSTRK_INTERVAL => PRI_HMC_CFG_PERIOD_DQSTRK_INTERVAL,
         PRI_HMC_CFG_DQSTRK_TO_VALID_LAST => PRI_HMC_CFG_DQSTRK_TO_VALID_LAST,
         PRI_HMC_CFG_DQSTRK_TO_VALID => PRI_HMC_CFG_DQSTRK_TO_VALID,
         PRI_HMC_CFG_RFSH_WARN_THRESHOLD => PRI_HMC_CFG_RFSH_WARN_THRESHOLD,
         PRI_HMC_CFG_SB_CG_DISABLE => PRI_HMC_CFG_SB_CG_DISABLE,
         PRI_HMC_CFG_USER_RFSH_EN => PRI_HMC_CFG_USER_RFSH_EN,
         PRI_HMC_CFG_SRF_AUTOEXIT_EN => PRI_HMC_CFG_SRF_AUTOEXIT_EN,
         PRI_HMC_CFG_SRF_ENTRY_EXIT_BLOCK => PRI_HMC_CFG_SRF_ENTRY_EXIT_BLOCK,
         PRI_HMC_CFG_SB_DDR4_MR3 => PRI_HMC_CFG_SB_DDR4_MR3,
         PRI_HMC_CFG_SB_DDR4_MR4 => PRI_HMC_CFG_SB_DDR4_MR4,
         PRI_HMC_CFG_SB_DDR4_MR5 => PRI_HMC_CFG_SB_DDR4_MR5,
         PRI_HMC_CFG_DDR4_MPS_ADDR_MIRROR => PRI_HMC_CFG_DDR4_MPS_ADDR_MIRROR,
         PRI_HMC_CFG_MEM_IF_COLADDR_WIDTH => PRI_HMC_CFG_MEM_IF_COLADDR_WIDTH,
         PRI_HMC_CFG_MEM_IF_ROWADDR_WIDTH => PRI_HMC_CFG_MEM_IF_ROWADDR_WIDTH,
         PRI_HMC_CFG_MEM_IF_BANKADDR_WIDTH => PRI_HMC_CFG_MEM_IF_BANKADDR_WIDTH,
         PRI_HMC_CFG_MEM_IF_BGADDR_WIDTH => PRI_HMC_CFG_MEM_IF_BGADDR_WIDTH,
         PRI_HMC_CFG_LOCAL_IF_CS_WIDTH => PRI_HMC_CFG_LOCAL_IF_CS_WIDTH,
         PRI_HMC_CFG_ADDR_ORDER => PRI_HMC_CFG_ADDR_ORDER,
         PRI_HMC_CFG_ACT_TO_RDWR => PRI_HMC_CFG_ACT_TO_RDWR,
         PRI_HMC_CFG_ACT_TO_PCH => PRI_HMC_CFG_ACT_TO_PCH,
         PRI_HMC_CFG_ACT_TO_ACT => PRI_HMC_CFG_ACT_TO_ACT,
         PRI_HMC_CFG_ACT_TO_ACT_DIFF_BANK => PRI_HMC_CFG_ACT_TO_ACT_DIFF_BANK,
         PRI_HMC_CFG_ACT_TO_ACT_DIFF_BG => PRI_HMC_CFG_ACT_TO_ACT_DIFF_BG,
         PRI_HMC_CFG_RD_TO_RD => PRI_HMC_CFG_RD_TO_RD,
         PRI_HMC_CFG_RD_TO_RD_DIFF_CHIP => PRI_HMC_CFG_RD_TO_RD_DIFF_CHIP,
         PRI_HMC_CFG_RD_TO_RD_DIFF_BG => PRI_HMC_CFG_RD_TO_RD_DIFF_BG,
         PRI_HMC_CFG_RD_TO_WR => PRI_HMC_CFG_RD_TO_WR,
         PRI_HMC_CFG_RD_TO_WR_DIFF_CHIP => PRI_HMC_CFG_RD_TO_WR_DIFF_CHIP,
         PRI_HMC_CFG_RD_TO_WR_DIFF_BG => PRI_HMC_CFG_RD_TO_WR_DIFF_BG,
         PRI_HMC_CFG_RD_TO_PCH => PRI_HMC_CFG_RD_TO_PCH,
         PRI_HMC_CFG_RD_AP_TO_VALID => PRI_HMC_CFG_RD_AP_TO_VALID,
         PRI_HMC_CFG_WR_TO_WR => PRI_HMC_CFG_WR_TO_WR,
         PRI_HMC_CFG_WR_TO_WR_DIFF_CHIP => PRI_HMC_CFG_WR_TO_WR_DIFF_CHIP,
         PRI_HMC_CFG_WR_TO_WR_DIFF_BG => PRI_HMC_CFG_WR_TO_WR_DIFF_BG,
         PRI_HMC_CFG_WR_TO_RD => PRI_HMC_CFG_WR_TO_RD,
         PRI_HMC_CFG_WR_TO_RD_DIFF_CHIP => PRI_HMC_CFG_WR_TO_RD_DIFF_CHIP,
         PRI_HMC_CFG_WR_TO_RD_DIFF_BG => PRI_HMC_CFG_WR_TO_RD_DIFF_BG,
         PRI_HMC_CFG_WR_TO_PCH => PRI_HMC_CFG_WR_TO_PCH,
         PRI_HMC_CFG_WR_AP_TO_VALID => PRI_HMC_CFG_WR_AP_TO_VALID,
         PRI_HMC_CFG_PCH_TO_VALID => PRI_HMC_CFG_PCH_TO_VALID,
         PRI_HMC_CFG_PCH_ALL_TO_VALID => PRI_HMC_CFG_PCH_ALL_TO_VALID,
         PRI_HMC_CFG_ARF_TO_VALID => PRI_HMC_CFG_ARF_TO_VALID,
         PRI_HMC_CFG_PDN_TO_VALID => PRI_HMC_CFG_PDN_TO_VALID,
         PRI_HMC_CFG_SRF_TO_VALID => PRI_HMC_CFG_SRF_TO_VALID,
         PRI_HMC_CFG_SRF_TO_ZQ_CAL => PRI_HMC_CFG_SRF_TO_ZQ_CAL,
         PRI_HMC_CFG_ARF_PERIOD => PRI_HMC_CFG_ARF_PERIOD,
         PRI_HMC_CFG_PDN_PERIOD => PRI_HMC_CFG_PDN_PERIOD,
         PRI_HMC_CFG_ZQCL_TO_VALID => PRI_HMC_CFG_ZQCL_TO_VALID,
         PRI_HMC_CFG_ZQCS_TO_VALID => PRI_HMC_CFG_ZQCS_TO_VALID,
         PRI_HMC_CFG_MRS_TO_VALID => PRI_HMC_CFG_MRS_TO_VALID,
         PRI_HMC_CFG_MPS_TO_VALID => PRI_HMC_CFG_MPS_TO_VALID,
         PRI_HMC_CFG_MRR_TO_VALID => PRI_HMC_CFG_MRR_TO_VALID,
         PRI_HMC_CFG_MPR_TO_VALID => PRI_HMC_CFG_MPR_TO_VALID,
         PRI_HMC_CFG_MPS_EXIT_CS_TO_CKE => PRI_HMC_CFG_MPS_EXIT_CS_TO_CKE,
         PRI_HMC_CFG_MPS_EXIT_CKE_TO_CS => PRI_HMC_CFG_MPS_EXIT_CKE_TO_CS,
         PRI_HMC_CFG_RLD3_MULTIBANK_REF_DELAY => PRI_HMC_CFG_RLD3_MULTIBANK_REF_DELAY,
         PRI_HMC_CFG_MMR_CMD_TO_VALID => PRI_HMC_CFG_MMR_CMD_TO_VALID,
         PRI_HMC_CFG_4_ACT_TO_ACT => PRI_HMC_CFG_4_ACT_TO_ACT,
         PRI_HMC_CFG_16_ACT_TO_ACT => PRI_HMC_CFG_16_ACT_TO_ACT,
         SEC_HMC_CFG_ENABLE_ECC => SEC_HMC_CFG_ENABLE_ECC,
         SEC_HMC_CFG_REORDER_DATA => SEC_HMC_CFG_REORDER_DATA,
         SEC_HMC_CFG_REORDER_READ => SEC_HMC_CFG_REORDER_READ,
         SEC_HMC_CFG_REORDER_RDATA => SEC_HMC_CFG_REORDER_RDATA,
         SEC_HMC_CFG_STARVE_LIMIT => SEC_HMC_CFG_STARVE_LIMIT,
         SEC_HMC_CFG_DQS_TRACKING_EN => SEC_HMC_CFG_DQS_TRACKING_EN,
         SEC_HMC_CFG_ARBITER_TYPE => SEC_HMC_CFG_ARBITER_TYPE,
         SEC_HMC_CFG_OPEN_PAGE_EN => SEC_HMC_CFG_OPEN_PAGE_EN,
         SEC_HMC_CFG_GEAR_DOWN_EN => SEC_HMC_CFG_GEAR_DOWN_EN,
         SEC_HMC_CFG_RLD3_MULTIBANK_MODE => SEC_HMC_CFG_RLD3_MULTIBANK_MODE,
         SEC_HMC_CFG_PING_PONG_MODE => SEC_HMC_CFG_PING_PONG_MODE,
         SEC_HMC_CFG_SLOT_ROTATE_EN => SEC_HMC_CFG_SLOT_ROTATE_EN,
         SEC_HMC_CFG_SLOT_OFFSET => SEC_HMC_CFG_SLOT_OFFSET,
         SEC_HMC_CFG_COL_CMD_SLOT => SEC_HMC_CFG_COL_CMD_SLOT,
         SEC_HMC_CFG_ROW_CMD_SLOT => SEC_HMC_CFG_ROW_CMD_SLOT,
         SEC_HMC_CFG_ENABLE_RC => SEC_HMC_CFG_ENABLE_RC,
         SEC_HMC_CFG_CS_TO_CHIP_MAPPING => SEC_HMC_CFG_CS_TO_CHIP_MAPPING,
         SEC_HMC_CFG_RB_RESERVED_ENTRY => SEC_HMC_CFG_RB_RESERVED_ENTRY,
         SEC_HMC_CFG_WB_RESERVED_ENTRY => SEC_HMC_CFG_WB_RESERVED_ENTRY,
         SEC_HMC_CFG_TCL => SEC_HMC_CFG_TCL,
         SEC_HMC_CFG_POWER_SAVING_EXIT_CYC => SEC_HMC_CFG_POWER_SAVING_EXIT_CYC,
         SEC_HMC_CFG_MEM_CLK_DISABLE_ENTRY_CYC => SEC_HMC_CFG_MEM_CLK_DISABLE_ENTRY_CYC,
         SEC_HMC_CFG_WRITE_ODT_CHIP => SEC_HMC_CFG_WRITE_ODT_CHIP,
         SEC_HMC_CFG_READ_ODT_CHIP => SEC_HMC_CFG_READ_ODT_CHIP,
         SEC_HMC_CFG_WR_ODT_ON => SEC_HMC_CFG_WR_ODT_ON,
         SEC_HMC_CFG_RD_ODT_ON => SEC_HMC_CFG_RD_ODT_ON,
         SEC_HMC_CFG_WR_ODT_PERIOD => SEC_HMC_CFG_WR_ODT_PERIOD,
         SEC_HMC_CFG_RD_ODT_PERIOD => SEC_HMC_CFG_RD_ODT_PERIOD,
         SEC_HMC_CFG_RLD3_REFRESH_SEQ0 => SEC_HMC_CFG_RLD3_REFRESH_SEQ0,
         SEC_HMC_CFG_RLD3_REFRESH_SEQ1 => SEC_HMC_CFG_RLD3_REFRESH_SEQ1,
         SEC_HMC_CFG_RLD3_REFRESH_SEQ2 => SEC_HMC_CFG_RLD3_REFRESH_SEQ2,
         SEC_HMC_CFG_RLD3_REFRESH_SEQ3 => SEC_HMC_CFG_RLD3_REFRESH_SEQ3,
         SEC_HMC_CFG_SRF_ZQCAL_DISABLE => SEC_HMC_CFG_SRF_ZQCAL_DISABLE,
         SEC_HMC_CFG_MPS_ZQCAL_DISABLE => SEC_HMC_CFG_MPS_ZQCAL_DISABLE,
         SEC_HMC_CFG_MPS_DQSTRK_DISABLE => SEC_HMC_CFG_MPS_DQSTRK_DISABLE,
         SEC_HMC_CFG_SHORT_DQSTRK_CTRL_EN => SEC_HMC_CFG_SHORT_DQSTRK_CTRL_EN,
         SEC_HMC_CFG_PERIOD_DQSTRK_CTRL_EN => SEC_HMC_CFG_PERIOD_DQSTRK_CTRL_EN,
         SEC_HMC_CFG_PERIOD_DQSTRK_INTERVAL => SEC_HMC_CFG_PERIOD_DQSTRK_INTERVAL,
         SEC_HMC_CFG_DQSTRK_TO_VALID_LAST => SEC_HMC_CFG_DQSTRK_TO_VALID_LAST,
         SEC_HMC_CFG_DQSTRK_TO_VALID => SEC_HMC_CFG_DQSTRK_TO_VALID,
         SEC_HMC_CFG_RFSH_WARN_THRESHOLD => SEC_HMC_CFG_RFSH_WARN_THRESHOLD,
         SEC_HMC_CFG_SB_CG_DISABLE => SEC_HMC_CFG_SB_CG_DISABLE,
         SEC_HMC_CFG_USER_RFSH_EN => SEC_HMC_CFG_USER_RFSH_EN,
         SEC_HMC_CFG_SRF_AUTOEXIT_EN => SEC_HMC_CFG_SRF_AUTOEXIT_EN,
         SEC_HMC_CFG_SRF_ENTRY_EXIT_BLOCK => SEC_HMC_CFG_SRF_ENTRY_EXIT_BLOCK,
         SEC_HMC_CFG_SB_DDR4_MR3 => SEC_HMC_CFG_SB_DDR4_MR3,
         SEC_HMC_CFG_SB_DDR4_MR4 => SEC_HMC_CFG_SB_DDR4_MR4,
         SEC_HMC_CFG_SB_DDR4_MR5 => SEC_HMC_CFG_SB_DDR4_MR5,
         SEC_HMC_CFG_DDR4_MPS_ADDR_MIRROR => SEC_HMC_CFG_DDR4_MPS_ADDR_MIRROR,
         SEC_HMC_CFG_MEM_IF_COLADDR_WIDTH => SEC_HMC_CFG_MEM_IF_COLADDR_WIDTH,
         SEC_HMC_CFG_MEM_IF_ROWADDR_WIDTH => SEC_HMC_CFG_MEM_IF_ROWADDR_WIDTH,
         SEC_HMC_CFG_MEM_IF_BANKADDR_WIDTH => SEC_HMC_CFG_MEM_IF_BANKADDR_WIDTH,
         SEC_HMC_CFG_MEM_IF_BGADDR_WIDTH => SEC_HMC_CFG_MEM_IF_BGADDR_WIDTH,
         SEC_HMC_CFG_LOCAL_IF_CS_WIDTH => SEC_HMC_CFG_LOCAL_IF_CS_WIDTH,
         SEC_HMC_CFG_ADDR_ORDER => SEC_HMC_CFG_ADDR_ORDER,
         SEC_HMC_CFG_ACT_TO_RDWR => SEC_HMC_CFG_ACT_TO_RDWR,
         SEC_HMC_CFG_ACT_TO_PCH => SEC_HMC_CFG_ACT_TO_PCH,
         SEC_HMC_CFG_ACT_TO_ACT => SEC_HMC_CFG_ACT_TO_ACT,
         SEC_HMC_CFG_ACT_TO_ACT_DIFF_BANK => SEC_HMC_CFG_ACT_TO_ACT_DIFF_BANK,
         SEC_HMC_CFG_ACT_TO_ACT_DIFF_BG => SEC_HMC_CFG_ACT_TO_ACT_DIFF_BG,
         SEC_HMC_CFG_RD_TO_RD => SEC_HMC_CFG_RD_TO_RD,
         SEC_HMC_CFG_RD_TO_RD_DIFF_CHIP => SEC_HMC_CFG_RD_TO_RD_DIFF_CHIP,
         SEC_HMC_CFG_RD_TO_RD_DIFF_BG => SEC_HMC_CFG_RD_TO_RD_DIFF_BG,
         SEC_HMC_CFG_RD_TO_WR => SEC_HMC_CFG_RD_TO_WR,
         SEC_HMC_CFG_RD_TO_WR_DIFF_CHIP => SEC_HMC_CFG_RD_TO_WR_DIFF_CHIP,
         SEC_HMC_CFG_RD_TO_WR_DIFF_BG => SEC_HMC_CFG_RD_TO_WR_DIFF_BG,
         SEC_HMC_CFG_RD_TO_PCH => SEC_HMC_CFG_RD_TO_PCH,
         SEC_HMC_CFG_RD_AP_TO_VALID => SEC_HMC_CFG_RD_AP_TO_VALID,
         SEC_HMC_CFG_WR_TO_WR => SEC_HMC_CFG_WR_TO_WR,
         SEC_HMC_CFG_WR_TO_WR_DIFF_CHIP => SEC_HMC_CFG_WR_TO_WR_DIFF_CHIP,
         SEC_HMC_CFG_WR_TO_WR_DIFF_BG => SEC_HMC_CFG_WR_TO_WR_DIFF_BG,
         SEC_HMC_CFG_WR_TO_RD => SEC_HMC_CFG_WR_TO_RD,
         SEC_HMC_CFG_WR_TO_RD_DIFF_CHIP => SEC_HMC_CFG_WR_TO_RD_DIFF_CHIP,
         SEC_HMC_CFG_WR_TO_RD_DIFF_BG => SEC_HMC_CFG_WR_TO_RD_DIFF_BG,
         SEC_HMC_CFG_WR_TO_PCH => SEC_HMC_CFG_WR_TO_PCH,
         SEC_HMC_CFG_WR_AP_TO_VALID => SEC_HMC_CFG_WR_AP_TO_VALID,
         SEC_HMC_CFG_PCH_TO_VALID => SEC_HMC_CFG_PCH_TO_VALID,
         SEC_HMC_CFG_PCH_ALL_TO_VALID => SEC_HMC_CFG_PCH_ALL_TO_VALID,
         SEC_HMC_CFG_ARF_TO_VALID => SEC_HMC_CFG_ARF_TO_VALID,
         SEC_HMC_CFG_PDN_TO_VALID => SEC_HMC_CFG_PDN_TO_VALID,
         SEC_HMC_CFG_SRF_TO_VALID => SEC_HMC_CFG_SRF_TO_VALID,
         SEC_HMC_CFG_SRF_TO_ZQ_CAL => SEC_HMC_CFG_SRF_TO_ZQ_CAL,
         SEC_HMC_CFG_ARF_PERIOD => SEC_HMC_CFG_ARF_PERIOD,
         SEC_HMC_CFG_PDN_PERIOD => SEC_HMC_CFG_PDN_PERIOD,
         SEC_HMC_CFG_ZQCL_TO_VALID => SEC_HMC_CFG_ZQCL_TO_VALID,
         SEC_HMC_CFG_ZQCS_TO_VALID => SEC_HMC_CFG_ZQCS_TO_VALID,
         SEC_HMC_CFG_MRS_TO_VALID => SEC_HMC_CFG_MRS_TO_VALID,
         SEC_HMC_CFG_MPS_TO_VALID => SEC_HMC_CFG_MPS_TO_VALID,
         SEC_HMC_CFG_MRR_TO_VALID => SEC_HMC_CFG_MRR_TO_VALID,
         SEC_HMC_CFG_MPR_TO_VALID => SEC_HMC_CFG_MPR_TO_VALID,
         SEC_HMC_CFG_MPS_EXIT_CS_TO_CKE => SEC_HMC_CFG_MPS_EXIT_CS_TO_CKE,
         SEC_HMC_CFG_MPS_EXIT_CKE_TO_CS => SEC_HMC_CFG_MPS_EXIT_CKE_TO_CS,
         SEC_HMC_CFG_RLD3_MULTIBANK_REF_DELAY => SEC_HMC_CFG_RLD3_MULTIBANK_REF_DELAY,
         SEC_HMC_CFG_MMR_CMD_TO_VALID => SEC_HMC_CFG_MMR_CMD_TO_VALID,
         SEC_HMC_CFG_4_ACT_TO_ACT => SEC_HMC_CFG_4_ACT_TO_ACT,
         SEC_HMC_CFG_16_ACT_TO_ACT => SEC_HMC_CFG_16_ACT_TO_ACT,
         PINS_PER_LANE => PINS_PER_LANE,
         LANES_PER_TILE => LANES_PER_TILE,
         OCT_CONTROL_WIDTH => OCT_CONTROL_WIDTH,
         PORT_MEM_CK_WIDTH => PORT_MEM_CK_WIDTH,
         PORT_MEM_CK_PINLOC_0 => PORT_MEM_CK_PINLOC_0,
         PORT_MEM_CK_PINLOC_1 => PORT_MEM_CK_PINLOC_1,
         PORT_MEM_CK_PINLOC_2 => PORT_MEM_CK_PINLOC_2,
         PORT_MEM_CK_PINLOC_3 => PORT_MEM_CK_PINLOC_3,
         PORT_MEM_CK_PINLOC_4 => PORT_MEM_CK_PINLOC_4,
         PORT_MEM_CK_PINLOC_5 => PORT_MEM_CK_PINLOC_5,
         PORT_MEM_CK_PINLOC_AUTOGEN_WCNT => PORT_MEM_CK_PINLOC_AUTOGEN_WCNT,
         PORT_MEM_CK_N_WIDTH => PORT_MEM_CK_N_WIDTH,
         PORT_MEM_CK_N_PINLOC_0 => PORT_MEM_CK_N_PINLOC_0,
         PORT_MEM_CK_N_PINLOC_1 => PORT_MEM_CK_N_PINLOC_1,
         PORT_MEM_CK_N_PINLOC_2 => PORT_MEM_CK_N_PINLOC_2,
         PORT_MEM_CK_N_PINLOC_3 => PORT_MEM_CK_N_PINLOC_3,
         PORT_MEM_CK_N_PINLOC_4 => PORT_MEM_CK_N_PINLOC_4,
         PORT_MEM_CK_N_PINLOC_5 => PORT_MEM_CK_N_PINLOC_5,
         PORT_MEM_CK_N_PINLOC_AUTOGEN_WCNT => PORT_MEM_CK_N_PINLOC_AUTOGEN_WCNT,
         PORT_MEM_CK_BIDIR_WIDTH => PORT_MEM_CK_BIDIR_WIDTH,
         PORT_MEM_CK_BIDIR_PINLOC_0 => PORT_MEM_CK_BIDIR_PINLOC_0,
         PORT_MEM_CK_BIDIR_PINLOC_1 => PORT_MEM_CK_BIDIR_PINLOC_1,
         PORT_MEM_CK_BIDIR_PINLOC_2 => PORT_MEM_CK_BIDIR_PINLOC_2,
         PORT_MEM_CK_BIDIR_PINLOC_3 => PORT_MEM_CK_BIDIR_PINLOC_3,
         PORT_MEM_CK_BIDIR_PINLOC_4 => PORT_MEM_CK_BIDIR_PINLOC_4,
         PORT_MEM_CK_BIDIR_PINLOC_5 => PORT_MEM_CK_BIDIR_PINLOC_5,
         PORT_MEM_CK_BIDIR_PINLOC_AUTOGEN_WCNT => PORT_MEM_CK_BIDIR_PINLOC_AUTOGEN_WCNT,
         PORT_MEM_CK_BIDIR_N_WIDTH => PORT_MEM_CK_BIDIR_N_WIDTH,
         PORT_MEM_CK_BIDIR_N_PINLOC_0 => PORT_MEM_CK_BIDIR_N_PINLOC_0,
         PORT_MEM_CK_BIDIR_N_PINLOC_1 => PORT_MEM_CK_BIDIR_N_PINLOC_1,
         PORT_MEM_CK_BIDIR_N_PINLOC_2 => PORT_MEM_CK_BIDIR_N_PINLOC_2,
         PORT_MEM_CK_BIDIR_N_PINLOC_3 => PORT_MEM_CK_BIDIR_N_PINLOC_3,
         PORT_MEM_CK_BIDIR_N_PINLOC_4 => PORT_MEM_CK_BIDIR_N_PINLOC_4,
         PORT_MEM_CK_BIDIR_N_PINLOC_5 => PORT_MEM_CK_BIDIR_N_PINLOC_5,
         PORT_MEM_CK_BIDIR_N_PINLOC_AUTOGEN_WCNT => PORT_MEM_CK_BIDIR_N_PINLOC_AUTOGEN_WCNT,
         PORT_MEM_DK_WIDTH => PORT_MEM_DK_WIDTH,
         PORT_MEM_DK_PINLOC_0 => PORT_MEM_DK_PINLOC_0,
         PORT_MEM_DK_PINLOC_1 => PORT_MEM_DK_PINLOC_1,
         PORT_MEM_DK_PINLOC_2 => PORT_MEM_DK_PINLOC_2,
         PORT_MEM_DK_PINLOC_3 => PORT_MEM_DK_PINLOC_3,
         PORT_MEM_DK_PINLOC_4 => PORT_MEM_DK_PINLOC_4,
         PORT_MEM_DK_PINLOC_5 => PORT_MEM_DK_PINLOC_5,
         PORT_MEM_DK_PINLOC_AUTOGEN_WCNT => PORT_MEM_DK_PINLOC_AUTOGEN_WCNT,
         PORT_MEM_DK_N_WIDTH => PORT_MEM_DK_N_WIDTH,
         PORT_MEM_DK_N_PINLOC_0 => PORT_MEM_DK_N_PINLOC_0,
         PORT_MEM_DK_N_PINLOC_1 => PORT_MEM_DK_N_PINLOC_1,
         PORT_MEM_DK_N_PINLOC_2 => PORT_MEM_DK_N_PINLOC_2,
         PORT_MEM_DK_N_PINLOC_3 => PORT_MEM_DK_N_PINLOC_3,
         PORT_MEM_DK_N_PINLOC_4 => PORT_MEM_DK_N_PINLOC_4,
         PORT_MEM_DK_N_PINLOC_5 => PORT_MEM_DK_N_PINLOC_5,
         PORT_MEM_DK_N_PINLOC_AUTOGEN_WCNT => PORT_MEM_DK_N_PINLOC_AUTOGEN_WCNT,
         PORT_MEM_DKA_WIDTH => PORT_MEM_DKA_WIDTH,
         PORT_MEM_DKA_PINLOC_0 => PORT_MEM_DKA_PINLOC_0,
         PORT_MEM_DKA_PINLOC_1 => PORT_MEM_DKA_PINLOC_1,
         PORT_MEM_DKA_PINLOC_2 => PORT_MEM_DKA_PINLOC_2,
         PORT_MEM_DKA_PINLOC_3 => PORT_MEM_DKA_PINLOC_3,
         PORT_MEM_DKA_PINLOC_4 => PORT_MEM_DKA_PINLOC_4,
         PORT_MEM_DKA_PINLOC_5 => PORT_MEM_DKA_PINLOC_5,
         PORT_MEM_DKA_PINLOC_AUTOGEN_WCNT => PORT_MEM_DKA_PINLOC_AUTOGEN_WCNT,
         PORT_MEM_DKA_N_WIDTH => PORT_MEM_DKA_N_WIDTH,
         PORT_MEM_DKA_N_PINLOC_0 => PORT_MEM_DKA_N_PINLOC_0,
         PORT_MEM_DKA_N_PINLOC_1 => PORT_MEM_DKA_N_PINLOC_1,
         PORT_MEM_DKA_N_PINLOC_2 => PORT_MEM_DKA_N_PINLOC_2,
         PORT_MEM_DKA_N_PINLOC_3 => PORT_MEM_DKA_N_PINLOC_3,
         PORT_MEM_DKA_N_PINLOC_4 => PORT_MEM_DKA_N_PINLOC_4,
         PORT_MEM_DKA_N_PINLOC_5 => PORT_MEM_DKA_N_PINLOC_5,
         PORT_MEM_DKA_N_PINLOC_AUTOGEN_WCNT => PORT_MEM_DKA_N_PINLOC_AUTOGEN_WCNT,
         PORT_MEM_DKB_WIDTH => PORT_MEM_DKB_WIDTH,
         PORT_MEM_DKB_PINLOC_0 => PORT_MEM_DKB_PINLOC_0,
         PORT_MEM_DKB_PINLOC_1 => PORT_MEM_DKB_PINLOC_1,
         PORT_MEM_DKB_PINLOC_2 => PORT_MEM_DKB_PINLOC_2,
         PORT_MEM_DKB_PINLOC_3 => PORT_MEM_DKB_PINLOC_3,
         PORT_MEM_DKB_PINLOC_4 => PORT_MEM_DKB_PINLOC_4,
         PORT_MEM_DKB_PINLOC_5 => PORT_MEM_DKB_PINLOC_5,
         PORT_MEM_DKB_PINLOC_AUTOGEN_WCNT => PORT_MEM_DKB_PINLOC_AUTOGEN_WCNT,
         PORT_MEM_DKB_N_WIDTH => PORT_MEM_DKB_N_WIDTH,
         PORT_MEM_DKB_N_PINLOC_0 => PORT_MEM_DKB_N_PINLOC_0,
         PORT_MEM_DKB_N_PINLOC_1 => PORT_MEM_DKB_N_PINLOC_1,
         PORT_MEM_DKB_N_PINLOC_2 => PORT_MEM_DKB_N_PINLOC_2,
         PORT_MEM_DKB_N_PINLOC_3 => PORT_MEM_DKB_N_PINLOC_3,
         PORT_MEM_DKB_N_PINLOC_4 => PORT_MEM_DKB_N_PINLOC_4,
         PORT_MEM_DKB_N_PINLOC_5 => PORT_MEM_DKB_N_PINLOC_5,
         PORT_MEM_DKB_N_PINLOC_AUTOGEN_WCNT => PORT_MEM_DKB_N_PINLOC_AUTOGEN_WCNT,
         PORT_MEM_K_WIDTH => PORT_MEM_K_WIDTH,
         PORT_MEM_K_PINLOC_0 => PORT_MEM_K_PINLOC_0,
         PORT_MEM_K_PINLOC_1 => PORT_MEM_K_PINLOC_1,
         PORT_MEM_K_PINLOC_2 => PORT_MEM_K_PINLOC_2,
         PORT_MEM_K_PINLOC_3 => PORT_MEM_K_PINLOC_3,
         PORT_MEM_K_PINLOC_4 => PORT_MEM_K_PINLOC_4,
         PORT_MEM_K_PINLOC_5 => PORT_MEM_K_PINLOC_5,
         PORT_MEM_K_PINLOC_AUTOGEN_WCNT => PORT_MEM_K_PINLOC_AUTOGEN_WCNT,
         PORT_MEM_K_N_WIDTH => PORT_MEM_K_N_WIDTH,
         PORT_MEM_K_N_PINLOC_0 => PORT_MEM_K_N_PINLOC_0,
         PORT_MEM_K_N_PINLOC_1 => PORT_MEM_K_N_PINLOC_1,
         PORT_MEM_K_N_PINLOC_2 => PORT_MEM_K_N_PINLOC_2,
         PORT_MEM_K_N_PINLOC_3 => PORT_MEM_K_N_PINLOC_3,
         PORT_MEM_K_N_PINLOC_4 => PORT_MEM_K_N_PINLOC_4,
         PORT_MEM_K_N_PINLOC_5 => PORT_MEM_K_N_PINLOC_5,
         PORT_MEM_K_N_PINLOC_AUTOGEN_WCNT => PORT_MEM_K_N_PINLOC_AUTOGEN_WCNT,
         PORT_MEM_A_WIDTH => PORT_MEM_A_WIDTH,
         PORT_MEM_A_PINLOC_0 => PORT_MEM_A_PINLOC_0,
         PORT_MEM_A_PINLOC_1 => PORT_MEM_A_PINLOC_1,
         PORT_MEM_A_PINLOC_2 => PORT_MEM_A_PINLOC_2,
         PORT_MEM_A_PINLOC_3 => PORT_MEM_A_PINLOC_3,
         PORT_MEM_A_PINLOC_4 => PORT_MEM_A_PINLOC_4,
         PORT_MEM_A_PINLOC_5 => PORT_MEM_A_PINLOC_5,
         PORT_MEM_A_PINLOC_6 => PORT_MEM_A_PINLOC_6,
         PORT_MEM_A_PINLOC_7 => PORT_MEM_A_PINLOC_7,
         PORT_MEM_A_PINLOC_8 => PORT_MEM_A_PINLOC_8,
         PORT_MEM_A_PINLOC_9 => PORT_MEM_A_PINLOC_9,
         PORT_MEM_A_PINLOC_10 => PORT_MEM_A_PINLOC_10,
         PORT_MEM_A_PINLOC_11 => PORT_MEM_A_PINLOC_11,
         PORT_MEM_A_PINLOC_12 => PORT_MEM_A_PINLOC_12,
         PORT_MEM_A_PINLOC_13 => PORT_MEM_A_PINLOC_13,
         PORT_MEM_A_PINLOC_14 => PORT_MEM_A_PINLOC_14,
         PORT_MEM_A_PINLOC_15 => PORT_MEM_A_PINLOC_15,
         PORT_MEM_A_PINLOC_16 => PORT_MEM_A_PINLOC_16,
         PORT_MEM_A_PINLOC_AUTOGEN_WCNT => PORT_MEM_A_PINLOC_AUTOGEN_WCNT,
         PORT_MEM_BA_WIDTH => PORT_MEM_BA_WIDTH,
         PORT_MEM_BA_PINLOC_0 => PORT_MEM_BA_PINLOC_0,
         PORT_MEM_BA_PINLOC_1 => PORT_MEM_BA_PINLOC_1,
         PORT_MEM_BA_PINLOC_2 => PORT_MEM_BA_PINLOC_2,
         PORT_MEM_BA_PINLOC_3 => PORT_MEM_BA_PINLOC_3,
         PORT_MEM_BA_PINLOC_4 => PORT_MEM_BA_PINLOC_4,
         PORT_MEM_BA_PINLOC_5 => PORT_MEM_BA_PINLOC_5,
         PORT_MEM_BA_PINLOC_AUTOGEN_WCNT => PORT_MEM_BA_PINLOC_AUTOGEN_WCNT,
         PORT_MEM_BG_WIDTH => PORT_MEM_BG_WIDTH,
         PORT_MEM_BG_PINLOC_0 => PORT_MEM_BG_PINLOC_0,
         PORT_MEM_BG_PINLOC_1 => PORT_MEM_BG_PINLOC_1,
         PORT_MEM_BG_PINLOC_2 => PORT_MEM_BG_PINLOC_2,
         PORT_MEM_BG_PINLOC_3 => PORT_MEM_BG_PINLOC_3,
         PORT_MEM_BG_PINLOC_4 => PORT_MEM_BG_PINLOC_4,
         PORT_MEM_BG_PINLOC_5 => PORT_MEM_BG_PINLOC_5,
         PORT_MEM_BG_PINLOC_AUTOGEN_WCNT => PORT_MEM_BG_PINLOC_AUTOGEN_WCNT,
         PORT_MEM_C_WIDTH => PORT_MEM_C_WIDTH,
         PORT_MEM_C_PINLOC_0 => PORT_MEM_C_PINLOC_0,
         PORT_MEM_C_PINLOC_1 => PORT_MEM_C_PINLOC_1,
         PORT_MEM_C_PINLOC_2 => PORT_MEM_C_PINLOC_2,
         PORT_MEM_C_PINLOC_3 => PORT_MEM_C_PINLOC_3,
         PORT_MEM_C_PINLOC_4 => PORT_MEM_C_PINLOC_4,
         PORT_MEM_C_PINLOC_5 => PORT_MEM_C_PINLOC_5,
         PORT_MEM_C_PINLOC_AUTOGEN_WCNT => PORT_MEM_C_PINLOC_AUTOGEN_WCNT,
         PORT_MEM_CKE_WIDTH => PORT_MEM_CKE_WIDTH,
         PORT_MEM_CKE_PINLOC_0 => PORT_MEM_CKE_PINLOC_0,
         PORT_MEM_CKE_PINLOC_1 => PORT_MEM_CKE_PINLOC_1,
         PORT_MEM_CKE_PINLOC_2 => PORT_MEM_CKE_PINLOC_2,
         PORT_MEM_CKE_PINLOC_3 => PORT_MEM_CKE_PINLOC_3,
         PORT_MEM_CKE_PINLOC_4 => PORT_MEM_CKE_PINLOC_4,
         PORT_MEM_CKE_PINLOC_5 => PORT_MEM_CKE_PINLOC_5,
         PORT_MEM_CKE_PINLOC_AUTOGEN_WCNT => PORT_MEM_CKE_PINLOC_AUTOGEN_WCNT,
         PORT_MEM_CS_N_WIDTH => PORT_MEM_CS_N_WIDTH,
         PORT_MEM_CS_N_PINLOC_0 => PORT_MEM_CS_N_PINLOC_0,
         PORT_MEM_CS_N_PINLOC_1 => PORT_MEM_CS_N_PINLOC_1,
         PORT_MEM_CS_N_PINLOC_2 => PORT_MEM_CS_N_PINLOC_2,
         PORT_MEM_CS_N_PINLOC_3 => PORT_MEM_CS_N_PINLOC_3,
         PORT_MEM_CS_N_PINLOC_4 => PORT_MEM_CS_N_PINLOC_4,
         PORT_MEM_CS_N_PINLOC_5 => PORT_MEM_CS_N_PINLOC_5,
         PORT_MEM_CS_N_PINLOC_AUTOGEN_WCNT => PORT_MEM_CS_N_PINLOC_AUTOGEN_WCNT,
         PORT_MEM_RM_WIDTH => PORT_MEM_RM_WIDTH,
         PORT_MEM_RM_PINLOC_0 => PORT_MEM_RM_PINLOC_0,
         PORT_MEM_RM_PINLOC_1 => PORT_MEM_RM_PINLOC_1,
         PORT_MEM_RM_PINLOC_2 => PORT_MEM_RM_PINLOC_2,
         PORT_MEM_RM_PINLOC_3 => PORT_MEM_RM_PINLOC_3,
         PORT_MEM_RM_PINLOC_4 => PORT_MEM_RM_PINLOC_4,
         PORT_MEM_RM_PINLOC_5 => PORT_MEM_RM_PINLOC_5,
         PORT_MEM_RM_PINLOC_AUTOGEN_WCNT => PORT_MEM_RM_PINLOC_AUTOGEN_WCNT,
         PORT_MEM_ODT_WIDTH => PORT_MEM_ODT_WIDTH,
         PORT_MEM_ODT_PINLOC_0 => PORT_MEM_ODT_PINLOC_0,
         PORT_MEM_ODT_PINLOC_1 => PORT_MEM_ODT_PINLOC_1,
         PORT_MEM_ODT_PINLOC_2 => PORT_MEM_ODT_PINLOC_2,
         PORT_MEM_ODT_PINLOC_3 => PORT_MEM_ODT_PINLOC_3,
         PORT_MEM_ODT_PINLOC_4 => PORT_MEM_ODT_PINLOC_4,
         PORT_MEM_ODT_PINLOC_5 => PORT_MEM_ODT_PINLOC_5,
         PORT_MEM_ODT_PINLOC_AUTOGEN_WCNT => PORT_MEM_ODT_PINLOC_AUTOGEN_WCNT,
         PORT_MEM_REQ_N_WIDTH => PORT_MEM_REQ_N_WIDTH,
         PORT_MEM_REQ_N_PINLOC_0 => PORT_MEM_REQ_N_PINLOC_0,
         PORT_MEM_REQ_N_PINLOC_1 => PORT_MEM_REQ_N_PINLOC_1,
         PORT_MEM_REQ_N_PINLOC_2 => PORT_MEM_REQ_N_PINLOC_2,
         PORT_MEM_REQ_N_PINLOC_3 => PORT_MEM_REQ_N_PINLOC_3,
         PORT_MEM_REQ_N_PINLOC_4 => PORT_MEM_REQ_N_PINLOC_4,
         PORT_MEM_REQ_N_PINLOC_5 => PORT_MEM_REQ_N_PINLOC_5,
         PORT_MEM_REQ_N_PINLOC_AUTOGEN_WCNT => PORT_MEM_REQ_N_PINLOC_AUTOGEN_WCNT,
         PORT_MEM_GNT_N_WIDTH => PORT_MEM_GNT_N_WIDTH,
         PORT_MEM_GNT_N_PINLOC_0 => PORT_MEM_GNT_N_PINLOC_0,
         PORT_MEM_GNT_N_PINLOC_1 => PORT_MEM_GNT_N_PINLOC_1,
         PORT_MEM_GNT_N_PINLOC_2 => PORT_MEM_GNT_N_PINLOC_2,
         PORT_MEM_GNT_N_PINLOC_3 => PORT_MEM_GNT_N_PINLOC_3,
         PORT_MEM_GNT_N_PINLOC_4 => PORT_MEM_GNT_N_PINLOC_4,
         PORT_MEM_GNT_N_PINLOC_5 => PORT_MEM_GNT_N_PINLOC_5,
         PORT_MEM_GNT_N_PINLOC_AUTOGEN_WCNT => PORT_MEM_GNT_N_PINLOC_AUTOGEN_WCNT,
         PORT_MEM_ERR_N_WIDTH => PORT_MEM_ERR_N_WIDTH,
         PORT_MEM_ERR_N_PINLOC_0 => PORT_MEM_ERR_N_PINLOC_0,
         PORT_MEM_ERR_N_PINLOC_1 => PORT_MEM_ERR_N_PINLOC_1,
         PORT_MEM_ERR_N_PINLOC_2 => PORT_MEM_ERR_N_PINLOC_2,
         PORT_MEM_ERR_N_PINLOC_3 => PORT_MEM_ERR_N_PINLOC_3,
         PORT_MEM_ERR_N_PINLOC_4 => PORT_MEM_ERR_N_PINLOC_4,
         PORT_MEM_ERR_N_PINLOC_5 => PORT_MEM_ERR_N_PINLOC_5,
         PORT_MEM_ERR_N_PINLOC_AUTOGEN_WCNT => PORT_MEM_ERR_N_PINLOC_AUTOGEN_WCNT,
         PORT_MEM_RAS_N_WIDTH => PORT_MEM_RAS_N_WIDTH,
         PORT_MEM_RAS_N_PINLOC_0 => PORT_MEM_RAS_N_PINLOC_0,
         PORT_MEM_RAS_N_PINLOC_1 => PORT_MEM_RAS_N_PINLOC_1,
         PORT_MEM_RAS_N_PINLOC_AUTOGEN_WCNT => PORT_MEM_RAS_N_PINLOC_AUTOGEN_WCNT,
         PORT_MEM_CAS_N_WIDTH => PORT_MEM_CAS_N_WIDTH,
         PORT_MEM_CAS_N_PINLOC_0 => PORT_MEM_CAS_N_PINLOC_0,
         PORT_MEM_CAS_N_PINLOC_1 => PORT_MEM_CAS_N_PINLOC_1,
         PORT_MEM_CAS_N_PINLOC_AUTOGEN_WCNT => PORT_MEM_CAS_N_PINLOC_AUTOGEN_WCNT,
         PORT_MEM_WE_N_WIDTH => PORT_MEM_WE_N_WIDTH,
         PORT_MEM_WE_N_PINLOC_0 => PORT_MEM_WE_N_PINLOC_0,
         PORT_MEM_WE_N_PINLOC_1 => PORT_MEM_WE_N_PINLOC_1,
         PORT_MEM_WE_N_PINLOC_AUTOGEN_WCNT => PORT_MEM_WE_N_PINLOC_AUTOGEN_WCNT,
         PORT_MEM_RESET_N_WIDTH => PORT_MEM_RESET_N_WIDTH,
         PORT_MEM_RESET_N_PINLOC_0 => PORT_MEM_RESET_N_PINLOC_0,
         PORT_MEM_RESET_N_PINLOC_1 => PORT_MEM_RESET_N_PINLOC_1,
         PORT_MEM_RESET_N_PINLOC_AUTOGEN_WCNT => PORT_MEM_RESET_N_PINLOC_AUTOGEN_WCNT,
         PORT_MEM_ACT_N_WIDTH => PORT_MEM_ACT_N_WIDTH,
         PORT_MEM_ACT_N_PINLOC_0 => PORT_MEM_ACT_N_PINLOC_0,
         PORT_MEM_ACT_N_PINLOC_1 => PORT_MEM_ACT_N_PINLOC_1,
         PORT_MEM_ACT_N_PINLOC_AUTOGEN_WCNT => PORT_MEM_ACT_N_PINLOC_AUTOGEN_WCNT,
         PORT_MEM_PAR_WIDTH => PORT_MEM_PAR_WIDTH,
         PORT_MEM_PAR_PINLOC_0 => PORT_MEM_PAR_PINLOC_0,
         PORT_MEM_PAR_PINLOC_1 => PORT_MEM_PAR_PINLOC_1,
         PORT_MEM_PAR_PINLOC_AUTOGEN_WCNT => PORT_MEM_PAR_PINLOC_AUTOGEN_WCNT,
         PORT_MEM_CA_WIDTH => PORT_MEM_CA_WIDTH,
         PORT_MEM_CA_PINLOC_0 => PORT_MEM_CA_PINLOC_0,
         PORT_MEM_CA_PINLOC_1 => PORT_MEM_CA_PINLOC_1,
         PORT_MEM_CA_PINLOC_2 => PORT_MEM_CA_PINLOC_2,
         PORT_MEM_CA_PINLOC_3 => PORT_MEM_CA_PINLOC_3,
         PORT_MEM_CA_PINLOC_4 => PORT_MEM_CA_PINLOC_4,
         PORT_MEM_CA_PINLOC_5 => PORT_MEM_CA_PINLOC_5,
         PORT_MEM_CA_PINLOC_6 => PORT_MEM_CA_PINLOC_6,
         PORT_MEM_CA_PINLOC_7 => PORT_MEM_CA_PINLOC_7,
         PORT_MEM_CA_PINLOC_8 => PORT_MEM_CA_PINLOC_8,
         PORT_MEM_CA_PINLOC_9 => PORT_MEM_CA_PINLOC_9,
         PORT_MEM_CA_PINLOC_10 => PORT_MEM_CA_PINLOC_10,
         PORT_MEM_CA_PINLOC_11 => PORT_MEM_CA_PINLOC_11,
         PORT_MEM_CA_PINLOC_12 => PORT_MEM_CA_PINLOC_12,
         PORT_MEM_CA_PINLOC_13 => PORT_MEM_CA_PINLOC_13,
         PORT_MEM_CA_PINLOC_14 => PORT_MEM_CA_PINLOC_14,
         PORT_MEM_CA_PINLOC_15 => PORT_MEM_CA_PINLOC_15,
         PORT_MEM_CA_PINLOC_16 => PORT_MEM_CA_PINLOC_16,
         PORT_MEM_CA_PINLOC_AUTOGEN_WCNT => PORT_MEM_CA_PINLOC_AUTOGEN_WCNT,
         PORT_MEM_REF_N_WIDTH => PORT_MEM_REF_N_WIDTH,
         PORT_MEM_REF_N_PINLOC_0 => PORT_MEM_REF_N_PINLOC_0,
         PORT_MEM_REF_N_PINLOC_AUTOGEN_WCNT => PORT_MEM_REF_N_PINLOC_AUTOGEN_WCNT,
         PORT_MEM_WPS_N_WIDTH => PORT_MEM_WPS_N_WIDTH,
         PORT_MEM_WPS_N_PINLOC_0 => PORT_MEM_WPS_N_PINLOC_0,
         PORT_MEM_WPS_N_PINLOC_AUTOGEN_WCNT => PORT_MEM_WPS_N_PINLOC_AUTOGEN_WCNT,
         PORT_MEM_RPS_N_WIDTH => PORT_MEM_RPS_N_WIDTH,
         PORT_MEM_RPS_N_PINLOC_0 => PORT_MEM_RPS_N_PINLOC_0,
         PORT_MEM_RPS_N_PINLOC_AUTOGEN_WCNT => PORT_MEM_RPS_N_PINLOC_AUTOGEN_WCNT,
         PORT_MEM_DOFF_N_WIDTH => PORT_MEM_DOFF_N_WIDTH,
         PORT_MEM_DOFF_N_PINLOC_0 => PORT_MEM_DOFF_N_PINLOC_0,
         PORT_MEM_DOFF_N_PINLOC_AUTOGEN_WCNT => PORT_MEM_DOFF_N_PINLOC_AUTOGEN_WCNT,
         PORT_MEM_LDA_N_WIDTH => PORT_MEM_LDA_N_WIDTH,
         PORT_MEM_LDA_N_PINLOC_0 => PORT_MEM_LDA_N_PINLOC_0,
         PORT_MEM_LDA_N_PINLOC_AUTOGEN_WCNT => PORT_MEM_LDA_N_PINLOC_AUTOGEN_WCNT,
         PORT_MEM_LDB_N_WIDTH => PORT_MEM_LDB_N_WIDTH,
         PORT_MEM_LDB_N_PINLOC_0 => PORT_MEM_LDB_N_PINLOC_0,
         PORT_MEM_LDB_N_PINLOC_AUTOGEN_WCNT => PORT_MEM_LDB_N_PINLOC_AUTOGEN_WCNT,
         PORT_MEM_RWA_N_WIDTH => PORT_MEM_RWA_N_WIDTH,
         PORT_MEM_RWA_N_PINLOC_0 => PORT_MEM_RWA_N_PINLOC_0,
         PORT_MEM_RWA_N_PINLOC_AUTOGEN_WCNT => PORT_MEM_RWA_N_PINLOC_AUTOGEN_WCNT,
         PORT_MEM_RWB_N_WIDTH => PORT_MEM_RWB_N_WIDTH,
         PORT_MEM_RWB_N_PINLOC_0 => PORT_MEM_RWB_N_PINLOC_0,
         PORT_MEM_RWB_N_PINLOC_AUTOGEN_WCNT => PORT_MEM_RWB_N_PINLOC_AUTOGEN_WCNT,
         PORT_MEM_LBK0_N_WIDTH => PORT_MEM_LBK0_N_WIDTH,
         PORT_MEM_LBK0_N_PINLOC_0 => PORT_MEM_LBK0_N_PINLOC_0,
         PORT_MEM_LBK0_N_PINLOC_AUTOGEN_WCNT => PORT_MEM_LBK0_N_PINLOC_AUTOGEN_WCNT,
         PORT_MEM_LBK1_N_WIDTH => PORT_MEM_LBK1_N_WIDTH,
         PORT_MEM_LBK1_N_PINLOC_0 => PORT_MEM_LBK1_N_PINLOC_0,
         PORT_MEM_LBK1_N_PINLOC_AUTOGEN_WCNT => PORT_MEM_LBK1_N_PINLOC_AUTOGEN_WCNT,
         PORT_MEM_CFG_N_WIDTH => PORT_MEM_CFG_N_WIDTH,
         PORT_MEM_CFG_N_PINLOC_0 => PORT_MEM_CFG_N_PINLOC_0,
         PORT_MEM_CFG_N_PINLOC_AUTOGEN_WCNT => PORT_MEM_CFG_N_PINLOC_AUTOGEN_WCNT,
         PORT_MEM_AP_WIDTH => PORT_MEM_AP_WIDTH,
         PORT_MEM_AP_PINLOC_0 => PORT_MEM_AP_PINLOC_0,
         PORT_MEM_AP_PINLOC_AUTOGEN_WCNT => PORT_MEM_AP_PINLOC_AUTOGEN_WCNT,
         PORT_MEM_AINV_WIDTH => PORT_MEM_AINV_WIDTH,
         PORT_MEM_AINV_PINLOC_0 => PORT_MEM_AINV_PINLOC_0,
         PORT_MEM_AINV_PINLOC_AUTOGEN_WCNT => PORT_MEM_AINV_PINLOC_AUTOGEN_WCNT,
         PORT_MEM_DM_WIDTH => PORT_MEM_DM_WIDTH,
         PORT_MEM_DM_PINLOC_0 => PORT_MEM_DM_PINLOC_0,
         PORT_MEM_DM_PINLOC_1 => PORT_MEM_DM_PINLOC_1,
         PORT_MEM_DM_PINLOC_2 => PORT_MEM_DM_PINLOC_2,
         PORT_MEM_DM_PINLOC_3 => PORT_MEM_DM_PINLOC_3,
         PORT_MEM_DM_PINLOC_4 => PORT_MEM_DM_PINLOC_4,
         PORT_MEM_DM_PINLOC_5 => PORT_MEM_DM_PINLOC_5,
         PORT_MEM_DM_PINLOC_6 => PORT_MEM_DM_PINLOC_6,
         PORT_MEM_DM_PINLOC_7 => PORT_MEM_DM_PINLOC_7,
         PORT_MEM_DM_PINLOC_8 => PORT_MEM_DM_PINLOC_8,
         PORT_MEM_DM_PINLOC_9 => PORT_MEM_DM_PINLOC_9,
         PORT_MEM_DM_PINLOC_10 => PORT_MEM_DM_PINLOC_10,
         PORT_MEM_DM_PINLOC_11 => PORT_MEM_DM_PINLOC_11,
         PORT_MEM_DM_PINLOC_12 => PORT_MEM_DM_PINLOC_12,
         PORT_MEM_DM_PINLOC_AUTOGEN_WCNT => PORT_MEM_DM_PINLOC_AUTOGEN_WCNT,
         PORT_MEM_BWS_N_WIDTH => PORT_MEM_BWS_N_WIDTH,
         PORT_MEM_BWS_N_PINLOC_0 => PORT_MEM_BWS_N_PINLOC_0,
         PORT_MEM_BWS_N_PINLOC_1 => PORT_MEM_BWS_N_PINLOC_1,
         PORT_MEM_BWS_N_PINLOC_2 => PORT_MEM_BWS_N_PINLOC_2,
         PORT_MEM_BWS_N_PINLOC_AUTOGEN_WCNT => PORT_MEM_BWS_N_PINLOC_AUTOGEN_WCNT,
         PORT_MEM_D_WIDTH => PORT_MEM_D_WIDTH,
         PORT_MEM_D_PINLOC_0 => PORT_MEM_D_PINLOC_0,
         PORT_MEM_D_PINLOC_1 => PORT_MEM_D_PINLOC_1,
         PORT_MEM_D_PINLOC_2 => PORT_MEM_D_PINLOC_2,
         PORT_MEM_D_PINLOC_3 => PORT_MEM_D_PINLOC_3,
         PORT_MEM_D_PINLOC_4 => PORT_MEM_D_PINLOC_4,
         PORT_MEM_D_PINLOC_5 => PORT_MEM_D_PINLOC_5,
         PORT_MEM_D_PINLOC_6 => PORT_MEM_D_PINLOC_6,
         PORT_MEM_D_PINLOC_7 => PORT_MEM_D_PINLOC_7,
         PORT_MEM_D_PINLOC_8 => PORT_MEM_D_PINLOC_8,
         PORT_MEM_D_PINLOC_9 => PORT_MEM_D_PINLOC_9,
         PORT_MEM_D_PINLOC_10 => PORT_MEM_D_PINLOC_10,
         PORT_MEM_D_PINLOC_11 => PORT_MEM_D_PINLOC_11,
         PORT_MEM_D_PINLOC_12 => PORT_MEM_D_PINLOC_12,
         PORT_MEM_D_PINLOC_13 => PORT_MEM_D_PINLOC_13,
         PORT_MEM_D_PINLOC_14 => PORT_MEM_D_PINLOC_14,
         PORT_MEM_D_PINLOC_15 => PORT_MEM_D_PINLOC_15,
         PORT_MEM_D_PINLOC_16 => PORT_MEM_D_PINLOC_16,
         PORT_MEM_D_PINLOC_17 => PORT_MEM_D_PINLOC_17,
         PORT_MEM_D_PINLOC_18 => PORT_MEM_D_PINLOC_18,
         PORT_MEM_D_PINLOC_19 => PORT_MEM_D_PINLOC_19,
         PORT_MEM_D_PINLOC_20 => PORT_MEM_D_PINLOC_20,
         PORT_MEM_D_PINLOC_21 => PORT_MEM_D_PINLOC_21,
         PORT_MEM_D_PINLOC_22 => PORT_MEM_D_PINLOC_22,
         PORT_MEM_D_PINLOC_23 => PORT_MEM_D_PINLOC_23,
         PORT_MEM_D_PINLOC_24 => PORT_MEM_D_PINLOC_24,
         PORT_MEM_D_PINLOC_25 => PORT_MEM_D_PINLOC_25,
         PORT_MEM_D_PINLOC_26 => PORT_MEM_D_PINLOC_26,
         PORT_MEM_D_PINLOC_27 => PORT_MEM_D_PINLOC_27,
         PORT_MEM_D_PINLOC_28 => PORT_MEM_D_PINLOC_28,
         PORT_MEM_D_PINLOC_29 => PORT_MEM_D_PINLOC_29,
         PORT_MEM_D_PINLOC_30 => PORT_MEM_D_PINLOC_30,
         PORT_MEM_D_PINLOC_31 => PORT_MEM_D_PINLOC_31,
         PORT_MEM_D_PINLOC_32 => PORT_MEM_D_PINLOC_32,
         PORT_MEM_D_PINLOC_33 => PORT_MEM_D_PINLOC_33,
         PORT_MEM_D_PINLOC_34 => PORT_MEM_D_PINLOC_34,
         PORT_MEM_D_PINLOC_35 => PORT_MEM_D_PINLOC_35,
         PORT_MEM_D_PINLOC_36 => PORT_MEM_D_PINLOC_36,
         PORT_MEM_D_PINLOC_37 => PORT_MEM_D_PINLOC_37,
         PORT_MEM_D_PINLOC_38 => PORT_MEM_D_PINLOC_38,
         PORT_MEM_D_PINLOC_39 => PORT_MEM_D_PINLOC_39,
         PORT_MEM_D_PINLOC_40 => PORT_MEM_D_PINLOC_40,
         PORT_MEM_D_PINLOC_41 => PORT_MEM_D_PINLOC_41,
         PORT_MEM_D_PINLOC_42 => PORT_MEM_D_PINLOC_42,
         PORT_MEM_D_PINLOC_43 => PORT_MEM_D_PINLOC_43,
         PORT_MEM_D_PINLOC_44 => PORT_MEM_D_PINLOC_44,
         PORT_MEM_D_PINLOC_45 => PORT_MEM_D_PINLOC_45,
         PORT_MEM_D_PINLOC_46 => PORT_MEM_D_PINLOC_46,
         PORT_MEM_D_PINLOC_47 => PORT_MEM_D_PINLOC_47,
         PORT_MEM_D_PINLOC_48 => PORT_MEM_D_PINLOC_48,
         PORT_MEM_D_PINLOC_AUTOGEN_WCNT => PORT_MEM_D_PINLOC_AUTOGEN_WCNT,
         PORT_MEM_DQ_WIDTH => PORT_MEM_DQ_WIDTH,
         PORT_MEM_DQ_PINLOC_0 => PORT_MEM_DQ_PINLOC_0,
         PORT_MEM_DQ_PINLOC_1 => PORT_MEM_DQ_PINLOC_1,
         PORT_MEM_DQ_PINLOC_2 => PORT_MEM_DQ_PINLOC_2,
         PORT_MEM_DQ_PINLOC_3 => PORT_MEM_DQ_PINLOC_3,
         PORT_MEM_DQ_PINLOC_4 => PORT_MEM_DQ_PINLOC_4,
         PORT_MEM_DQ_PINLOC_5 => PORT_MEM_DQ_PINLOC_5,
         PORT_MEM_DQ_PINLOC_6 => PORT_MEM_DQ_PINLOC_6,
         PORT_MEM_DQ_PINLOC_7 => PORT_MEM_DQ_PINLOC_7,
         PORT_MEM_DQ_PINLOC_8 => PORT_MEM_DQ_PINLOC_8,
         PORT_MEM_DQ_PINLOC_9 => PORT_MEM_DQ_PINLOC_9,
         PORT_MEM_DQ_PINLOC_10 => PORT_MEM_DQ_PINLOC_10,
         PORT_MEM_DQ_PINLOC_11 => PORT_MEM_DQ_PINLOC_11,
         PORT_MEM_DQ_PINLOC_12 => PORT_MEM_DQ_PINLOC_12,
         PORT_MEM_DQ_PINLOC_13 => PORT_MEM_DQ_PINLOC_13,
         PORT_MEM_DQ_PINLOC_14 => PORT_MEM_DQ_PINLOC_14,
         PORT_MEM_DQ_PINLOC_15 => PORT_MEM_DQ_PINLOC_15,
         PORT_MEM_DQ_PINLOC_16 => PORT_MEM_DQ_PINLOC_16,
         PORT_MEM_DQ_PINLOC_17 => PORT_MEM_DQ_PINLOC_17,
         PORT_MEM_DQ_PINLOC_18 => PORT_MEM_DQ_PINLOC_18,
         PORT_MEM_DQ_PINLOC_19 => PORT_MEM_DQ_PINLOC_19,
         PORT_MEM_DQ_PINLOC_20 => PORT_MEM_DQ_PINLOC_20,
         PORT_MEM_DQ_PINLOC_21 => PORT_MEM_DQ_PINLOC_21,
         PORT_MEM_DQ_PINLOC_22 => PORT_MEM_DQ_PINLOC_22,
         PORT_MEM_DQ_PINLOC_23 => PORT_MEM_DQ_PINLOC_23,
         PORT_MEM_DQ_PINLOC_24 => PORT_MEM_DQ_PINLOC_24,
         PORT_MEM_DQ_PINLOC_25 => PORT_MEM_DQ_PINLOC_25,
         PORT_MEM_DQ_PINLOC_26 => PORT_MEM_DQ_PINLOC_26,
         PORT_MEM_DQ_PINLOC_27 => PORT_MEM_DQ_PINLOC_27,
         PORT_MEM_DQ_PINLOC_28 => PORT_MEM_DQ_PINLOC_28,
         PORT_MEM_DQ_PINLOC_29 => PORT_MEM_DQ_PINLOC_29,
         PORT_MEM_DQ_PINLOC_30 => PORT_MEM_DQ_PINLOC_30,
         PORT_MEM_DQ_PINLOC_31 => PORT_MEM_DQ_PINLOC_31,
         PORT_MEM_DQ_PINLOC_32 => PORT_MEM_DQ_PINLOC_32,
         PORT_MEM_DQ_PINLOC_33 => PORT_MEM_DQ_PINLOC_33,
         PORT_MEM_DQ_PINLOC_34 => PORT_MEM_DQ_PINLOC_34,
         PORT_MEM_DQ_PINLOC_35 => PORT_MEM_DQ_PINLOC_35,
         PORT_MEM_DQ_PINLOC_36 => PORT_MEM_DQ_PINLOC_36,
         PORT_MEM_DQ_PINLOC_37 => PORT_MEM_DQ_PINLOC_37,
         PORT_MEM_DQ_PINLOC_38 => PORT_MEM_DQ_PINLOC_38,
         PORT_MEM_DQ_PINLOC_39 => PORT_MEM_DQ_PINLOC_39,
         PORT_MEM_DQ_PINLOC_40 => PORT_MEM_DQ_PINLOC_40,
         PORT_MEM_DQ_PINLOC_41 => PORT_MEM_DQ_PINLOC_41,
         PORT_MEM_DQ_PINLOC_42 => PORT_MEM_DQ_PINLOC_42,
         PORT_MEM_DQ_PINLOC_43 => PORT_MEM_DQ_PINLOC_43,
         PORT_MEM_DQ_PINLOC_44 => PORT_MEM_DQ_PINLOC_44,
         PORT_MEM_DQ_PINLOC_45 => PORT_MEM_DQ_PINLOC_45,
         PORT_MEM_DQ_PINLOC_46 => PORT_MEM_DQ_PINLOC_46,
         PORT_MEM_DQ_PINLOC_47 => PORT_MEM_DQ_PINLOC_47,
         PORT_MEM_DQ_PINLOC_48 => PORT_MEM_DQ_PINLOC_48,
         PORT_MEM_DQ_PINLOC_AUTOGEN_WCNT => PORT_MEM_DQ_PINLOC_AUTOGEN_WCNT,
         PORT_MEM_DBI_N_WIDTH => PORT_MEM_DBI_N_WIDTH,
         PORT_MEM_DBI_N_PINLOC_0 => PORT_MEM_DBI_N_PINLOC_0,
         PORT_MEM_DBI_N_PINLOC_1 => PORT_MEM_DBI_N_PINLOC_1,
         PORT_MEM_DBI_N_PINLOC_2 => PORT_MEM_DBI_N_PINLOC_2,
         PORT_MEM_DBI_N_PINLOC_3 => PORT_MEM_DBI_N_PINLOC_3,
         PORT_MEM_DBI_N_PINLOC_4 => PORT_MEM_DBI_N_PINLOC_4,
         PORT_MEM_DBI_N_PINLOC_5 => PORT_MEM_DBI_N_PINLOC_5,
         PORT_MEM_DBI_N_PINLOC_6 => PORT_MEM_DBI_N_PINLOC_6,
         PORT_MEM_DBI_N_PINLOC_AUTOGEN_WCNT => PORT_MEM_DBI_N_PINLOC_AUTOGEN_WCNT,
         PORT_MEM_DQA_WIDTH => PORT_MEM_DQA_WIDTH,
         PORT_MEM_DQA_PINLOC_0 => PORT_MEM_DQA_PINLOC_0,
         PORT_MEM_DQA_PINLOC_1 => PORT_MEM_DQA_PINLOC_1,
         PORT_MEM_DQA_PINLOC_2 => PORT_MEM_DQA_PINLOC_2,
         PORT_MEM_DQA_PINLOC_3 => PORT_MEM_DQA_PINLOC_3,
         PORT_MEM_DQA_PINLOC_4 => PORT_MEM_DQA_PINLOC_4,
         PORT_MEM_DQA_PINLOC_5 => PORT_MEM_DQA_PINLOC_5,
         PORT_MEM_DQA_PINLOC_6 => PORT_MEM_DQA_PINLOC_6,
         PORT_MEM_DQA_PINLOC_7 => PORT_MEM_DQA_PINLOC_7,
         PORT_MEM_DQA_PINLOC_8 => PORT_MEM_DQA_PINLOC_8,
         PORT_MEM_DQA_PINLOC_9 => PORT_MEM_DQA_PINLOC_9,
         PORT_MEM_DQA_PINLOC_10 => PORT_MEM_DQA_PINLOC_10,
         PORT_MEM_DQA_PINLOC_11 => PORT_MEM_DQA_PINLOC_11,
         PORT_MEM_DQA_PINLOC_12 => PORT_MEM_DQA_PINLOC_12,
         PORT_MEM_DQA_PINLOC_13 => PORT_MEM_DQA_PINLOC_13,
         PORT_MEM_DQA_PINLOC_14 => PORT_MEM_DQA_PINLOC_14,
         PORT_MEM_DQA_PINLOC_15 => PORT_MEM_DQA_PINLOC_15,
         PORT_MEM_DQA_PINLOC_16 => PORT_MEM_DQA_PINLOC_16,
         PORT_MEM_DQA_PINLOC_17 => PORT_MEM_DQA_PINLOC_17,
         PORT_MEM_DQA_PINLOC_18 => PORT_MEM_DQA_PINLOC_18,
         PORT_MEM_DQA_PINLOC_19 => PORT_MEM_DQA_PINLOC_19,
         PORT_MEM_DQA_PINLOC_20 => PORT_MEM_DQA_PINLOC_20,
         PORT_MEM_DQA_PINLOC_21 => PORT_MEM_DQA_PINLOC_21,
         PORT_MEM_DQA_PINLOC_22 => PORT_MEM_DQA_PINLOC_22,
         PORT_MEM_DQA_PINLOC_23 => PORT_MEM_DQA_PINLOC_23,
         PORT_MEM_DQA_PINLOC_24 => PORT_MEM_DQA_PINLOC_24,
         PORT_MEM_DQA_PINLOC_25 => PORT_MEM_DQA_PINLOC_25,
         PORT_MEM_DQA_PINLOC_26 => PORT_MEM_DQA_PINLOC_26,
         PORT_MEM_DQA_PINLOC_27 => PORT_MEM_DQA_PINLOC_27,
         PORT_MEM_DQA_PINLOC_28 => PORT_MEM_DQA_PINLOC_28,
         PORT_MEM_DQA_PINLOC_29 => PORT_MEM_DQA_PINLOC_29,
         PORT_MEM_DQA_PINLOC_30 => PORT_MEM_DQA_PINLOC_30,
         PORT_MEM_DQA_PINLOC_31 => PORT_MEM_DQA_PINLOC_31,
         PORT_MEM_DQA_PINLOC_32 => PORT_MEM_DQA_PINLOC_32,
         PORT_MEM_DQA_PINLOC_33 => PORT_MEM_DQA_PINLOC_33,
         PORT_MEM_DQA_PINLOC_34 => PORT_MEM_DQA_PINLOC_34,
         PORT_MEM_DQA_PINLOC_35 => PORT_MEM_DQA_PINLOC_35,
         PORT_MEM_DQA_PINLOC_36 => PORT_MEM_DQA_PINLOC_36,
         PORT_MEM_DQA_PINLOC_37 => PORT_MEM_DQA_PINLOC_37,
         PORT_MEM_DQA_PINLOC_38 => PORT_MEM_DQA_PINLOC_38,
         PORT_MEM_DQA_PINLOC_39 => PORT_MEM_DQA_PINLOC_39,
         PORT_MEM_DQA_PINLOC_40 => PORT_MEM_DQA_PINLOC_40,
         PORT_MEM_DQA_PINLOC_41 => PORT_MEM_DQA_PINLOC_41,
         PORT_MEM_DQA_PINLOC_42 => PORT_MEM_DQA_PINLOC_42,
         PORT_MEM_DQA_PINLOC_43 => PORT_MEM_DQA_PINLOC_43,
         PORT_MEM_DQA_PINLOC_44 => PORT_MEM_DQA_PINLOC_44,
         PORT_MEM_DQA_PINLOC_45 => PORT_MEM_DQA_PINLOC_45,
         PORT_MEM_DQA_PINLOC_46 => PORT_MEM_DQA_PINLOC_46,
         PORT_MEM_DQA_PINLOC_47 => PORT_MEM_DQA_PINLOC_47,
         PORT_MEM_DQA_PINLOC_48 => PORT_MEM_DQA_PINLOC_48,
         PORT_MEM_DQA_PINLOC_AUTOGEN_WCNT => PORT_MEM_DQA_PINLOC_AUTOGEN_WCNT,
         PORT_MEM_DQB_WIDTH => PORT_MEM_DQB_WIDTH,
         PORT_MEM_DQB_PINLOC_0 => PORT_MEM_DQB_PINLOC_0,
         PORT_MEM_DQB_PINLOC_1 => PORT_MEM_DQB_PINLOC_1,
         PORT_MEM_DQB_PINLOC_2 => PORT_MEM_DQB_PINLOC_2,
         PORT_MEM_DQB_PINLOC_3 => PORT_MEM_DQB_PINLOC_3,
         PORT_MEM_DQB_PINLOC_4 => PORT_MEM_DQB_PINLOC_4,
         PORT_MEM_DQB_PINLOC_5 => PORT_MEM_DQB_PINLOC_5,
         PORT_MEM_DQB_PINLOC_6 => PORT_MEM_DQB_PINLOC_6,
         PORT_MEM_DQB_PINLOC_7 => PORT_MEM_DQB_PINLOC_7,
         PORT_MEM_DQB_PINLOC_8 => PORT_MEM_DQB_PINLOC_8,
         PORT_MEM_DQB_PINLOC_9 => PORT_MEM_DQB_PINLOC_9,
         PORT_MEM_DQB_PINLOC_10 => PORT_MEM_DQB_PINLOC_10,
         PORT_MEM_DQB_PINLOC_11 => PORT_MEM_DQB_PINLOC_11,
         PORT_MEM_DQB_PINLOC_12 => PORT_MEM_DQB_PINLOC_12,
         PORT_MEM_DQB_PINLOC_13 => PORT_MEM_DQB_PINLOC_13,
         PORT_MEM_DQB_PINLOC_14 => PORT_MEM_DQB_PINLOC_14,
         PORT_MEM_DQB_PINLOC_15 => PORT_MEM_DQB_PINLOC_15,
         PORT_MEM_DQB_PINLOC_16 => PORT_MEM_DQB_PINLOC_16,
         PORT_MEM_DQB_PINLOC_17 => PORT_MEM_DQB_PINLOC_17,
         PORT_MEM_DQB_PINLOC_18 => PORT_MEM_DQB_PINLOC_18,
         PORT_MEM_DQB_PINLOC_19 => PORT_MEM_DQB_PINLOC_19,
         PORT_MEM_DQB_PINLOC_20 => PORT_MEM_DQB_PINLOC_20,
         PORT_MEM_DQB_PINLOC_21 => PORT_MEM_DQB_PINLOC_21,
         PORT_MEM_DQB_PINLOC_22 => PORT_MEM_DQB_PINLOC_22,
         PORT_MEM_DQB_PINLOC_23 => PORT_MEM_DQB_PINLOC_23,
         PORT_MEM_DQB_PINLOC_24 => PORT_MEM_DQB_PINLOC_24,
         PORT_MEM_DQB_PINLOC_25 => PORT_MEM_DQB_PINLOC_25,
         PORT_MEM_DQB_PINLOC_26 => PORT_MEM_DQB_PINLOC_26,
         PORT_MEM_DQB_PINLOC_27 => PORT_MEM_DQB_PINLOC_27,
         PORT_MEM_DQB_PINLOC_28 => PORT_MEM_DQB_PINLOC_28,
         PORT_MEM_DQB_PINLOC_29 => PORT_MEM_DQB_PINLOC_29,
         PORT_MEM_DQB_PINLOC_30 => PORT_MEM_DQB_PINLOC_30,
         PORT_MEM_DQB_PINLOC_31 => PORT_MEM_DQB_PINLOC_31,
         PORT_MEM_DQB_PINLOC_32 => PORT_MEM_DQB_PINLOC_32,
         PORT_MEM_DQB_PINLOC_33 => PORT_MEM_DQB_PINLOC_33,
         PORT_MEM_DQB_PINLOC_34 => PORT_MEM_DQB_PINLOC_34,
         PORT_MEM_DQB_PINLOC_35 => PORT_MEM_DQB_PINLOC_35,
         PORT_MEM_DQB_PINLOC_36 => PORT_MEM_DQB_PINLOC_36,
         PORT_MEM_DQB_PINLOC_37 => PORT_MEM_DQB_PINLOC_37,
         PORT_MEM_DQB_PINLOC_38 => PORT_MEM_DQB_PINLOC_38,
         PORT_MEM_DQB_PINLOC_39 => PORT_MEM_DQB_PINLOC_39,
         PORT_MEM_DQB_PINLOC_40 => PORT_MEM_DQB_PINLOC_40,
         PORT_MEM_DQB_PINLOC_41 => PORT_MEM_DQB_PINLOC_41,
         PORT_MEM_DQB_PINLOC_42 => PORT_MEM_DQB_PINLOC_42,
         PORT_MEM_DQB_PINLOC_43 => PORT_MEM_DQB_PINLOC_43,
         PORT_MEM_DQB_PINLOC_44 => PORT_MEM_DQB_PINLOC_44,
         PORT_MEM_DQB_PINLOC_45 => PORT_MEM_DQB_PINLOC_45,
         PORT_MEM_DQB_PINLOC_46 => PORT_MEM_DQB_PINLOC_46,
         PORT_MEM_DQB_PINLOC_47 => PORT_MEM_DQB_PINLOC_47,
         PORT_MEM_DQB_PINLOC_48 => PORT_MEM_DQB_PINLOC_48,
         PORT_MEM_DQB_PINLOC_AUTOGEN_WCNT => PORT_MEM_DQB_PINLOC_AUTOGEN_WCNT,
         PORT_MEM_DINVA_WIDTH => PORT_MEM_DINVA_WIDTH,
         PORT_MEM_DINVA_PINLOC_0 => PORT_MEM_DINVA_PINLOC_0,
         PORT_MEM_DINVA_PINLOC_1 => PORT_MEM_DINVA_PINLOC_1,
         PORT_MEM_DINVA_PINLOC_2 => PORT_MEM_DINVA_PINLOC_2,
         PORT_MEM_DINVA_PINLOC_AUTOGEN_WCNT => PORT_MEM_DINVA_PINLOC_AUTOGEN_WCNT,
         PORT_MEM_DINVB_WIDTH => PORT_MEM_DINVB_WIDTH,
         PORT_MEM_DINVB_PINLOC_0 => PORT_MEM_DINVB_PINLOC_0,
         PORT_MEM_DINVB_PINLOC_1 => PORT_MEM_DINVB_PINLOC_1,
         PORT_MEM_DINVB_PINLOC_2 => PORT_MEM_DINVB_PINLOC_2,
         PORT_MEM_DINVB_PINLOC_AUTOGEN_WCNT => PORT_MEM_DINVB_PINLOC_AUTOGEN_WCNT,
         PORT_MEM_Q_WIDTH => PORT_MEM_Q_WIDTH,
         PORT_MEM_Q_PINLOC_0 => PORT_MEM_Q_PINLOC_0,
         PORT_MEM_Q_PINLOC_1 => PORT_MEM_Q_PINLOC_1,
         PORT_MEM_Q_PINLOC_2 => PORT_MEM_Q_PINLOC_2,
         PORT_MEM_Q_PINLOC_3 => PORT_MEM_Q_PINLOC_3,
         PORT_MEM_Q_PINLOC_4 => PORT_MEM_Q_PINLOC_4,
         PORT_MEM_Q_PINLOC_5 => PORT_MEM_Q_PINLOC_5,
         PORT_MEM_Q_PINLOC_6 => PORT_MEM_Q_PINLOC_6,
         PORT_MEM_Q_PINLOC_7 => PORT_MEM_Q_PINLOC_7,
         PORT_MEM_Q_PINLOC_8 => PORT_MEM_Q_PINLOC_8,
         PORT_MEM_Q_PINLOC_9 => PORT_MEM_Q_PINLOC_9,
         PORT_MEM_Q_PINLOC_10 => PORT_MEM_Q_PINLOC_10,
         PORT_MEM_Q_PINLOC_11 => PORT_MEM_Q_PINLOC_11,
         PORT_MEM_Q_PINLOC_12 => PORT_MEM_Q_PINLOC_12,
         PORT_MEM_Q_PINLOC_13 => PORT_MEM_Q_PINLOC_13,
         PORT_MEM_Q_PINLOC_14 => PORT_MEM_Q_PINLOC_14,
         PORT_MEM_Q_PINLOC_15 => PORT_MEM_Q_PINLOC_15,
         PORT_MEM_Q_PINLOC_16 => PORT_MEM_Q_PINLOC_16,
         PORT_MEM_Q_PINLOC_17 => PORT_MEM_Q_PINLOC_17,
         PORT_MEM_Q_PINLOC_18 => PORT_MEM_Q_PINLOC_18,
         PORT_MEM_Q_PINLOC_19 => PORT_MEM_Q_PINLOC_19,
         PORT_MEM_Q_PINLOC_20 => PORT_MEM_Q_PINLOC_20,
         PORT_MEM_Q_PINLOC_21 => PORT_MEM_Q_PINLOC_21,
         PORT_MEM_Q_PINLOC_22 => PORT_MEM_Q_PINLOC_22,
         PORT_MEM_Q_PINLOC_23 => PORT_MEM_Q_PINLOC_23,
         PORT_MEM_Q_PINLOC_24 => PORT_MEM_Q_PINLOC_24,
         PORT_MEM_Q_PINLOC_25 => PORT_MEM_Q_PINLOC_25,
         PORT_MEM_Q_PINLOC_26 => PORT_MEM_Q_PINLOC_26,
         PORT_MEM_Q_PINLOC_27 => PORT_MEM_Q_PINLOC_27,
         PORT_MEM_Q_PINLOC_28 => PORT_MEM_Q_PINLOC_28,
         PORT_MEM_Q_PINLOC_29 => PORT_MEM_Q_PINLOC_29,
         PORT_MEM_Q_PINLOC_30 => PORT_MEM_Q_PINLOC_30,
         PORT_MEM_Q_PINLOC_31 => PORT_MEM_Q_PINLOC_31,
         PORT_MEM_Q_PINLOC_32 => PORT_MEM_Q_PINLOC_32,
         PORT_MEM_Q_PINLOC_33 => PORT_MEM_Q_PINLOC_33,
         PORT_MEM_Q_PINLOC_34 => PORT_MEM_Q_PINLOC_34,
         PORT_MEM_Q_PINLOC_35 => PORT_MEM_Q_PINLOC_35,
         PORT_MEM_Q_PINLOC_36 => PORT_MEM_Q_PINLOC_36,
         PORT_MEM_Q_PINLOC_37 => PORT_MEM_Q_PINLOC_37,
         PORT_MEM_Q_PINLOC_38 => PORT_MEM_Q_PINLOC_38,
         PORT_MEM_Q_PINLOC_39 => PORT_MEM_Q_PINLOC_39,
         PORT_MEM_Q_PINLOC_40 => PORT_MEM_Q_PINLOC_40,
         PORT_MEM_Q_PINLOC_41 => PORT_MEM_Q_PINLOC_41,
         PORT_MEM_Q_PINLOC_42 => PORT_MEM_Q_PINLOC_42,
         PORT_MEM_Q_PINLOC_43 => PORT_MEM_Q_PINLOC_43,
         PORT_MEM_Q_PINLOC_44 => PORT_MEM_Q_PINLOC_44,
         PORT_MEM_Q_PINLOC_45 => PORT_MEM_Q_PINLOC_45,
         PORT_MEM_Q_PINLOC_46 => PORT_MEM_Q_PINLOC_46,
         PORT_MEM_Q_PINLOC_47 => PORT_MEM_Q_PINLOC_47,
         PORT_MEM_Q_PINLOC_48 => PORT_MEM_Q_PINLOC_48,
         PORT_MEM_Q_PINLOC_AUTOGEN_WCNT => PORT_MEM_Q_PINLOC_AUTOGEN_WCNT,
         PORT_MEM_DQS_WIDTH => PORT_MEM_DQS_WIDTH,
         PORT_MEM_DQS_PINLOC_0 => PORT_MEM_DQS_PINLOC_0,
         PORT_MEM_DQS_PINLOC_1 => PORT_MEM_DQS_PINLOC_1,
         PORT_MEM_DQS_PINLOC_2 => PORT_MEM_DQS_PINLOC_2,
         PORT_MEM_DQS_PINLOC_3 => PORT_MEM_DQS_PINLOC_3,
         PORT_MEM_DQS_PINLOC_4 => PORT_MEM_DQS_PINLOC_4,
         PORT_MEM_DQS_PINLOC_5 => PORT_MEM_DQS_PINLOC_5,
         PORT_MEM_DQS_PINLOC_6 => PORT_MEM_DQS_PINLOC_6,
         PORT_MEM_DQS_PINLOC_7 => PORT_MEM_DQS_PINLOC_7,
         PORT_MEM_DQS_PINLOC_8 => PORT_MEM_DQS_PINLOC_8,
         PORT_MEM_DQS_PINLOC_9 => PORT_MEM_DQS_PINLOC_9,
         PORT_MEM_DQS_PINLOC_10 => PORT_MEM_DQS_PINLOC_10,
         PORT_MEM_DQS_PINLOC_11 => PORT_MEM_DQS_PINLOC_11,
         PORT_MEM_DQS_PINLOC_12 => PORT_MEM_DQS_PINLOC_12,
         PORT_MEM_DQS_PINLOC_AUTOGEN_WCNT => PORT_MEM_DQS_PINLOC_AUTOGEN_WCNT,
         PORT_MEM_DQS_N_WIDTH => PORT_MEM_DQS_N_WIDTH,
         PORT_MEM_DQS_N_PINLOC_0 => PORT_MEM_DQS_N_PINLOC_0,
         PORT_MEM_DQS_N_PINLOC_1 => PORT_MEM_DQS_N_PINLOC_1,
         PORT_MEM_DQS_N_PINLOC_2 => PORT_MEM_DQS_N_PINLOC_2,
         PORT_MEM_DQS_N_PINLOC_3 => PORT_MEM_DQS_N_PINLOC_3,
         PORT_MEM_DQS_N_PINLOC_4 => PORT_MEM_DQS_N_PINLOC_4,
         PORT_MEM_DQS_N_PINLOC_5 => PORT_MEM_DQS_N_PINLOC_5,
         PORT_MEM_DQS_N_PINLOC_6 => PORT_MEM_DQS_N_PINLOC_6,
         PORT_MEM_DQS_N_PINLOC_7 => PORT_MEM_DQS_N_PINLOC_7,
         PORT_MEM_DQS_N_PINLOC_8 => PORT_MEM_DQS_N_PINLOC_8,
         PORT_MEM_DQS_N_PINLOC_9 => PORT_MEM_DQS_N_PINLOC_9,
         PORT_MEM_DQS_N_PINLOC_10 => PORT_MEM_DQS_N_PINLOC_10,
         PORT_MEM_DQS_N_PINLOC_11 => PORT_MEM_DQS_N_PINLOC_11,
         PORT_MEM_DQS_N_PINLOC_12 => PORT_MEM_DQS_N_PINLOC_12,
         PORT_MEM_DQS_N_PINLOC_AUTOGEN_WCNT => PORT_MEM_DQS_N_PINLOC_AUTOGEN_WCNT,
         PORT_MEM_QK_WIDTH => PORT_MEM_QK_WIDTH,
         PORT_MEM_QK_PINLOC_0 => PORT_MEM_QK_PINLOC_0,
         PORT_MEM_QK_PINLOC_1 => PORT_MEM_QK_PINLOC_1,
         PORT_MEM_QK_PINLOC_2 => PORT_MEM_QK_PINLOC_2,
         PORT_MEM_QK_PINLOC_3 => PORT_MEM_QK_PINLOC_3,
         PORT_MEM_QK_PINLOC_4 => PORT_MEM_QK_PINLOC_4,
         PORT_MEM_QK_PINLOC_5 => PORT_MEM_QK_PINLOC_5,
         PORT_MEM_QK_PINLOC_AUTOGEN_WCNT => PORT_MEM_QK_PINLOC_AUTOGEN_WCNT,
         PORT_MEM_QK_N_WIDTH => PORT_MEM_QK_N_WIDTH,
         PORT_MEM_QK_N_PINLOC_0 => PORT_MEM_QK_N_PINLOC_0,
         PORT_MEM_QK_N_PINLOC_1 => PORT_MEM_QK_N_PINLOC_1,
         PORT_MEM_QK_N_PINLOC_2 => PORT_MEM_QK_N_PINLOC_2,
         PORT_MEM_QK_N_PINLOC_3 => PORT_MEM_QK_N_PINLOC_3,
         PORT_MEM_QK_N_PINLOC_4 => PORT_MEM_QK_N_PINLOC_4,
         PORT_MEM_QK_N_PINLOC_5 => PORT_MEM_QK_N_PINLOC_5,
         PORT_MEM_QK_N_PINLOC_AUTOGEN_WCNT => PORT_MEM_QK_N_PINLOC_AUTOGEN_WCNT,
         PORT_MEM_QKA_WIDTH => PORT_MEM_QKA_WIDTH,
         PORT_MEM_QKA_PINLOC_0 => PORT_MEM_QKA_PINLOC_0,
         PORT_MEM_QKA_PINLOC_1 => PORT_MEM_QKA_PINLOC_1,
         PORT_MEM_QKA_PINLOC_2 => PORT_MEM_QKA_PINLOC_2,
         PORT_MEM_QKA_PINLOC_3 => PORT_MEM_QKA_PINLOC_3,
         PORT_MEM_QKA_PINLOC_4 => PORT_MEM_QKA_PINLOC_4,
         PORT_MEM_QKA_PINLOC_5 => PORT_MEM_QKA_PINLOC_5,
         PORT_MEM_QKA_PINLOC_AUTOGEN_WCNT => PORT_MEM_QKA_PINLOC_AUTOGEN_WCNT,
         PORT_MEM_QKA_N_WIDTH => PORT_MEM_QKA_N_WIDTH,
         PORT_MEM_QKA_N_PINLOC_0 => PORT_MEM_QKA_N_PINLOC_0,
         PORT_MEM_QKA_N_PINLOC_1 => PORT_MEM_QKA_N_PINLOC_1,
         PORT_MEM_QKA_N_PINLOC_2 => PORT_MEM_QKA_N_PINLOC_2,
         PORT_MEM_QKA_N_PINLOC_3 => PORT_MEM_QKA_N_PINLOC_3,
         PORT_MEM_QKA_N_PINLOC_4 => PORT_MEM_QKA_N_PINLOC_4,
         PORT_MEM_QKA_N_PINLOC_5 => PORT_MEM_QKA_N_PINLOC_5,
         PORT_MEM_QKA_N_PINLOC_AUTOGEN_WCNT => PORT_MEM_QKA_N_PINLOC_AUTOGEN_WCNT,
         PORT_MEM_QKB_WIDTH => PORT_MEM_QKB_WIDTH,
         PORT_MEM_QKB_PINLOC_0 => PORT_MEM_QKB_PINLOC_0,
         PORT_MEM_QKB_PINLOC_1 => PORT_MEM_QKB_PINLOC_1,
         PORT_MEM_QKB_PINLOC_2 => PORT_MEM_QKB_PINLOC_2,
         PORT_MEM_QKB_PINLOC_3 => PORT_MEM_QKB_PINLOC_3,
         PORT_MEM_QKB_PINLOC_4 => PORT_MEM_QKB_PINLOC_4,
         PORT_MEM_QKB_PINLOC_5 => PORT_MEM_QKB_PINLOC_5,
         PORT_MEM_QKB_PINLOC_AUTOGEN_WCNT => PORT_MEM_QKB_PINLOC_AUTOGEN_WCNT,
         PORT_MEM_QKB_N_WIDTH => PORT_MEM_QKB_N_WIDTH,
         PORT_MEM_QKB_N_PINLOC_0 => PORT_MEM_QKB_N_PINLOC_0,
         PORT_MEM_QKB_N_PINLOC_1 => PORT_MEM_QKB_N_PINLOC_1,
         PORT_MEM_QKB_N_PINLOC_2 => PORT_MEM_QKB_N_PINLOC_2,
         PORT_MEM_QKB_N_PINLOC_3 => PORT_MEM_QKB_N_PINLOC_3,
         PORT_MEM_QKB_N_PINLOC_4 => PORT_MEM_QKB_N_PINLOC_4,
         PORT_MEM_QKB_N_PINLOC_5 => PORT_MEM_QKB_N_PINLOC_5,
         PORT_MEM_QKB_N_PINLOC_AUTOGEN_WCNT => PORT_MEM_QKB_N_PINLOC_AUTOGEN_WCNT,
         PORT_MEM_CQ_WIDTH => PORT_MEM_CQ_WIDTH,
         PORT_MEM_CQ_PINLOC_0 => PORT_MEM_CQ_PINLOC_0,
         PORT_MEM_CQ_PINLOC_1 => PORT_MEM_CQ_PINLOC_1,
         PORT_MEM_CQ_PINLOC_AUTOGEN_WCNT => PORT_MEM_CQ_PINLOC_AUTOGEN_WCNT,
         PORT_MEM_CQ_N_WIDTH => PORT_MEM_CQ_N_WIDTH,
         PORT_MEM_CQ_N_PINLOC_0 => PORT_MEM_CQ_N_PINLOC_0,
         PORT_MEM_CQ_N_PINLOC_1 => PORT_MEM_CQ_N_PINLOC_1,
         PORT_MEM_CQ_N_PINLOC_AUTOGEN_WCNT => PORT_MEM_CQ_N_PINLOC_AUTOGEN_WCNT,
         PORT_MEM_ALERT_N_WIDTH => PORT_MEM_ALERT_N_WIDTH,
         PORT_MEM_ALERT_N_PINLOC_0 => PORT_MEM_ALERT_N_PINLOC_0,
         PORT_MEM_ALERT_N_PINLOC_1 => PORT_MEM_ALERT_N_PINLOC_1,
         PORT_MEM_ALERT_N_PINLOC_AUTOGEN_WCNT => PORT_MEM_ALERT_N_PINLOC_AUTOGEN_WCNT,
         PORT_MEM_PE_N_WIDTH => PORT_MEM_PE_N_WIDTH,
         PORT_MEM_PE_N_PINLOC_0 => PORT_MEM_PE_N_PINLOC_0,
         PORT_MEM_PE_N_PINLOC_1 => PORT_MEM_PE_N_PINLOC_1,
         PORT_MEM_PE_N_PINLOC_AUTOGEN_WCNT => PORT_MEM_PE_N_PINLOC_AUTOGEN_WCNT,
         PORT_CLKS_SHARING_MASTER_OUT_WIDTH => PORT_CLKS_SHARING_MASTER_OUT_WIDTH,
         PORT_CLKS_SHARING_SLAVE_IN_WIDTH => PORT_CLKS_SHARING_SLAVE_IN_WIDTH,
         PORT_CLKS_SHARING_SLAVE_OUT_WIDTH => PORT_CLKS_SHARING_SLAVE_OUT_WIDTH,
         PORT_AFI_RLAT_WIDTH => PORT_AFI_RLAT_WIDTH,
         PORT_AFI_WLAT_WIDTH => PORT_AFI_WLAT_WIDTH,
         PORT_AFI_SEQ_BUSY_WIDTH => PORT_AFI_SEQ_BUSY_WIDTH,
         PORT_AFI_ADDR_WIDTH => PORT_AFI_ADDR_WIDTH,
         PORT_AFI_BA_WIDTH => PORT_AFI_BA_WIDTH,
         PORT_AFI_BG_WIDTH => PORT_AFI_BG_WIDTH,
         PORT_AFI_C_WIDTH => PORT_AFI_C_WIDTH,
         PORT_AFI_CKE_WIDTH => PORT_AFI_CKE_WIDTH,
         PORT_AFI_CS_N_WIDTH => PORT_AFI_CS_N_WIDTH,
         PORT_AFI_RM_WIDTH => PORT_AFI_RM_WIDTH,
         PORT_AFI_ODT_WIDTH => PORT_AFI_ODT_WIDTH,
         PORT_AFI_RAS_N_WIDTH => PORT_AFI_RAS_N_WIDTH,
         PORT_AFI_CAS_N_WIDTH => PORT_AFI_CAS_N_WIDTH,
         PORT_AFI_WE_N_WIDTH => PORT_AFI_WE_N_WIDTH,
         PORT_AFI_RST_N_WIDTH => PORT_AFI_RST_N_WIDTH,
         PORT_AFI_ACT_N_WIDTH => PORT_AFI_ACT_N_WIDTH,
         PORT_AFI_REQ_N_WIDTH => PORT_AFI_REQ_N_WIDTH,
         PORT_AFI_GNT_N_WIDTH => PORT_AFI_GNT_N_WIDTH,
         PORT_AFI_ERR_N_WIDTH => PORT_AFI_ERR_N_WIDTH,
         PORT_AFI_PAR_WIDTH => PORT_AFI_PAR_WIDTH,
         PORT_AFI_CA_WIDTH => PORT_AFI_CA_WIDTH,
         PORT_AFI_REF_N_WIDTH => PORT_AFI_REF_N_WIDTH,
         PORT_AFI_WPS_N_WIDTH => PORT_AFI_WPS_N_WIDTH,
         PORT_AFI_RPS_N_WIDTH => PORT_AFI_RPS_N_WIDTH,
         PORT_AFI_DOFF_N_WIDTH => PORT_AFI_DOFF_N_WIDTH,
         PORT_AFI_LD_N_WIDTH => PORT_AFI_LD_N_WIDTH,
         PORT_AFI_RW_N_WIDTH => PORT_AFI_RW_N_WIDTH,
         PORT_AFI_LBK0_N_WIDTH => PORT_AFI_LBK0_N_WIDTH,
         PORT_AFI_LBK1_N_WIDTH => PORT_AFI_LBK1_N_WIDTH,
         PORT_AFI_CFG_N_WIDTH => PORT_AFI_CFG_N_WIDTH,
         PORT_AFI_AP_WIDTH => PORT_AFI_AP_WIDTH,
         PORT_AFI_AINV_WIDTH => PORT_AFI_AINV_WIDTH,
         PORT_AFI_DM_WIDTH => PORT_AFI_DM_WIDTH,
         PORT_AFI_DM_N_WIDTH => PORT_AFI_DM_N_WIDTH,
         PORT_AFI_BWS_N_WIDTH => PORT_AFI_BWS_N_WIDTH,
         PORT_AFI_RDATA_DBI_N_WIDTH => PORT_AFI_RDATA_DBI_N_WIDTH,
         PORT_AFI_WDATA_DBI_N_WIDTH => PORT_AFI_WDATA_DBI_N_WIDTH,
         PORT_AFI_RDATA_DINV_WIDTH => PORT_AFI_RDATA_DINV_WIDTH,
         PORT_AFI_WDATA_DINV_WIDTH => PORT_AFI_WDATA_DINV_WIDTH,
         PORT_AFI_DQS_BURST_WIDTH => PORT_AFI_DQS_BURST_WIDTH,
         PORT_AFI_WDATA_VALID_WIDTH => PORT_AFI_WDATA_VALID_WIDTH,
         PORT_AFI_WDATA_WIDTH => PORT_AFI_WDATA_WIDTH,
         PORT_AFI_RDATA_EN_FULL_WIDTH => PORT_AFI_RDATA_EN_FULL_WIDTH,
         PORT_AFI_RDATA_WIDTH => PORT_AFI_RDATA_WIDTH,
         PORT_AFI_RDATA_VALID_WIDTH => PORT_AFI_RDATA_VALID_WIDTH,
         PORT_AFI_RRANK_WIDTH => PORT_AFI_RRANK_WIDTH,
         PORT_AFI_WRANK_WIDTH => PORT_AFI_WRANK_WIDTH,
         PORT_AFI_ALERT_N_WIDTH => PORT_AFI_ALERT_N_WIDTH,
         PORT_AFI_PE_N_WIDTH => PORT_AFI_PE_N_WIDTH,
         PORT_CTRL_AST_CMD_DATA_WIDTH => PORT_CTRL_AST_CMD_DATA_WIDTH,
         PORT_CTRL_AST_WR_DATA_WIDTH => PORT_CTRL_AST_WR_DATA_WIDTH,
         PORT_CTRL_AST_RD_DATA_WIDTH => PORT_CTRL_AST_RD_DATA_WIDTH,
         PORT_CTRL_AMM_ADDRESS_WIDTH => PORT_CTRL_AMM_ADDRESS_WIDTH,
         PORT_CTRL_AMM_RDATA_WIDTH => PORT_CTRL_AMM_RDATA_WIDTH,
         PORT_CTRL_AMM_WDATA_WIDTH => PORT_CTRL_AMM_WDATA_WIDTH,
         PORT_CTRL_AMM_BCOUNT_WIDTH => PORT_CTRL_AMM_BCOUNT_WIDTH,
         PORT_CTRL_AMM_BYTEEN_WIDTH => PORT_CTRL_AMM_BYTEEN_WIDTH,
         PORT_CTRL_USER_REFRESH_REQ_WIDTH => PORT_CTRL_USER_REFRESH_REQ_WIDTH,
         PORT_CTRL_USER_REFRESH_BANK_WIDTH => PORT_CTRL_USER_REFRESH_BANK_WIDTH,
         PORT_CTRL_SELF_REFRESH_REQ_WIDTH => PORT_CTRL_SELF_REFRESH_REQ_WIDTH,
         PORT_CTRL_ECC_WRITE_INFO_WIDTH => PORT_CTRL_ECC_WRITE_INFO_WIDTH,
         PORT_CTRL_ECC_RDATA_ID_WIDTH => PORT_CTRL_ECC_RDATA_ID_WIDTH,
         PORT_CTRL_ECC_READ_INFO_WIDTH => PORT_CTRL_ECC_READ_INFO_WIDTH,
         PORT_CTRL_ECC_CMD_INFO_WIDTH => PORT_CTRL_ECC_CMD_INFO_WIDTH,
         PORT_CTRL_ECC_WB_POINTER_WIDTH => PORT_CTRL_ECC_WB_POINTER_WIDTH,
         PORT_CTRL_MMR_SLAVE_ADDRESS_WIDTH => PORT_CTRL_MMR_SLAVE_ADDRESS_WIDTH,
         PORT_CTRL_MMR_SLAVE_RDATA_WIDTH => PORT_CTRL_MMR_SLAVE_RDATA_WIDTH,
         PORT_CTRL_MMR_SLAVE_WDATA_WIDTH => PORT_CTRL_MMR_SLAVE_WDATA_WIDTH,
         PORT_CTRL_MMR_SLAVE_BCOUNT_WIDTH => PORT_CTRL_MMR_SLAVE_BCOUNT_WIDTH,
         PORT_HPS_EMIF_H2E_WIDTH => PORT_HPS_EMIF_H2E_WIDTH,
         PORT_HPS_EMIF_E2H_WIDTH => PORT_HPS_EMIF_E2H_WIDTH,
         PORT_HPS_EMIF_H2E_GP_WIDTH => PORT_HPS_EMIF_H2E_GP_WIDTH,
         PORT_HPS_EMIF_E2H_GP_WIDTH => PORT_HPS_EMIF_E2H_GP_WIDTH,
         PORT_CAL_DEBUG_ADDRESS_WIDTH => PORT_CAL_DEBUG_ADDRESS_WIDTH,
         PORT_CAL_DEBUG_RDATA_WIDTH => PORT_CAL_DEBUG_RDATA_WIDTH,
         PORT_CAL_DEBUG_WDATA_WIDTH => PORT_CAL_DEBUG_WDATA_WIDTH,
         PORT_CAL_DEBUG_BYTEEN_WIDTH => PORT_CAL_DEBUG_BYTEEN_WIDTH,
         PORT_CAL_DEBUG_OUT_ADDRESS_WIDTH => PORT_CAL_DEBUG_OUT_ADDRESS_WIDTH,
         PORT_CAL_DEBUG_OUT_RDATA_WIDTH => PORT_CAL_DEBUG_OUT_RDATA_WIDTH,
         PORT_CAL_DEBUG_OUT_WDATA_WIDTH => PORT_CAL_DEBUG_OUT_WDATA_WIDTH,
         PORT_CAL_DEBUG_OUT_BYTEEN_WIDTH => PORT_CAL_DEBUG_OUT_BYTEEN_WIDTH,
         PORT_CAL_MASTER_ADDRESS_WIDTH => PORT_CAL_MASTER_ADDRESS_WIDTH,
         PORT_CAL_MASTER_RDATA_WIDTH => PORT_CAL_MASTER_RDATA_WIDTH,
         PORT_CAL_MASTER_WDATA_WIDTH => PORT_CAL_MASTER_WDATA_WIDTH,
         PORT_CAL_MASTER_BYTEEN_WIDTH => PORT_CAL_MASTER_BYTEEN_WIDTH,
         PORT_DFT_NF_IOAUX_PIO_IN_WIDTH => PORT_DFT_NF_IOAUX_PIO_IN_WIDTH,
         PORT_DFT_NF_IOAUX_PIO_OUT_WIDTH => PORT_DFT_NF_IOAUX_PIO_OUT_WIDTH,
         PORT_DFT_NF_PA_DPRIO_REG_ADDR_WIDTH => PORT_DFT_NF_PA_DPRIO_REG_ADDR_WIDTH,
         PORT_DFT_NF_PA_DPRIO_WRITEDATA_WIDTH => PORT_DFT_NF_PA_DPRIO_WRITEDATA_WIDTH,
         PORT_DFT_NF_PA_DPRIO_READDATA_WIDTH => PORT_DFT_NF_PA_DPRIO_READDATA_WIDTH,
         PORT_DFT_NF_PLL_CNTSEL_WIDTH => PORT_DFT_NF_PLL_CNTSEL_WIDTH,
         PORT_DFT_NF_PLL_NUM_SHIFT_WIDTH => PORT_DFT_NF_PLL_NUM_SHIFT_WIDTH,
         PORT_DFT_NF_PLL_CORE_REFCLK_WIDTH => PORT_DFT_NF_PLL_CORE_REFCLK_WIDTH,
         PORT_DFT_NF_CORE_CLK_BUF_OUT_WIDTH => PORT_DFT_NF_CORE_CLK_BUF_OUT_WIDTH,
         PORT_DFT_NF_CORE_CLK_LOCKED_WIDTH => PORT_DFT_NF_CORE_CLK_LOCKED_WIDTH,
         PLL_VCO_FREQ_MHZ_INT => PLL_VCO_FREQ_MHZ_INT,
         PLL_VCO_TO_MEM_CLK_FREQ_RATIO => PLL_VCO_TO_MEM_CLK_FREQ_RATIO,
         PLL_PHY_CLK_VCO_PHASE => PLL_PHY_CLK_VCO_PHASE,
         PLL_VCO_FREQ_PS_STR => PLL_VCO_FREQ_PS_STR,
         PLL_REF_CLK_FREQ_PS_STR => PLL_REF_CLK_FREQ_PS_STR,
         PLL_REF_CLK_FREQ_PS => PLL_REF_CLK_FREQ_PS,
         PLL_SIM_VCO_FREQ_PS => PLL_SIM_VCO_FREQ_PS,
         PLL_SIM_PHYCLK_0_FREQ_PS => PLL_SIM_PHYCLK_0_FREQ_PS,
         PLL_SIM_PHYCLK_1_FREQ_PS => PLL_SIM_PHYCLK_1_FREQ_PS,
         PLL_SIM_PHYCLK_FB_FREQ_PS => PLL_SIM_PHYCLK_FB_FREQ_PS,
         PLL_SIM_PHY_CLK_VCO_PHASE_PS => PLL_SIM_PHY_CLK_VCO_PHASE_PS,
         PLL_SIM_CAL_SLAVE_CLK_FREQ_PS => PLL_SIM_CAL_SLAVE_CLK_FREQ_PS,
         PLL_SIM_CAL_MASTER_CLK_FREQ_PS => PLL_SIM_CAL_MASTER_CLK_FREQ_PS,
         PLL_M_CNT_HIGH => PLL_M_CNT_HIGH,
         PLL_M_CNT_LOW => PLL_M_CNT_LOW,
         PLL_N_CNT_HIGH => PLL_N_CNT_HIGH,
         PLL_N_CNT_LOW => PLL_N_CNT_LOW,
         PLL_M_CNT_BYPASS_EN => PLL_M_CNT_BYPASS_EN,
         PLL_N_CNT_BYPASS_EN => PLL_N_CNT_BYPASS_EN,
         PLL_M_CNT_EVEN_DUTY_EN => PLL_M_CNT_EVEN_DUTY_EN,
         PLL_N_CNT_EVEN_DUTY_EN => PLL_N_CNT_EVEN_DUTY_EN,
         PLL_FBCLK_MUX_1 => PLL_FBCLK_MUX_1,
         PLL_FBCLK_MUX_2 => PLL_FBCLK_MUX_2,
         PLL_M_CNT_IN_SRC => PLL_M_CNT_IN_SRC,
         PLL_CP_SETTING => PLL_CP_SETTING,
         PLL_BW_CTRL => PLL_BW_CTRL,
         PLL_BW_SEL => PLL_BW_SEL,
         PLL_C_CNT_HIGH_0 => PLL_C_CNT_HIGH_0,
         PLL_C_CNT_LOW_0 => PLL_C_CNT_LOW_0,
         PLL_C_CNT_PRST_0 => PLL_C_CNT_PRST_0,
         PLL_C_CNT_PH_MUX_PRST_0 => PLL_C_CNT_PH_MUX_PRST_0,
         PLL_C_CNT_BYPASS_EN_0 => PLL_C_CNT_BYPASS_EN_0,
         PLL_C_CNT_EVEN_DUTY_EN_0 => PLL_C_CNT_EVEN_DUTY_EN_0,
         PLL_C_CNT_FREQ_PS_STR_0 => PLL_C_CNT_FREQ_PS_STR_0,
         PLL_C_CNT_PHASE_PS_STR_0 => PLL_C_CNT_PHASE_PS_STR_0,
         PLL_C_CNT_DUTY_CYCLE_0 => PLL_C_CNT_DUTY_CYCLE_0,
         PLL_C_CNT_OUT_EN_0 => PLL_C_CNT_OUT_EN_0,
         PLL_C_CNT_HIGH_1 => PLL_C_CNT_HIGH_1,
         PLL_C_CNT_LOW_1 => PLL_C_CNT_LOW_1,
         PLL_C_CNT_PRST_1 => PLL_C_CNT_PRST_1,
         PLL_C_CNT_PH_MUX_PRST_1 => PLL_C_CNT_PH_MUX_PRST_1,
         PLL_C_CNT_BYPASS_EN_1 => PLL_C_CNT_BYPASS_EN_1,
         PLL_C_CNT_EVEN_DUTY_EN_1 => PLL_C_CNT_EVEN_DUTY_EN_1,
         PLL_C_CNT_FREQ_PS_STR_1 => PLL_C_CNT_FREQ_PS_STR_1,
         PLL_C_CNT_PHASE_PS_STR_1 => PLL_C_CNT_PHASE_PS_STR_1,
         PLL_C_CNT_DUTY_CYCLE_1 => PLL_C_CNT_DUTY_CYCLE_1,
         PLL_C_CNT_OUT_EN_1 => PLL_C_CNT_OUT_EN_1,
         PLL_C_CNT_HIGH_2 => PLL_C_CNT_HIGH_2,
         PLL_C_CNT_LOW_2 => PLL_C_CNT_LOW_2,
         PLL_C_CNT_PRST_2 => PLL_C_CNT_PRST_2,
         PLL_C_CNT_PH_MUX_PRST_2 => PLL_C_CNT_PH_MUX_PRST_2,
         PLL_C_CNT_BYPASS_EN_2 => PLL_C_CNT_BYPASS_EN_2,
         PLL_C_CNT_EVEN_DUTY_EN_2 => PLL_C_CNT_EVEN_DUTY_EN_2,
         PLL_C_CNT_FREQ_PS_STR_2 => PLL_C_CNT_FREQ_PS_STR_2,
         PLL_C_CNT_PHASE_PS_STR_2 => PLL_C_CNT_PHASE_PS_STR_2,
         PLL_C_CNT_DUTY_CYCLE_2 => PLL_C_CNT_DUTY_CYCLE_2,
         PLL_C_CNT_OUT_EN_2 => PLL_C_CNT_OUT_EN_2,
         PLL_C_CNT_HIGH_3 => PLL_C_CNT_HIGH_3,
         PLL_C_CNT_LOW_3 => PLL_C_CNT_LOW_3,
         PLL_C_CNT_PRST_3 => PLL_C_CNT_PRST_3,
         PLL_C_CNT_PH_MUX_PRST_3 => PLL_C_CNT_PH_MUX_PRST_3,
         PLL_C_CNT_BYPASS_EN_3 => PLL_C_CNT_BYPASS_EN_3,
         PLL_C_CNT_EVEN_DUTY_EN_3 => PLL_C_CNT_EVEN_DUTY_EN_3,
         PLL_C_CNT_FREQ_PS_STR_3 => PLL_C_CNT_FREQ_PS_STR_3,
         PLL_C_CNT_PHASE_PS_STR_3 => PLL_C_CNT_PHASE_PS_STR_3,
         PLL_C_CNT_DUTY_CYCLE_3 => PLL_C_CNT_DUTY_CYCLE_3,
         PLL_C_CNT_OUT_EN_3 => PLL_C_CNT_OUT_EN_3,
         PLL_C_CNT_HIGH_4 => PLL_C_CNT_HIGH_4,
         PLL_C_CNT_LOW_4 => PLL_C_CNT_LOW_4,
         PLL_C_CNT_PRST_4 => PLL_C_CNT_PRST_4,
         PLL_C_CNT_PH_MUX_PRST_4 => PLL_C_CNT_PH_MUX_PRST_4,
         PLL_C_CNT_BYPASS_EN_4 => PLL_C_CNT_BYPASS_EN_4,
         PLL_C_CNT_EVEN_DUTY_EN_4 => PLL_C_CNT_EVEN_DUTY_EN_4,
         PLL_C_CNT_FREQ_PS_STR_4 => PLL_C_CNT_FREQ_PS_STR_4,
         PLL_C_CNT_PHASE_PS_STR_4 => PLL_C_CNT_PHASE_PS_STR_4,
         PLL_C_CNT_DUTY_CYCLE_4 => PLL_C_CNT_DUTY_CYCLE_4,
         PLL_C_CNT_OUT_EN_4 => PLL_C_CNT_OUT_EN_4,
         PLL_C_CNT_HIGH_5 => PLL_C_CNT_HIGH_5,
         PLL_C_CNT_LOW_5 => PLL_C_CNT_LOW_5,
         PLL_C_CNT_PRST_5 => PLL_C_CNT_PRST_5,
         PLL_C_CNT_PH_MUX_PRST_5 => PLL_C_CNT_PH_MUX_PRST_5,
         PLL_C_CNT_BYPASS_EN_5 => PLL_C_CNT_BYPASS_EN_5,
         PLL_C_CNT_EVEN_DUTY_EN_5 => PLL_C_CNT_EVEN_DUTY_EN_5,
         PLL_C_CNT_FREQ_PS_STR_5 => PLL_C_CNT_FREQ_PS_STR_5,
         PLL_C_CNT_PHASE_PS_STR_5 => PLL_C_CNT_PHASE_PS_STR_5,
         PLL_C_CNT_DUTY_CYCLE_5 => PLL_C_CNT_DUTY_CYCLE_5,
         PLL_C_CNT_OUT_EN_5 => PLL_C_CNT_OUT_EN_5,
         PLL_C_CNT_HIGH_6 => PLL_C_CNT_HIGH_6,
         PLL_C_CNT_LOW_6 => PLL_C_CNT_LOW_6,
         PLL_C_CNT_PRST_6 => PLL_C_CNT_PRST_6,
         PLL_C_CNT_PH_MUX_PRST_6 => PLL_C_CNT_PH_MUX_PRST_6,
         PLL_C_CNT_BYPASS_EN_6 => PLL_C_CNT_BYPASS_EN_6,
         PLL_C_CNT_EVEN_DUTY_EN_6 => PLL_C_CNT_EVEN_DUTY_EN_6,
         PLL_C_CNT_FREQ_PS_STR_6 => PLL_C_CNT_FREQ_PS_STR_6,
         PLL_C_CNT_PHASE_PS_STR_6 => PLL_C_CNT_PHASE_PS_STR_6,
         PLL_C_CNT_DUTY_CYCLE_6 => PLL_C_CNT_DUTY_CYCLE_6,
         PLL_C_CNT_OUT_EN_6 => PLL_C_CNT_OUT_EN_6,
         PLL_C_CNT_HIGH_7 => PLL_C_CNT_HIGH_7,
         PLL_C_CNT_LOW_7 => PLL_C_CNT_LOW_7,
         PLL_C_CNT_PRST_7 => PLL_C_CNT_PRST_7,
         PLL_C_CNT_PH_MUX_PRST_7 => PLL_C_CNT_PH_MUX_PRST_7,
         PLL_C_CNT_BYPASS_EN_7 => PLL_C_CNT_BYPASS_EN_7,
         PLL_C_CNT_EVEN_DUTY_EN_7 => PLL_C_CNT_EVEN_DUTY_EN_7,
         PLL_C_CNT_FREQ_PS_STR_7 => PLL_C_CNT_FREQ_PS_STR_7,
         PLL_C_CNT_PHASE_PS_STR_7 => PLL_C_CNT_PHASE_PS_STR_7,
         PLL_C_CNT_DUTY_CYCLE_7 => PLL_C_CNT_DUTY_CYCLE_7,
         PLL_C_CNT_OUT_EN_7 => PLL_C_CNT_OUT_EN_7,
         PLL_C_CNT_HIGH_8 => PLL_C_CNT_HIGH_8,
         PLL_C_CNT_LOW_8 => PLL_C_CNT_LOW_8,
         PLL_C_CNT_PRST_8 => PLL_C_CNT_PRST_8,
         PLL_C_CNT_PH_MUX_PRST_8 => PLL_C_CNT_PH_MUX_PRST_8,
         PLL_C_CNT_BYPASS_EN_8 => PLL_C_CNT_BYPASS_EN_8,
         PLL_C_CNT_EVEN_DUTY_EN_8 => PLL_C_CNT_EVEN_DUTY_EN_8,
         PLL_C_CNT_FREQ_PS_STR_8 => PLL_C_CNT_FREQ_PS_STR_8,
         PLL_C_CNT_PHASE_PS_STR_8 => PLL_C_CNT_PHASE_PS_STR_8,
         PLL_C_CNT_DUTY_CYCLE_8 => PLL_C_CNT_DUTY_CYCLE_8,
         PLL_C_CNT_OUT_EN_8 => PLL_C_CNT_OUT_EN_8,
         SEQ_SYNTH_PARAMS_HEX_FILENAME => "SDRAM_altera_emif_arch_nf_191_65stwui_seq_params_synth.hex",
         SEQ_SIM_PARAMS_HEX_FILENAME => "SDRAM_altera_emif_arch_nf_191_65stwui_seq_params_sim.hex",
         SEQ_CODE_HEX_FILENAME => "SDRAM_altera_emif_arch_nf_191_65stwui_seq_cal.hex"
      )
      port map (
         global_reset_n => global_reset_n,
         pll_ref_clk => pll_ref_clk,
         pll_locked => pll_locked,
         pll_extra_clk_0 => pll_extra_clk_0,
         pll_extra_clk_1 => pll_extra_clk_1,
         pll_extra_clk_2 => pll_extra_clk_2,
         pll_extra_clk_3 => pll_extra_clk_3,
         oct_rzqin => oct_rzqin,
         mem_ck => mem_ck,
         mem_ck_n => mem_ck_n,
         mem_a => mem_a,
         mem_act_n => mem_act_n,
         mem_ba => mem_ba,
         mem_bg => mem_bg,
         mem_c => mem_c,
         mem_cke => mem_cke,
         mem_cs_n => mem_cs_n,
         mem_rm => mem_rm,
         mem_odt => mem_odt,
         mem_reset_n => mem_reset_n,
         mem_par => mem_par,
         mem_alert_n => mem_alert_n,
         mem_dqs => mem_dqs,
         mem_dqs_n => mem_dqs_n,
         mem_dq => mem_dq,
         mem_dbi_n => mem_dbi_n,
         mem_ck_bidir => mem_ck_bidir,
         mem_ck_bidir_n => mem_ck_bidir_n,
         mem_dk => mem_dk,
         mem_dk_n => mem_dk_n,
         mem_dka => mem_dka,
         mem_dka_n => mem_dka_n,
         mem_dkb => mem_dkb,
         mem_dkb_n => mem_dkb_n,
         mem_k => mem_k,
         mem_k_n => mem_k_n,
         mem_req_n => mem_req_n,
         mem_gnt_n => mem_gnt_n,
         mem_err_n => mem_err_n,
         mem_ras_n => mem_ras_n,
         mem_cas_n => mem_cas_n,
         mem_we_n => mem_we_n,
         mem_ca => mem_ca,
         mem_ref_n => mem_ref_n,
         mem_wps_n => mem_wps_n,
         mem_rps_n => mem_rps_n,
         mem_doff_n => mem_doff_n,
         mem_lda_n => mem_lda_n,
         mem_ldb_n => mem_ldb_n,
         mem_rwa_n => mem_rwa_n,
         mem_rwb_n => mem_rwb_n,
         mem_lbk0_n => mem_lbk0_n,
         mem_lbk1_n => mem_lbk1_n,
         mem_cfg_n => mem_cfg_n,
         mem_ap => mem_ap,
         mem_ainv => mem_ainv,
         mem_dm => mem_dm,
         mem_bws_n => mem_bws_n,
         mem_d => mem_d,
         mem_dqa => mem_dqa,
         mem_dqb => mem_dqb,
         mem_dinva => mem_dinva,
         mem_dinvb => mem_dinvb,
         mem_q => mem_q,
         mem_qk => mem_qk,
         mem_qk_n => mem_qk_n,
         mem_qka => mem_qka,
         mem_qka_n => mem_qka_n,
         mem_qkb => mem_qkb,
         mem_qkb_n => mem_qkb_n,
         mem_cq => mem_cq,
         mem_cq_n => mem_cq_n,
         mem_pe_n => mem_pe_n,
         local_cal_success => local_cal_success,
         local_cal_fail => local_cal_fail,
         vid_cal_done_persist => vid_cal_done_persist,
         afi_reset_n => afi_reset_n,
         afi_clk => afi_clk,
         afi_half_clk => afi_half_clk,
         emif_usr_reset_n => emif_usr_reset_n,
         emif_usr_clk => emif_usr_clk,
         emif_usr_half_clk => emif_usr_half_clk,
         emif_usr_reset_n_sec => emif_usr_reset_n_sec,
         emif_usr_clk_sec => emif_usr_clk_sec,
         emif_usr_half_clk_sec => emif_usr_half_clk_sec,
         cal_master_reset_n => cal_master_reset_n,
         cal_master_clk => cal_master_clk,
         cal_slave_reset_n => cal_slave_reset_n,
         cal_slave_clk => cal_slave_clk,
         cal_slave_reset_n_in => cal_slave_reset_n_in,
         cal_slave_clk_in => cal_slave_clk_in,
         cal_debug_reset_n => cal_debug_reset_n,
         cal_debug_clk => cal_debug_clk,
         cal_debug_out_reset_n => cal_debug_out_reset_n,
         cal_debug_out_clk => cal_debug_out_clk,
         clks_sharing_master_out => clks_sharing_master_out,
         clks_sharing_slave_in => clks_sharing_slave_in,
         clks_sharing_slave_out => clks_sharing_slave_out,
         afi_cal_success => afi_cal_success,
         afi_cal_fail => afi_cal_fail,
         afi_cal_req => afi_cal_req,
         afi_rlat => afi_rlat,
         afi_wlat => afi_wlat,
         afi_seq_busy => afi_seq_busy,
         afi_ctl_refresh_done => afi_ctl_refresh_done,
         afi_ctl_long_idle => afi_ctl_long_idle,
         afi_mps_req => afi_mps_req,
         afi_mps_ack => afi_mps_ack,
         afi_addr => afi_addr,
         afi_ba => afi_ba,
         afi_bg => afi_bg,
         afi_c => afi_c,
         afi_cke => afi_cke,
         afi_cs_n => afi_cs_n,
         afi_rm => afi_rm,
         afi_odt => afi_odt,
         afi_ras_n => afi_ras_n,
         afi_cas_n => afi_cas_n,
         afi_we_n => afi_we_n,
         afi_rst_n => afi_rst_n,
         afi_act_n => afi_act_n,
         afi_req_n => afi_req_n,
         afi_gnt_n => afi_gnt_n,
         afi_err_n => afi_err_n,
         afi_par => afi_par,
         afi_ca => afi_ca,
         afi_ref_n => afi_ref_n,
         afi_wps_n => afi_wps_n,
         afi_rps_n => afi_rps_n,
         afi_doff_n => afi_doff_n,
         afi_ld_n => afi_ld_n,
         afi_rw_n => afi_rw_n,
         afi_lbk0_n => afi_lbk0_n,
         afi_lbk1_n => afi_lbk1_n,
         afi_cfg_n => afi_cfg_n,
         afi_ap => afi_ap,
         afi_ainv => afi_ainv,
         afi_dm => afi_dm,
         afi_dm_n => afi_dm_n,
         afi_bws_n => afi_bws_n,
         afi_rdata_dbi_n => afi_rdata_dbi_n,
         afi_wdata_dbi_n => afi_wdata_dbi_n,
         afi_rdata_dinv => afi_rdata_dinv,
         afi_wdata_dinv => afi_wdata_dinv,
         afi_dqs_burst => afi_dqs_burst,
         afi_wdata_valid => afi_wdata_valid,
         afi_wdata => afi_wdata,
         afi_rdata_en_full => afi_rdata_en_full,
         afi_rdata => afi_rdata,
         afi_rdata_valid => afi_rdata_valid,
         afi_rrank => afi_rrank,
         afi_wrank => afi_wrank,
         afi_alert_n => afi_alert_n,
         afi_pe_n => afi_pe_n,
         ast_cmd_data_0 => ast_cmd_data_0,
         ast_cmd_valid_0 => ast_cmd_valid_0,
         ast_cmd_ready_0 => ast_cmd_ready_0,
         ast_cmd_data_1 => ast_cmd_data_1,
         ast_cmd_valid_1 => ast_cmd_valid_1,
         ast_cmd_ready_1 => ast_cmd_ready_1,
         ast_wr_data_0 => ast_wr_data_0,
         ast_wr_valid_0 => ast_wr_valid_0,
         ast_wr_ready_0 => ast_wr_ready_0,
         ast_wr_data_1 => ast_wr_data_1,
         ast_wr_valid_1 => ast_wr_valid_1,
         ast_wr_ready_1 => ast_wr_ready_1,
         ast_rd_data_0 => ast_rd_data_0,
         ast_rd_valid_0 => ast_rd_valid_0,
         ast_rd_ready_0 => ast_rd_ready_0,
         ast_rd_data_1 => ast_rd_data_1,
         ast_rd_valid_1 => ast_rd_valid_1,
         ast_rd_ready_1 => ast_rd_ready_1,
         amm_ready_0 => amm_ready_0,
         amm_read_0 => amm_read_0,
         amm_write_0 => amm_write_0,
         amm_address_0 => amm_address_0,
         amm_readdata_0 => amm_readdata_0,
         amm_writedata_0 => amm_writedata_0,
         amm_burstcount_0 => amm_burstcount_0,
         amm_byteenable_0 => amm_byteenable_0,
         amm_beginbursttransfer_0 => amm_beginbursttransfer_0,
         amm_readdatavalid_0 => amm_readdatavalid_0,
         amm_ready_1 => amm_ready_1,
         amm_read_1 => amm_read_1,
         amm_write_1 => amm_write_1,
         amm_address_1 => amm_address_1,
         amm_readdata_1 => amm_readdata_1,
         amm_writedata_1 => amm_writedata_1,
         amm_burstcount_1 => amm_burstcount_1,
         amm_byteenable_1 => amm_byteenable_1,
         amm_beginbursttransfer_1 => amm_beginbursttransfer_1,
         amm_readdatavalid_1 => amm_readdatavalid_1,
         ctrl_user_priority_hi_0 => ctrl_user_priority_hi_0,
         ctrl_user_priority_hi_1 => ctrl_user_priority_hi_1,
         ctrl_auto_precharge_req_0 => ctrl_auto_precharge_req_0,
         ctrl_auto_precharge_req_1 => ctrl_auto_precharge_req_1,
         ctrl_user_refresh_req => ctrl_user_refresh_req,
         ctrl_user_refresh_bank => ctrl_user_refresh_bank,
         ctrl_user_refresh_ack => ctrl_user_refresh_ack,
         ctrl_self_refresh_req => ctrl_self_refresh_req,
         ctrl_self_refresh_ack => ctrl_self_refresh_ack,
         ctrl_will_refresh => ctrl_will_refresh,
         ctrl_deep_power_down_req => ctrl_deep_power_down_req,
         ctrl_deep_power_down_ack => ctrl_deep_power_down_ack,
         ctrl_power_down_ack => ctrl_power_down_ack,
         ctrl_zq_cal_long_req => ctrl_zq_cal_long_req,
         ctrl_zq_cal_short_req => ctrl_zq_cal_short_req,
         ctrl_zq_cal_ack => ctrl_zq_cal_ack,
         ctrl_ecc_write_info_0 => ctrl_ecc_write_info_0,
         ctrl_ecc_rdata_id_0 => ctrl_ecc_rdata_id_0,
         ctrl_ecc_read_info_0 => ctrl_ecc_read_info_0,
         ctrl_ecc_cmd_info_0 => ctrl_ecc_cmd_info_0,
         ctrl_ecc_idle_0 => ctrl_ecc_idle_0,
         ctrl_ecc_wr_pointer_info_0 => ctrl_ecc_wr_pointer_info_0,
         ctrl_ecc_write_info_1 => ctrl_ecc_write_info_1,
         ctrl_ecc_rdata_id_1 => ctrl_ecc_rdata_id_1,
         ctrl_ecc_read_info_1 => ctrl_ecc_read_info_1,
         ctrl_ecc_cmd_info_1 => ctrl_ecc_cmd_info_1,
         ctrl_ecc_idle_1 => ctrl_ecc_idle_1,
         ctrl_ecc_wr_pointer_info_1 => ctrl_ecc_wr_pointer_info_1,
         mmr_slave_waitrequest_0 => mmr_slave_waitrequest_0,
         mmr_slave_read_0 => mmr_slave_read_0,
         mmr_slave_write_0 => mmr_slave_write_0,
         mmr_slave_address_0 => mmr_slave_address_0,
         mmr_slave_readdata_0 => mmr_slave_readdata_0,
         mmr_slave_writedata_0 => mmr_slave_writedata_0,
         mmr_slave_burstcount_0 => mmr_slave_burstcount_0,
         mmr_slave_beginbursttransfer_0 => mmr_slave_beginbursttransfer_0,
         mmr_slave_readdatavalid_0 => mmr_slave_readdatavalid_0,
         mmr_slave_waitrequest_1 => mmr_slave_waitrequest_1,
         mmr_slave_read_1 => mmr_slave_read_1,
         mmr_slave_write_1 => mmr_slave_write_1,
         mmr_slave_address_1 => mmr_slave_address_1,
         mmr_slave_readdata_1 => mmr_slave_readdata_1,
         mmr_slave_writedata_1 => mmr_slave_writedata_1,
         mmr_slave_burstcount_1 => mmr_slave_burstcount_1,
         mmr_slave_beginbursttransfer_1 => mmr_slave_beginbursttransfer_1,
         mmr_slave_readdatavalid_1 => mmr_slave_readdatavalid_1,
         hps_to_emif => hps_to_emif,
         emif_to_hps => emif_to_hps,
         hps_to_emif_gp => hps_to_emif_gp,
         emif_to_hps_gp => emif_to_hps_gp,
         cal_debug_waitrequest => cal_debug_waitrequest,
         cal_debug_read => cal_debug_read,
         cal_debug_write => cal_debug_write,
         cal_debug_addr => cal_debug_addr,
         cal_debug_read_data => cal_debug_read_data,
         cal_debug_write_data => cal_debug_write_data,
         cal_debug_byteenable => cal_debug_byteenable,
         cal_debug_read_data_valid => cal_debug_read_data_valid,
         cal_debug_out_waitrequest => cal_debug_out_waitrequest,
         cal_debug_out_read => cal_debug_out_read,
         cal_debug_out_write => cal_debug_out_write,
         cal_debug_out_addr => cal_debug_out_addr,
         cal_debug_out_read_data => cal_debug_out_read_data,
         cal_debug_out_write_data => cal_debug_out_write_data,
         cal_debug_out_byteenable => cal_debug_out_byteenable,
         cal_debug_out_read_data_valid => cal_debug_out_read_data_valid,
         cal_master_waitrequest => cal_master_waitrequest,
         cal_master_read => cal_master_read,
         cal_master_write => cal_master_write,
         cal_master_addr => cal_master_addr,
         cal_master_read_data => cal_master_read_data,
         cal_master_write_data => cal_master_write_data,
         cal_master_byteenable => cal_master_byteenable,
         cal_master_read_data_valid => cal_master_read_data_valid,
         cal_master_burstcount => cal_master_burstcount,
         cal_master_debugaccess => cal_master_debugaccess,
         ioaux_pio_in => ioaux_pio_in,
         ioaux_pio_out => ioaux_pio_out,
         pa_dprio_clk => pa_dprio_clk,
         pa_dprio_read => pa_dprio_read,
         pa_dprio_reg_addr => pa_dprio_reg_addr,
         pa_dprio_rst_n => pa_dprio_rst_n,
         pa_dprio_write => pa_dprio_write,
         pa_dprio_writedata => pa_dprio_writedata,
         pa_dprio_block_select => pa_dprio_block_select,
         pa_dprio_readdata => pa_dprio_readdata,
         pll_phase_en => pll_phase_en,
         pll_up_dn => pll_up_dn,
         pll_cnt_sel => pll_cnt_sel,
         pll_num_phase_shifts => pll_num_phase_shifts,
         pll_phase_done => pll_phase_done,
         pll_core_refclk => pll_core_refclk,
         dft_core_clk_buf_out => dft_core_clk_buf_out,
         dft_core_clk_locked => dft_core_clk_locked
      );
end architecture rtl;
