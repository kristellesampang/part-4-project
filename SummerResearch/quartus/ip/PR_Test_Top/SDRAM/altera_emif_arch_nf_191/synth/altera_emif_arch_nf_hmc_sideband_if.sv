// (C) 2001-2024 Intel Corporation. All rights reserved.
// Your use of Intel Corporation's design tools, logic functions and other 
// software and tools, and its AMPP partner logic functions, and any output 
// files from any of the foregoing (including device programming or simulation 
// files), and any associated documentation or information are expressly subject 
// to the terms and conditions of the Intel Program License Subscription 
// Agreement, Intel FPGA IP License Agreement, or other applicable 
// license agreement, including, without limitation, that your use is for the 
// sole purpose of programming logic devices manufactured by Intel and sold by 
// Intel or its authorized distributors.  Please refer to the applicable 
// agreement for further details.


///////////////////////////////////////////////////////////////////////////////
// This module is responsible for exposing the controller sideband interfaces 
// through which soft logic interacts with the Hard Memory Controller. 
// The tile WYSIWYG blocks collapse the individual sideband signals into big
// buses. This module re-logics the big buses into proper interfaces.
// 
///////////////////////////////////////////////////////////////////////////////
module altera_emif_arch_nf_hmc_sideband_if #(

   // Parameters describing lanes/tiles
   parameter PHY_PING_PONG_EN                        = 0,
   parameter LANES_PER_TILE                          = 1,
   parameter NUM_OF_RTL_TILES                        = 1,
   parameter PRI_AC_TILE_INDEX                       = -1,
   parameter SEC_AC_TILE_INDEX                       = -1,
   parameter PRI_RDATA_TILE_INDEX                    = -1,
   parameter PRI_RDATA_LANE_INDEX                    = -1,
   parameter PRI_WDATA_TILE_INDEX                    = -1,
   parameter PRI_WDATA_LANE_INDEX                    = -1,
   parameter SEC_RDATA_TILE_INDEX                    = -1,
   parameter SEC_RDATA_LANE_INDEX                    = -1,
   parameter SEC_WDATA_TILE_INDEX                    = -1,
   parameter SEC_WDATA_LANE_INDEX                    = -1,
   parameter PRI_HMC_DBC_SHADOW_LANE_INDEX           = -1,

   // Definition of port widths for "ctrl_user_refresh" interface
   parameter PORT_CTRL_USER_REFRESH_REQ_WIDTH        = 1,
   parameter PORT_CTRL_USER_REFRESH_BANK_WIDTH       = 1,

   // Definition of port widths for "ctrl_self_refresh" interface
   parameter PORT_CTRL_SELF_REFRESH_REQ_WIDTH        = 1,

   // Definition describing ECC
   parameter PRI_HMC_CFG_ENABLE_ECC                  = "disable",
   parameter SEC_HMC_CFG_ENABLE_ECC                  = "disable",

   // Definition of port widths for "ctrl_ecc" interface
   parameter PORT_CTRL_ECC_WRITE_INFO_WIDTH          = 1,
   parameter PORT_CTRL_ECC_READ_INFO_WIDTH           = 1,
   parameter PORT_CTRL_ECC_CMD_INFO_WIDTH            = 1,
   parameter PORT_CTRL_ECC_WB_POINTER_WIDTH          = 1,
   parameter PORT_CTRL_ECC_RDATA_ID_WIDTH            = 1
   
) (
   // Collapsed sideband signals going into/out of tiles
   output logic [41:0]                                                  core2ctl_sideband_0,
   input  logic [13:0]                                                  ctl2core_sideband_0,
   output logic [41:0]                                                  core2ctl_sideband_1,
   input  logic [13:0]                                                  ctl2core_sideband_1,
   
   // Additional ECC signals going into/out of lanes
   output logic [12:0]                                                  core2l_wr_ecc_info_0,
   output logic [12:0]                                                  core2l_wr_ecc_info_1,
   input  logic [12:0]                                                  ctl2core_avl_rdata_id_0,
   input  logic [12:0]                                                  ctl2core_avl_rdata_id_1,
   input  logic [NUM_OF_RTL_TILES-1:0][LANES_PER_TILE-1:0][11:0]        l2core_wb_pointer_for_ecc,

   // Ports for "ctrl_user_refresh" interface
   input  logic [PORT_CTRL_USER_REFRESH_REQ_WIDTH-1:0]                  ctrl_user_refresh_req,
   input  logic [PORT_CTRL_USER_REFRESH_BANK_WIDTH-1:0]                 ctrl_user_refresh_bank,
   output logic                                                         ctrl_user_refresh_ack,

   // Ports for "ctrl_self_refresh" interface
   input  logic [PORT_CTRL_SELF_REFRESH_REQ_WIDTH-1:0]                  ctrl_self_refresh_req,
   output logic                                                         ctrl_self_refresh_ack,

   // Ports for "ctrl_will_refresh" interface
   output logic                                                         ctrl_will_refresh,
   
   // Ports for "ctrl_deep_power_down" interface
   input  logic                                                         ctrl_deep_power_down_req,
   output logic                                                         ctrl_deep_power_down_ack,

   // Ports for "ctrl_power_down" interface
   output logic                                                         ctrl_power_down_ack,
      
   // Ports for "ctrl_zq_cal" interface
   input  logic                                                         ctrl_zq_cal_long_req,
   input  logic                                                         ctrl_zq_cal_short_req,
   output logic                                                         ctrl_zq_cal_ack,

   // Ports for "ctrl_ecc" interface
   input  logic [PORT_CTRL_ECC_WRITE_INFO_WIDTH-1:0]                    ctrl_ecc_write_info_0,
   output logic [PORT_CTRL_ECC_WB_POINTER_WIDTH-1:0]                    ctrl_ecc_wr_pointer_info_0,
   output logic [PORT_CTRL_ECC_READ_INFO_WIDTH-1:0]                     ctrl_ecc_read_info_0,
   output logic [PORT_CTRL_ECC_CMD_INFO_WIDTH-1:0]                      ctrl_ecc_cmd_info_0,
   output logic                                                         ctrl_ecc_idle_0,
   output logic [PORT_CTRL_ECC_RDATA_ID_WIDTH-1:0]                      ctrl_ecc_rdata_id_0,
   
   input  logic [PORT_CTRL_ECC_WRITE_INFO_WIDTH-1:0]                    ctrl_ecc_write_info_1,
   output logic [PORT_CTRL_ECC_WB_POINTER_WIDTH-1:0]                    ctrl_ecc_wr_pointer_info_1,
   output logic [PORT_CTRL_ECC_READ_INFO_WIDTH-1:0]                     ctrl_ecc_read_info_1,
   output logic [PORT_CTRL_ECC_CMD_INFO_WIDTH-1:0]                      ctrl_ecc_cmd_info_1,
   output logic                                                         ctrl_ecc_idle_1,
   output logic [PORT_CTRL_ECC_RDATA_ID_WIDTH-1:0]                      ctrl_ecc_rdata_id_1

);
   timeunit 1ns;
   timeprecision 1ps;
   
   assign core2ctl_sideband_0[3:0]   = ctrl_user_refresh_req;
   assign core2ctl_sideband_0[19:4]  = ctrl_user_refresh_bank;
   assign core2ctl_sideband_0[20]    = ctrl_deep_power_down_req;
   assign core2ctl_sideband_0[24:21] = ctrl_self_refresh_req;
   assign core2ctl_sideband_0[25]    = ctrl_zq_cal_long_req;
   assign core2ctl_sideband_0[26]    = ctrl_zq_cal_short_req;
   
   assign ctrl_user_refresh_ack      = ctl2core_sideband_0[6];
   assign ctrl_deep_power_down_ack   = ctl2core_sideband_0[7];
   assign ctrl_power_down_ack        = ctl2core_sideband_0[8];
   assign ctrl_self_refresh_ack      = ctl2core_sideband_0[9];
   assign ctrl_zq_cal_ack            = ctl2core_sideband_0[10];
   assign ctrl_will_refresh          = ctl2core_sideband_0[13];
   
   assign core2ctl_sideband_1[3:0]   = '0;  
   assign core2ctl_sideband_1[19:4]  = '0;  
   assign core2ctl_sideband_1[20]    = '0;  
   assign core2ctl_sideband_1[24:21] = '0;  
   assign core2ctl_sideband_1[25]    = '0;  
   assign core2ctl_sideband_1[26]    = '0;  
   
   
   assign ctrl_ecc_read_info_0       = ctl2core_sideband_0[2:0];
   assign ctrl_ecc_cmd_info_0        = ctl2core_sideband_0[5:3];
   assign ctrl_ecc_idle_0            = ctl2core_sideband_0[12];
   assign ctrl_ecc_rdata_id_0        = ctl2core_avl_rdata_id_0;
   
   assign ctrl_ecc_wr_pointer_info_0 = l2core_wb_pointer_for_ecc[PRI_AC_TILE_INDEX][PRI_HMC_DBC_SHADOW_LANE_INDEX];
   
   assign core2ctl_sideband_0[41:27] = ctrl_ecc_write_info_0;
   assign core2l_wr_ecc_info_0       = ctrl_ecc_write_info_0[14:2];
   
   assign ctrl_ecc_read_info_1       = ctl2core_sideband_1[2:0];
   assign ctrl_ecc_cmd_info_1        = ctl2core_sideband_1[5:3];
   assign ctrl_ecc_idle_1            = ctl2core_sideband_1[12];
   assign ctrl_ecc_rdata_id_1        = ctl2core_avl_rdata_id_1;
   assign ctrl_ecc_wr_pointer_info_1 = l2core_wb_pointer_for_ecc[SEC_WDATA_TILE_INDEX][SEC_WDATA_LANE_INDEX];
   assign core2ctl_sideband_1[41:27] = ctrl_ecc_write_info_1;
   assign core2l_wr_ecc_info_1       = ctrl_ecc_write_info_1[14:2];
endmodule

`ifdef QUESTA_INTEL_OEM
`pragma questa_oem_00 "BUwlYfHqKCYPcXIP7xrnQ9Se5hc+WCW/arrBU5W+lFIz4wbefW0My5vX6xBCsRZeDFCX8Hl+sMTGZJoji0A9oHt2LftULu0FLRu+mk51alTJ3uomtGywN+PFSqf/fgMyVODG1WmlMI935DQeVtY8c0/LTaQ2hClpkazdo6IufJnkztEP0APWCavclOK3sEzCw6ySG+devOHMd72zKrOr9Hnpx+hVpDg9hFG/o4PmuuHY7oVA+gGYsx3HsBJJc28DM6YR/Djn61/aFR1fxASVJ54HuAQ+1g5NzNgtaNBi10xZIvR+ICWxW4f/vxxjV/IArLuczLtjRhEluViXRYdIOysAAzwyd8k3O68tCCJKNQx8JSBHpT9HdmTBAcqkwRSC8GpyWcksqWK/+/RFvvOt9quWBBG0dQ5rrvZkGCXw+XUt1PNWDC/eIfxtzPScOWdRE3GlBqu5dNWxd0AOCnp6cxUlxqqUbdk5jbfo62jhsphsvyhLL6is86m6ufRJFBOcP4yyFWa8C6MAeiMDrG5/+Ne2/dImeaI1P+hdkh10/WZ0Kf0y1lRVz29oIIkERHOhidX3NSH0w6F+khBL5Nk18tFg1AMmk2yGRijaBQZRFJ/ibbBVZ7yhmkRRCrbYgt0HHakiAdTsWapHp/PSendPkaQLhs03JF01LrKQyexNii62xAPYRnFvV1SIKKcqIR2V7/GTK56RB/4W6T9CeOD11W2nNu9q8Zfsuzx8v4DlSO3C/TIL1hzI+AqYwwYS4bSv/2GfwfpOMsmM1YqS6rJqI7Inh/sZ3LapQttsihbMLoU0GnZ0aZu26JpUhex1+Rhb7mZWw8xJs6RzIj/0+FnnK1+X5eDaOAEaf3fJdlPerJhFQRpITyqbYVZCS+PL+VzLL04XjLRQV1T1QmOTyFNKiVt56VrgQcSnYVObnz8yBV7oGLuooBdfkz/zUN6RSFX5+VIBE3zK02HhH8zVIjcyceFV3ISkJayBVpAz9sgGA3Ks9E7POFCKVvOnIeOSLDV0"
`endif