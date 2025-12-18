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


module altera_emif_arch_nf_buf_udir_df_o #(
   parameter OCT_CONTROL_WIDTH = 1,
   parameter CALIBRATED_OCT = 1
) (
   input  logic i,
   input  logic ibar,
   output logic o,
   output logic obar,
   input  logic [OCT_CONTROL_WIDTH-1:0] oct_stc,
   input  logic [OCT_CONTROL_WIDTH-1:0] oct_ptc
);
   timeunit 1ns;
   timeprecision 1ps;
   
   logic pdiff_out_o;
   logic pdiff_out_obar;
   
   logic pdiff_out_oe;
   logic pdiff_out_oebar;

   twentynm_pseudo_diff_out # (
      .feedthrough("true")
   ) pdiff_out (
      .i(i),
      .ibar(ibar),
      .o(pdiff_out_o),
      .obar(pdiff_out_obar),
      .oein(1'b1),
      .oebin(1'b1),
      .oeout(pdiff_out_oe),
      .oebout(pdiff_out_oebar),
      .dtcin(),
      .dtcbarin(),
      .dtc(),
      .dtcbar()
   );

   generate
      if (CALIBRATED_OCT) 
      begin : cal_oct
         twentynm_io_obuf obuf (
            .i(pdiff_out_o),
            .o(o),
            .oe(pdiff_out_oe),
            .seriesterminationcontrol(oct_stc),
            .parallelterminationcontrol(oct_ptc),
            .dynamicterminationcontrol(),
            .obar(),
            .devoe()
         );              
            
         twentynm_io_obuf obuf_bar (
            .i(pdiff_out_obar),
            .o(obar),
            .oe(pdiff_out_oebar),
            .seriesterminationcontrol(oct_stc),
            .parallelterminationcontrol(oct_ptc),
            .dynamicterminationcontrol(),
            .obar(),
            .devoe()
         );
      end else 
      begin : no_oct
         twentynm_io_obuf obuf (
            .i(pdiff_out_o),
            .o(o),
            .oe(pdiff_out_oe),
            .seriesterminationcontrol(),
            .parallelterminationcontrol(),
            .dynamicterminationcontrol(),
            .obar(),
            .devoe()
         );              
            
         twentynm_io_obuf obuf_bar (
            .i(pdiff_out_obar),
            .o(obar),
            .oe(pdiff_out_oebar),
            .seriesterminationcontrol(),
            .parallelterminationcontrol(),
            .dynamicterminationcontrol(),
            .obar(),
            .devoe()
         );
      end
   endgenerate
   
endmodule
`ifdef QUESTA_INTEL_OEM
`pragma questa_oem_00 "BUwlYfHqKCYPcXIP7xrnQ9Se5hc+WCW/arrBU5W+lFIz4wbefW0My5vX6xBCsRZeDFCX8Hl+sMTGZJoji0A9oHt2LftULu0FLRu+mk51alTJ3uomtGywN+PFSqf/fgMyVODG1WmlMI935DQeVtY8c0/LTaQ2hClpkazdo6IufJnkztEP0APWCavclOK3sEzCw6ySG+devOHMd72zKrOr9Hnpx+hVpDg9hFG/o4PmuuG3m6ldzM34Mr6IZIu7slNcQsM1baFsMR50GtZJvMuwPfc5ZXvZfw2cStgCDC2ufiLGsnaUFNdw7VLq6p68xNLSbE1b1uK2s+/amhf64wAMZhnII10j/FAkGi12H7PPY4MLW8ieJPmkfHSH/m5/wjS/KBMfiBmQ0EwX8gwX/ZZDSZliL1SdhhMwGqNvNdiizMXPv7+mDDCaGpcsZ8CRIduvAhvnnnjIFl70gwSa1OJSjciWUOLGkDJIDOw/WMZsIWRyQsLBGcJk7BTtQ8+nuV9MwZgX4Z6aq+LaHQbO0K8xG7/lTKMFYaCXWyAFNpj4+ZlQmh6Zkvusw74L4G36fgtLLxZsZZZptUTUfq2en8PIuI3J5g6+vQdDiemhb/kQrSEEqdXTsDr+o5u6aWvTi0tdHqKRWPAr3JSIYl+s/9Q1LgSwConIxG9v+HWn9Lu+fCJKyF73fbpDb/T2a9RDrSO7IlUOz/S0j+g6TXgAKH5C64Z5OYdXgxVLbIT9bIjic3LLKSQWjDScEjUPfjq1c/lZ7/oLaKbsyx/Pcxs3hKCo4exH/v3MBeP4kc0PjyW7uCyP6bULlcdkMB55jXfxYxDnmkmrlYR1OWjTYE8HMRrVNrBFqq9tOuVIrX/CsCsAim1L+k5liCyDMt4inh550JKUjq7L1xr3j0acnUEy6rEdM/XDiq6Nd1LmwZ6y5VhaSMEDHPrVS4J8mLixLBxXxlFKHinhHn/feNyJOHbPMlO5Rj8XrdeQ+ytvRkdmma/mrdb4iJ7d0aesfbwhsNhQ3hg4"
`endif
