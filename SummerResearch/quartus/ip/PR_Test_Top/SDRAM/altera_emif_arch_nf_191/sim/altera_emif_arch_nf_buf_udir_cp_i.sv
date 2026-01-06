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


module altera_emif_arch_nf_buf_udir_cp_i # (
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
   
   generate
      if (CALIBRATED_OCT) 
      begin : cal_oct      
         twentynm_io_ibuf ibuf(
            .i(i),
            .o(o),
            .ibar(),
            .seriesterminationcontrol(oct_stc),
            .parallelterminationcontrol(oct_ptc),
            .dynamicterminationcontrol()
            );
            
         twentynm_io_ibuf ibuf_bar(
            .i(ibar),
            .o(obar),
            .ibar(),
            .seriesterminationcontrol(oct_stc),
            .parallelterminationcontrol(oct_ptc),
            .dynamicterminationcontrol()
            );
      end else 
      begin : no_oct
         twentynm_io_ibuf ibuf(
            .i(i),
            .o(o),
            .ibar(),
            .seriesterminationcontrol(),
            .parallelterminationcontrol(),
            .dynamicterminationcontrol()
            );
            
         twentynm_io_ibuf ibuf_bar(
            .i(ibar),
            .o(obar),
            .ibar(),
            .seriesterminationcontrol(),
            .parallelterminationcontrol(),
            .dynamicterminationcontrol()
            );      
      end
   endgenerate
endmodule

`ifdef QUESTA_INTEL_OEM
`pragma questa_oem_00 "BUwlYfHqKCYPcXIP7xrnQ9Se5hc+WCW/arrBU5W+lFIz4wbefW0My5vX6xBCsRZeDFCX8Hl+sMTGZJoji0A9oHt2LftULu0FLRu+mk51alTJ3uomtGywN+PFSqf/fgMyVODG1WmlMI935DQeVtY8c0/LTaQ2hClpkazdo6IufJnkztEP0APWCavclOK3sEzCw6ySG+devOHMd72zKrOr9Hnpx+hVpDg9hFG/o4PmuuGsDXfjak7ucSEuU78aLqp7D43+cOBEaLiPUFBytJF7SrvItJW6NPhG6TwSSbu0qykXGGiXFTGBRrODEEVICp5QHrWJZiBG4zzEpY/DSkXrKw7mIrYryxsPVZ+gUrkXUfGSvgIHI/rsNRN4oYGGxQd6u2RYnBADVlLAv8ITm2I3mdJVui7Wfq05PShFi0cHxOcz+ae3r5QFW/v6mw9jRqC1Du0rUS7hZL3bIkeT7HBXGlJu8xvATgPbqZtWt0GQQaA4wH5CKXNA0ce3HQkeQLmKcAOkpuGXK34Co6vH3CK+kDiDrNG2ZUMl9NTc0TKiiRFewjXvtRAPARgLDlSbOv9iLAUtGq3R5LCTVcA4d1hkqJP+m8IdBYT/J7eKqURDcyDT3Oc9L3u3SGDPOVLyxQ7dQF6zh+AFk3d+167per/xPC6QH4Ozx30NUWiZaPDefTAE7ZTxA4KEuFgQSZLOAT4MLCCvPy10dQRHzOQdXmqSDvzvuQvd9OpHxjx4h14oX/hmFujSZoBpyrVPGuwxZZ4PyEsLXhWSff8Cm7cePHCKjmdUZAypq/IWZ/T3kohIeemcJppaU30pbTWinEfrmZH+2Ubkof/kSiD6ppBhy1IEFxBE9FSXNjSHAbKoeGGCuvY4o8dqssDxClqDmcqSLouJ3Ub9t+SNBymRdLssU1c+8u6mjBi8VE44ASgG8rWtqOrDBUTUTEccluxh+CWvV9bAZJjB9xf8+1jihtYgiqc2mmzzvtcafBi54yoOpBB1mUju214KTqozjXZnIL/1PThQ"
`endif
