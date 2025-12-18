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


module altera_emif_arch_nf_buf_udir_df_i # (
   parameter OCT_CONTROL_WIDTH = 1,
   parameter CALIBRATED_OCT = 1
) (
   input  logic i,
   input  logic ibar,
   output logic o,
   input  logic [OCT_CONTROL_WIDTH-1:0] oct_stc,
   input  logic [OCT_CONTROL_WIDTH-1:0] oct_ptc
);
   timeunit 1ns;
   timeprecision 1ps;
   
   generate
      if (CALIBRATED_OCT) 
      begin : cal_oct   
         twentynm_io_ibuf  # (
            .differential_mode ("true")
         ) ibuf (
            .i(i),
            .ibar(ibar),
            .o(o),
            .seriesterminationcontrol(oct_stc),
            .parallelterminationcontrol(oct_ptc),
            .dynamicterminationcontrol()
            );
      end else 
      begin : no_oct
         twentynm_io_ibuf  # (
            .differential_mode ("true")
         ) ibuf (
            .i(i),
            .ibar(ibar),
            .o(o),
            .seriesterminationcontrol(),
            .parallelterminationcontrol(),
            .dynamicterminationcontrol()
            );      
      end
   endgenerate      
endmodule

`ifdef QUESTA_INTEL_OEM
`pragma questa_oem_00 "BUwlYfHqKCYPcXIP7xrnQ9Se5hc+WCW/arrBU5W+lFIz4wbefW0My5vX6xBCsRZeDFCX8Hl+sMTGZJoji0A9oHt2LftULu0FLRu+mk51alTJ3uomtGywN+PFSqf/fgMyVODG1WmlMI935DQeVtY8c0/LTaQ2hClpkazdo6IufJnkztEP0APWCavclOK3sEzCw6ySG+devOHMd72zKrOr9Hnpx+hVpDg9hFG/o4PmuuEPIpvip8MTVo330kiihfZ8Zbeh4olvHY44LBdQ4SFoH1Bn6RzvbdBPCgNNVeesCqopZTVEEtzspqmArWjVimLLUHUsCms1YORMH010FP/sopaAYoW2NK1hQecGu6p/+RyPAJJouohujAcKbmvEXsXgBuo2aZNJliCVM1r7n52eN1Ex78IRVPCUWJT2wOL7UHlaKdTduH0QV4G9abXlIiLrXPA0UitONxA7daIjNLI0V8T5C4FzhrJ61vF9x+fTa4ZlmsB2J3FDCXZvRUVS2bmvuf5t47kzq2pGYizKBmB0g490DqF0Ocgyz1VditlVaW2c+c3juryfcyfcfzc2UP6hDYAo/6tpz22JY+fif6JLXeUVLR/z6BJRoc40GPgmnfKTwxjSXGzKh5wgay2iS59ATuVK3KRoq40X/9eNsp6ByzQ5hSvu8pge+8Nx4rIN2eQLjXQlbfgtSF6BlxB4SW3YFwFzs/YEIKHWk2SYE24mhtyqFnGNoz4gkVOxEfkXJPnu+4E4d8ukEhuXpSQ7KTWbYOJEViz5X7bHR7Pa7jkTPtaDE5CjjQXPOBg9MsV8YVn1loqiPMPtKMqkl5HgiAGCcXGq5fyOCr2s2yXblAg+AFIsrr+iFfSTVPcI6BaeNdFp3OwOVo7nivIUESaqPIIwExUnzclsCpGSPDb48izDuLcJhWsEYvevVmrR1S3/MCHWZhYzfAynZcylB+DdaQDvkkCs9x5dm0VfcL4XrJrKuJzhn6QkFEF0t/8k03bNqdBW19ZuvGqBLf/+eY9urcL1"
`endif
