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


module altera_emif_arch_nf_buf_udir_se_o #(
   parameter OCT_CONTROL_WIDTH = 1,
   parameter CALIBRATED_OCT = 1
) (
   input  logic i,
   output logic o,
   input  logic [OCT_CONTROL_WIDTH-1:0] oct_stc,
   input  logic [OCT_CONTROL_WIDTH-1:0] oct_ptc
);
   timeunit 1ns;
   timeprecision 1ps;

   generate
      if (CALIBRATED_OCT) 
      begin : cal_oct
         twentynm_io_obuf obuf (
            .i(i),
            .o(o),
            .seriesterminationcontrol(oct_stc),
            .parallelterminationcontrol(oct_ptc),
            .obar(),
            .oe(),
            .dynamicterminationcontrol(),
            .devoe()
            );    
      end else 
      begin : no_oct
         twentynm_io_obuf obuf (
            .i(i),
            .o(o),
            .seriesterminationcontrol(),
            .parallelterminationcontrol(),
            .obar(),
            .oe(),
            .dynamicterminationcontrol(),
            .devoe()
            );    
      end
   endgenerate
endmodule

`ifdef QUESTA_INTEL_OEM
`pragma questa_oem_00 "BUwlYfHqKCYPcXIP7xrnQ9Se5hc+WCW/arrBU5W+lFIz4wbefW0My5vX6xBCsRZeDFCX8Hl+sMTGZJoji0A9oHt2LftULu0FLRu+mk51alTJ3uomtGywN+PFSqf/fgMyVODG1WmlMI935DQeVtY8c0/LTaQ2hClpkazdo6IufJnkztEP0APWCavclOK3sEzCw6ySG+devOHMd72zKrOr9Hnpx+hVpDg9hFG/o4PmuuGGYNta0xJ9x72352nHj3K6uwzJerN4+84LfpAsgNPcw+yl3QDRp5YVOJwa7CYSylz6avU7x3Em4UOLG/Vy6J9l7oTxvWccxYqNhxhfxo4Oi/K0xeUGBaWZasqrgSbASgAwxRpR7nQ97rEBqErMCduW+XUrBDP1SO021Es+jYFUe8t7+wP+5vW26a6XXoPhrx7lRv9YfKUqE6w1ZkvpiTd6+Zo065JV13aqTyZaLLyU4S349Pla+Xhey+lvZ71EzGIlIueiaJwg+hKEf8p2JjS0bKnulZ/owVB3krn6APz+RjpQEG8hNhEAlWfGy7ltEFXi6uLyXt+bSzd6itNztPqhvyWoxtGjok3NmmuTlZEAbIzHuX0iq1C6BdkMHo+Xz1amS+zHdMFgK0+js++hN2evTYSrLpCeGGnXJagBUXFrnkdEWoS/USQ/J9Yz/5i5ElPgHM7FaII7CBYo3zaL7JVuuSQcantEYXmNFuTcqbrARp6C+n9aEikLOKUvh5yAwDBrerHN7TLWtw5l20zJKxNxlp4A3hP8m1f4nQn6lyxR5zfHCI1C6z3BNg9xy8jyeNj+gcD1JdXXahKGQyc9i5S/aMaSyzIQ0InDjPo2MrCewEu/c1SmaQ5YAvRh87qVP0qQKpY55Yr+q0yadUxOEzT+Zh2O51aSeXqx2wEjuZvOzbbuAOY0vkvgqoYIeDiW20RYupkogW1egiD8lmlENAjeCTtc1Mca85b2yEuCp09hk34Jky3+wMpPUFTwvRNXCkuVniPEUutc6Ll5g29LrtKq"
`endif
