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


module altera_emif_arch_nf_buf_udir_se_i #(
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
         twentynm_io_ibuf ibuf(
            .i(i),
            .o(o),
            .seriesterminationcontrol(oct_stc),
            .parallelterminationcontrol(oct_ptc),
            .ibar(),
            .dynamicterminationcontrol()
            );    
      end else 
      begin : no_oct
         twentynm_io_ibuf ibuf(
            .i(i),
            .o(o),
            .seriesterminationcontrol(),
            .parallelterminationcontrol(),
            .ibar(),
            .dynamicterminationcontrol()
            );
      end
   endgenerate
endmodule

`ifdef QUESTA_INTEL_OEM
`pragma questa_oem_00 "BUwlYfHqKCYPcXIP7xrnQ9Se5hc+WCW/arrBU5W+lFIz4wbefW0My5vX6xBCsRZeDFCX8Hl+sMTGZJoji0A9oHt2LftULu0FLRu+mk51alTJ3uomtGywN+PFSqf/fgMyVODG1WmlMI935DQeVtY8c0/LTaQ2hClpkazdo6IufJnkztEP0APWCavclOK3sEzCw6ySG+devOHMd72zKrOr9Hnpx+hVpDg9hFG/o4PmuuFgrVdyNZuDVLXF7YQWXdx1Xi8jxomsb2XJ6IiNSX2BkPRKvl0C1iiyY/4tJcYX+fV8u4A0hB+GWaxCTBTlDL7nrf4On5xH4V0B/zzPXtYHR7ivpPUMF+xKoq0bYTW7YPzWYE/Wa35k1ZLHt4FMOru2z8ehR4RuKCK44rQAv8Zmyg17HBZ6zsrNjI3z9YRc4AycNkTqn7o7/S6EwkCGfzbgXz+AL53HXmf8aSasXvq8k2UEB9Qo8rWsSEooQkifY60kvlVZJhhmwT36L5GpUUIzB9vc8bcws6zL4YXg5zQeRzgvNsehKVZyvpyPBzWfW0NEEj0BkRVLekPnFPsW6vk7fI8+i6Jz5/fwWio0RLyhH4qPV6fokTpLgzP7V85SUeqNAscgnWzElYJFMhJXOXTM/HL2uA7poagb0p1DRGMY3YJkjLnJFEuAO7dbhj7XZKFOFKzSpSr5t9tqdYiIZqZYipmJvQ+qKAO7EESSiI97iHbHsmm5gniw3QUM4n1C1YYW+/QbkFbqerRkLsMQ7i/J8gLEB3rhOh1GcOn+ZokmmBIx6HaGhU7YaiIHTMuIIlNgp2CtISwqlFIRiJd3uftB+Rds3vpnfcWZOAgbJOmHa27ClbzWAWjdidmKN88rUBFra16l2QnOHnyCMP8YoRHKZgNAsqGeXBLYmJ4XYY6IZKAV+P68F8JN2Yxw03GnGR7sSGERnG34ZT1HlZ0nExvNr/3PUSBVeVbdYJW1hmpQTSpfFGQ+NfFMZfJhGFDEvGDBjDDBGj8NjgrZFfgopXnM"
`endif
