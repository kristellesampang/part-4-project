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


module altera_emif_arch_nf_buf_bdir_se #(
   parameter OCT_CONTROL_WIDTH = 1,
   parameter CALIBRATED_OCT = 1
) (
   inout  tri   io,
   output logic ibuf_o,
   input  logic obuf_i,
   input  logic obuf_oe,
   input  logic obuf_dtc,
   input  logic [OCT_CONTROL_WIDTH-1:0] oct_stc,
   input  logic [OCT_CONTROL_WIDTH-1:0] oct_ptc
);
   timeunit 1ns;
   timeprecision 1ps;
   
   generate
      if (CALIBRATED_OCT) 
      begin : cal_oct
         twentynm_io_ibuf ibuf(
            .i(io),
            .o(ibuf_o),
            .seriesterminationcontrol(oct_stc),
            .parallelterminationcontrol(oct_ptc),
            .dynamicterminationcontrol(),
            .ibar()
            );
            
         twentynm_io_obuf obuf (
            .i(obuf_i),
            .o(io),
            .oe(obuf_oe),
            .dynamicterminationcontrol(obuf_dtc),
            .seriesterminationcontrol(oct_stc),
            .parallelterminationcontrol(oct_ptc),
            .obar(),
            .devoe()
            );
      end else 
      begin : no_oct
         twentynm_io_ibuf ibuf(
            .i(io),
            .o(ibuf_o),
            .seriesterminationcontrol(),
            .parallelterminationcontrol(),
            .dynamicterminationcontrol(),
            .ibar()
            );
            
         twentynm_io_obuf obuf (
            .i(obuf_i),
            .o(io),
            .oe(obuf_oe),
            .dynamicterminationcontrol(),
            .seriesterminationcontrol(),
            .parallelterminationcontrol(),
            .obar(),
            .devoe()
            );      
      end
   endgenerate            
endmodule

`ifdef QUESTA_INTEL_OEM
`pragma questa_oem_00 "BUwlYfHqKCYPcXIP7xrnQ9Se5hc+WCW/arrBU5W+lFIz4wbefW0My5vX6xBCsRZeDFCX8Hl+sMTGZJoji0A9oHt2LftULu0FLRu+mk51alTJ3uomtGywN+PFSqf/fgMyVODG1WmlMI935DQeVtY8c0/LTaQ2hClpkazdo6IufJnkztEP0APWCavclOK3sEzCw6ySG+devOHMd72zKrOr9Hnpx+hVpDg9hFG/o4PmuuEIwyu2QiRo3FtGp7LYmALy97ydr2BvM/3408bCqpgekfgBx3kU+SFkWLH/1e9BuRrdOUt46/3N8Z+iPyCGPg4fgmJlCVNbd/xb7ejYUFhP10zTRZQMBkBMILvx5LkCzLamqmrmDyx6xEw4ELtA4UpLW+Qq1U3MHW3sf2UCfyxZRYZuq31dMiJiZVcpj5/rpj0Wbq+AXgowXOAyF7k3O5iFernITmM4NXYQVpUEFin4XBZwccFU4eJmxwSnzsbZ9pwcFCuKVc6OMilE5r5rpLg6vpV2uhQfzuzyZCgBAYSbxGL7WKmlKkMoWqmEKCg55O1sXvkMFMCwSEXb0BNPK6kiFR0FQd3aYr4YMjR1NrZQUQsd6iTS0TdnSZabBQn0n5aNOgqafCXm3TeNbpi1SS2tKi9lakG5X6k1dQ41xjkdFu8cvVva/yFbhbmQ5YsdHXzBqACtT8DwH2vfNxU/EzJ0Hv8txQERF2mxeKluSJNrR9+4/KVlkPFoIvu0zg/UsVe0RmPI2JwL5W2NZ1e/ojpKLT8l84DdmfrIr3MsiNQV64VahX5E+K+pjLHATZogHSKKIXped1NFfLE+6O/IgbNMQ+NTGd5SL2hfrOCGCm/xtIkfO2P+rPxW07wtbbRH0W4FBbhm0PvIcv2GqjnVAD0Mg39VtQPe2J5M5CUGLXOkSJtjp2+FXnlAfdOkKbj8PSOKDFzzoaj3qft9MM1wju2whRWl8MRl6YYkUVNfewRhG79XdUhOzNeo2IP6O+jHEUD5qFeqeZCluUfUFFYCKlFm"
`endif
