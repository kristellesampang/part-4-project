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


module altera_emif_arch_nf_buf_bdir_df #(
   parameter OCT_CONTROL_WIDTH = 1,
   parameter CALIBRATED_OCT = 1
) (
   inout  tri   io,
   inout  tri   iobar,
   output logic ibuf_o,
   input  logic obuf_i,
   input  logic obuf_ibar,
   input  logic obuf_oe,
   input  logic obuf_oebar,
   input  logic obuf_dtc,
   input  logic obuf_dtcbar,
   input  logic [OCT_CONTROL_WIDTH-1:0] oct_stc,
   input  logic [OCT_CONTROL_WIDTH-1:0] oct_ptc
);
   timeunit 1ns;
   timeprecision 1ps;
   
   logic pdiff_out_o;
   logic pdiff_out_obar;
   logic pdiff_out_oe;
   logic pdiff_out_oebar;
   
   generate
      if (CALIBRATED_OCT) 
      begin : cal_oct   
         logic pdiff_out_dtc;
         logic pdiff_out_dtcbar;
         
         twentynm_io_ibuf # (
            .differential_mode ("true")
         ) ibuf (
            .i(io),
            .ibar(iobar),
            .o(ibuf_o),
            .seriesterminationcontrol(oct_stc),
            .parallelterminationcontrol(oct_ptc),
            .dynamicterminationcontrol()
         );
               
         twentynm_pseudo_diff_out # (
            .feedthrough ("true")
         ) pdiff_out (
            .i(obuf_i),
            .ibar(obuf_ibar),
            .oein(obuf_oe),
            .oebin(obuf_oebar),
            .dtcin(obuf_dtc),
            .dtcbarin(obuf_dtcbar),
            .o(pdiff_out_o),
            .obar(pdiff_out_obar),
            .oeout(pdiff_out_oe),
            .oebout(pdiff_out_oebar),
            .dtc(pdiff_out_dtc),
            .dtcbar(pdiff_out_dtcbar)
         );
            
         twentynm_io_obuf obuf (
            .i(pdiff_out_o),
            .o(io),
            .oe(pdiff_out_oe),
            .dynamicterminationcontrol(pdiff_out_dtc),
            .seriesterminationcontrol(oct_stc),
            .parallelterminationcontrol(oct_ptc),
            .obar(),
            .devoe()
         );              
            
         twentynm_io_obuf obuf_bar (
            .i(pdiff_out_obar),
            .o(iobar),
            .oe(pdiff_out_oebar),
            .dynamicterminationcontrol(pdiff_out_dtcbar),
            .seriesterminationcontrol(oct_stc),
            .parallelterminationcontrol(oct_ptc),
            .obar(),
            .devoe()
         );
      end else 
      begin : no_oct
         twentynm_io_ibuf  # (
            .differential_mode ("true")
         ) ibuf (
            .i(io),
            .ibar(iobar),
            .o(ibuf_o),
            .seriesterminationcontrol(),
            .parallelterminationcontrol(),
            .dynamicterminationcontrol()
         );
               
         twentynm_pseudo_diff_out # (
            .feedthrough ("true")
         ) pdiff_out (
            .i(obuf_i),
            .ibar(obuf_ibar),
            .oein(obuf_oe),
            .oebin(obuf_oebar),
            .dtcin(),
            .dtcbarin(),
            .o(pdiff_out_o),
            .obar(pdiff_out_obar),
            .oeout(pdiff_out_oe),
            .oebout(pdiff_out_oebar),
            .dtc(),
            .dtcbar()
         );
            
         twentynm_io_obuf obuf (
            .i(pdiff_out_o),
            .o(io),
            .oe(pdiff_out_oe),
            .dynamicterminationcontrol(),
            .seriesterminationcontrol(),
            .parallelterminationcontrol(),
            .obar(),
            .devoe()
         );              
            
         twentynm_io_obuf obuf_bar (
            .i(pdiff_out_obar),
            .o(iobar),
            .oe(pdiff_out_oebar),
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
`pragma questa_oem_00 "BUwlYfHqKCYPcXIP7xrnQ9Se5hc+WCW/arrBU5W+lFIz4wbefW0My5vX6xBCsRZeDFCX8Hl+sMTGZJoji0A9oHt2LftULu0FLRu+mk51alTJ3uomtGywN+PFSqf/fgMyVODG1WmlMI935DQeVtY8c0/LTaQ2hClpkazdo6IufJnkztEP0APWCavclOK3sEzCw6ySG+devOHMd72zKrOr9Hnpx+hVpDg9hFG/o4PmuuF3dRuNNTa3HFhaOqqtXgYwtD2GvOypaW7DMU7RnmjkIEUOXg/Cx8ly7+KOLcDZECpbXotkMUglZVUm4jD2EKqxbxovKsGu+yhmjpbjunT16ueNn9aKqHwzr/IjU2UDXEfZoe+stJWXh7HCpG4O+ffIEskdQnZkGkia0g7fNWKdgCGml0OtmHmXQbw1WHs8xHdy8TYe6+DKrFyvZVW+9e0+mbeTAMRX7/89RiTObt54kD/6waPrGQJnSYdgx2LzgCm8zOPgYgHYbwPoj8yhNaLSQppyrlv9mowm/i9lJWteiWaUjfbK6tXu3KWocTTvBLHXEdjSBtFN0sM3TQYLSBIjJY8Lcpw+dtjIbYCKJESlbfkR4XGCnYsJahtELlp0XY7H/34tLawpVCYsTWB+oYf3A87H3OHMfrkWXEXe0fHyqnXUG7QWtN5pa9M7Yar+6Lf3Q4bAtky6ykK+Hzu6wD2IBSZyJfhigSCLdpvTyBLhxxGdrkw++L/ndX0PruA/ecxrckNsnqlQYWHc/fZ+PSMK3odFRpmoZ77ScHEsgoh1INZp+PxHbXQhIouM4uNhvUZTQuuHgvhbtOySzH95rW/ZegMEiuXyYVOsTzOdBmwq5lJ4MIQARY2YHky+9/Ap7+KgA7xTnplbzln4ZqozPbYMBxuoyl2kw423ZtYTdqTHo31CzDVl8yvdUA8pePzVnzKVNQO4G34zXKh+bbzVVUGZcwsoHsZ/8zm4TP/JO8IGpVlS5okLm5z/XbYVNA8ILq0tmjG0fobgn4ykQVkoA0Cm"
`endif
