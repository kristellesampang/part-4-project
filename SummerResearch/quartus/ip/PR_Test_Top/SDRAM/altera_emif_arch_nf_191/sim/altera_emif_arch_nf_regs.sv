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
// This module handles the creation of a conditional register stage
// This module may be used to implement a synchronizer (with properly selected
// REGISTER value)
///////////////////////////////////////////////////////////////////////////////

// The following ensures that the register stage isn't synthesized into
// RAM-based shift-regs (especially if customer logic implements another follow-on
// pipeline stage). RAM-based shift-regs can degrade timing for C2P/P2C transfers.
(* altera_attribute = "-name AUTO_SHIFT_REGISTER_RECOGNITION OFF" *)

 module altera_emif_arch_nf_regs #(
   parameter REGISTER       = 0,
   parameter WIDTH          = 0
) (
   input  logic              clk,
   input  logic              reset_n,
   input  logic [WIDTH-1:0]  data_in,
   output logic [WIDTH-1:0]  data_out
) /* synthesis dont_merge */;
   timeunit 1ns;
   timeprecision 1ps;

   generate
      if (REGISTER == 0) begin
         assign data_out = data_in;

      end else begin
         logic [WIDTH-1:0] sr_out;
         always_ff @(posedge clk or negedge reset_n) begin
            if (~reset_n) begin
               sr_out <= '0;
            end else begin
               sr_out <= data_in;
            end
         end
         assign data_out = sr_out;
      end
   endgenerate
endmodule
`ifdef QUESTA_INTEL_OEM
`pragma questa_oem_00 "O/UL8p4fjkdiniiI1fb4ptOFsAdF7Y3Hxp7T6jxKv0MGrEyaXaFTwgtlHVroTr2ly77oTzeBnz/H2ugF2Lm64wLDISiNxLkGwFrph7cf8l5oezq71gySptz6a1i1+sH+uTBP/IhfWlKmvqzXkENOz1PT/J8+EwAOHl+z3hJ5W8p9iA7X2pLUvopxcSXzYyZFe7Y+SdxGDVeHq8Bixu/tMkSijiUVBAWYj9rL+Bb7bFFRnVhEZgc6MJP6bioU7MjIF7HehIJ44BrzSWKKjTHu5umrmjNqvdfnln15wpT652Ktg8pCDC+zluXLTcRvtceQvdtQZJoGmwZt7hCKrMOhHo0evxZvcZ4Ijh7KWqsF/2wMjZlm5NdcaJMHy50HGyrsbf42O99ybAaCzxwx5i0Q5ufX550HhhjFe2GcsCaww2Iz6Nh9mqcXzPdEr4+99jRSZeP0Bd5E8MFaHpMQwCYiV0FCLFQVO8HxKv15t4D7qK1YNOqNdAMXZrn8bQtvoz+ZwV8T3TKjo7Ls3CxBfuVa0PnL58qHI9WxdIx778vmp6kbLcgcWgz8GHvReZtLGua2Wg5KspyJTYhLVdxweqwXas1Y9OL7jAuEl9wiOhFprGqwPatn2b1whA5JcU25E7sFYCO0pGEEYW3aHqBz88hDK9o+lohppQRoGelseTevHZb62qfR3Y62xLCNF6x87G9MSOl9ff41otQ0hYA9+AveMCLfqz2snmz7aQw1eXTG79VRHKRQM46zKuBsnSOvJPIyz58N1+hJ+3sf0U4XOUKoHXZoed6V398aFgSyS2y/qJ/aQBfL+BnOsag0lpyjx1VzmywjIVgr7keLUZBSe16y5SuMswpIUPQYS0IW/Vj7G8GmoeOtiKETbgd3+G9wCequKyB4Gk+j9x4S608pOHXgiG5EsNOU10a4xKUeGBA5i+Zh3AbwwQRPD0OChbdPkAdOOiM45BrrFZy/ZBLOxX3CO4DZ6BfDWK45a7o9IOY+5tCTMbCx8Y4mVj5glsE1aY4u"
`endif