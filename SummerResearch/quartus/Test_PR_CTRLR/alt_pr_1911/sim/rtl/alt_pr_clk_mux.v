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


`timescale 1ns/1ns
// Clock Multiplexing module from Synthesis Cookbook:
//      https://www.altera.com/content/dam/altera-www/global/en_US/pdfs/literature/manual/stx_cookbook.pdf
/////////////////////////////////////////////////////////////////////////////

module alt_pr_clk_mux (clk,clk_select,clk_out);

parameter NUM_CLOCKS = 4;
parameter USE_FOLLOWERS = 1'b0;

input [NUM_CLOCKS-1:0] clk;
input [NUM_CLOCKS-1:0] clk_select; // one hot
output clk_out;

genvar i;

reg [NUM_CLOCKS-1:0] ena_r0;
reg [NUM_CLOCKS-1:0] ena_r1;
reg [NUM_CLOCKS-1:0] ena_r2;
wire [NUM_CLOCKS-1:0] qualified_sel;

// A LUT can glitch when multiple inputs slew 
// simultaneously (in theory indepently of the function).  
// Insert a hard LCELL to prevent the unrelated clocks
// from appearing on the same LUT.

wire [NUM_CLOCKS-1:0] gated_clks /* synthesis keep */;

initial begin
  ena_r0 = 0;
  ena_r1 = 0;
  ena_r2 = 0;
end

generate
for (i=0; i<NUM_CLOCKS; i=i+1) 
begin : lp0

  wire [NUM_CLOCKS-1:0] tmp_mask;
  assign tmp_mask = {NUM_CLOCKS{1'b1}} ^ (1 << i);

  assign qualified_sel[i] = clk_select[i] & 
			(~|(ena_r2 & tmp_mask));
  
  always @(posedge clk[i]) begin
    ena_r0[i] <= qualified_sel[i];    	
    ena_r1[i] <= ena_r0[i];    	
  end
  
  always @(negedge clk[i]) begin
    ena_r2[i] <= ena_r1[i];    	
  end

  if (USE_FOLLOWERS) begin
     wire cf_out;
     clock_follow cf (.clk_in(clk[i]),.clk_out(cf_out));
	 assign gated_clks[i] = cf_out & ena_r2[i];
  end
  else begin
     assign gated_clks[i] = clk[i] & ena_r2[i];
  end
end
endgenerate

// these will not exhibit simultaneous toggle by construction.
assign clk_out = |gated_clks;

endmodule
`ifdef QUESTA_INTEL_OEM
`pragma questa_oem_00 "mXd8TZA7wcZD1M13kc63+RPfQcbqACFa9i0aqHtSQ62nFR2tUIuo2NzwpI/4cEyddIaPImJhX2CebOmYVV5f401JpALFknks4OMLrt70wgtKfnPjlusCxrHWEBHooOs6pkJXBTjC6oUhD2WVo05l5UBMB3Bob+qvd+BE/CfQQqbe8vYxUQkLM+Zgo0aA7Fpi0A6ZVpcWA/DDjPpZK8kPgz9Vfztbmx0x1RvFi9qEstIo11fsBHWS9y5VrMekkvrLdDJOEQyNbeAN1WO/ytc3vFXbl6brr/4eveMMPPvbbn7DkPJ1I7VvK7K2TNiPuPMYT32Wbe2F2AKR6BJSOiBVTS1/hnp5QsmtTLMecOej4/9IAwvWCkfmNJWsV6pEdVtjMi3z/MyYA2q43buiHDGpFs+48gL7Z+ssXPJo5FewYZT5lgjdTFC73rzOc3f9OJ+xgpgUAEk3EyubMazrf+Vxgt4Iw6S17u+X8/acYOZCfx+PTQWeBKYF4u4JR1RMNESo5vvKrDlW1X9mOHlMD61Gd9AOqJp4kpmJ3F7+UAgDG+WY4xZfjMg4aCuhFjyVDbqcHN41XRyyXFUnQLnL1WJEw8AhZN7F6jHlKhMh02P6fOuzizhCHDPWrLmRH5wDRbsNWmsgeBHYM5/6fHkky/GDVch8M77M5HFWVNOCtzPeQ6j+u25lBk+boA1mqNiGa05LBUN8ymMtq7dcc/tINBM6rXO+ZzDx2rR/Mp89OFQQMCsbCTCsBZcM63bXyR4VATmxxfn8qSsq+zPDSSHNMsUKRkcxJLTVUVUHAvEQPtY3YJA7lFg5aRc7oXpS218bz9hmU/4FTDC20oZOSEPw/+X8HBE4gQRGrEGPf+B2h+/yPQqnQw7zEzXKAIbVa+Ud5Umt+bfeDtWo+LlqwU45nmptd1qaHMl3avn7WTlRXHaNZBQV0fPPTQqyST2Np4m+YVdLcYEpQr82J8plc+KqSQHlusIh7zJ2//cvMvrOAeD/qY5zOMWb+CSZGolESxxvrNTK"
`endif