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



////////////////////////////////////////////////////////////////////////////////////////////////////////////
//  Expose extra core clocks from IOPLL
//
////////////////////////////////////////////////////////////////////////////////////////////////////////////
module altera_emif_arch_nf_pll_extra_clks #(
   parameter PLL_NUM_OF_EXTRA_CLKS = 0,
   parameter DIAG_SIM_REGTEST_MODE = 0
) (
   input  logic                                               pll_locked,            
   input  logic [8:0]                                         pll_c_counters,        
   output logic                                               pll_extra_clk_0,       
   output logic                                               pll_extra_clk_1,
   output logic                                               pll_extra_clk_2,
   output logic                                               pll_extra_clk_3,
   output logic                                               pll_extra_clk_diag_ok
);
   timeunit 1ns;
   timeprecision 1ps;
   
   logic [3:0] pll_extra_clks;
   
   // Extra core clocks to user logic.
   // These clocks are unrelated to EMIF core clock domains. The feature is intended as a
   // way to reuse EMIF PLL to generate core clocks for designs in which physical PLLs are scarce.
   assign pll_extra_clks   = pll_c_counters[8:5];
   assign pll_extra_clk_0  = pll_extra_clks[0];
   assign pll_extra_clk_1  = pll_extra_clks[1];
   assign pll_extra_clk_2  = pll_extra_clks[2];
   assign pll_extra_clk_3  = pll_extra_clks[3];
   
   // In internal test mode, generate additional counters clocked by the extra clocks
   generate
      genvar i;
      
      if (DIAG_SIM_REGTEST_MODE && PLL_NUM_OF_EXTRA_CLKS > 0) begin: test_mode
         logic [PLL_NUM_OF_EXTRA_CLKS-1:0] pll_extra_clk_diag_done;
      
         for (i = 0; i < PLL_NUM_OF_EXTRA_CLKS; ++i)
         begin : extra_clk
            logic [9:0] counter;

            always_ff @(posedge pll_extra_clks[i] or negedge pll_locked) begin
               if (~pll_locked) begin	
                  counter <= '0;
                  pll_extra_clk_diag_done[i] <= 1'b0;
               end else begin
                  if (~counter[9]) begin
                     counter <= counter + 1'b1;
                  end
                  pll_extra_clk_diag_done[i] <= counter[9];
               end
            end         
         end
         
         assign pll_extra_clk_diag_ok = &pll_extra_clk_diag_done;
         
      end else begin : normal_mode
         assign pll_extra_clk_diag_ok = 1'b1;
      end
   endgenerate
   
endmodule
`ifdef QUESTA_INTEL_OEM
`pragma questa_oem_00 "O/UL8p4fjkdiniiI1fb4ptOFsAdF7Y3Hxp7T6jxKv0MGrEyaXaFTwgtlHVroTr2ly77oTzeBnz/H2ugF2Lm64wLDISiNxLkGwFrph7cf8l5oezq71gySptz6a1i1+sH+uTBP/IhfWlKmvqzXkENOz1PT/J8+EwAOHl+z3hJ5W8p9iA7X2pLUvopxcSXzYyZFe7Y+SdxGDVeHq8Bixu/tMkSijiUVBAWYj9rL+Bb7bFHJ05MdFAKcL5QVCELrUfXXktiEDDCHLl1WefAQ6dGTsZjN0LUW7ebbD3JGWoownM6fPBIk9rt7teo/QMAG63MxQxmi8i1y9E6k7M0qLodE2E8c3TM+dypxVAysEbBlyufzN9sq3eMfQNhRHcNWaGZgVvbkeJHs5ld7HvpscGNid4/AY0SEzwJPbV0wpF4dCyHcbO4hOLaZCevbwEvLsUmGjwA4Ji95YxUbfksTG7Dji0vGDAZSouHz7SyLtE7G8cByyFFP+rn68yQaL8qlmWJU+Y1L8yTVvH8k4KHjWmsNSjEfNC4LJXroyEPtLc76PUzAD4Jc5HeooOpgNatiIkTCRCvWEU9+UeFozhJOssgiSwaKxaWFuGJVoYInaE4nUl+ty06rPzdilj3WvTf02kHSDdMBCY7rJn3ulCg6KgAfgOVboZHLXTJtZTfnKPG4d4ISn7gLrfBaAsQ2GZjdz4hGzi8QQ7ZN6FdbNlMgE37z/3dJ1KcrCFm1608mRrDQPzkZq9jtaZs8+5O/U7jk0AFeP9YkvfxKD6yA0hAJ973qx7jdayeg0yjazJ5xgGS/T+SkmStNPhx85MBmJ6hu3A85Nm5fUJ+B/Gp6ILOBh1AlGBGqSCAoPVKP+fFLCi6Vu+5g/9R+WbXOAb8TRlGeolOXRDBFsahaHJCYeqaFqVc+t4o8MofTm/Q1r+oPIXx7HSnh4FOqngnQOZltujVHv7wlV6FpQXy21PHXD58xps2zP3aY7ilu+kg9UKGaGBo7ncw8VG8SgfyAUdr/LXdXDhK5"
`endif