module PR_Test_Top (
		output wire        sdram_pll_locked_pll_locked,    // sdram_pll_locked.pll_locked
		input  wire        ddr4_emif_oct_oct_rzqin,        //    ddr4_emif_oct.oct_rzqin
		output wire [0:0]  ddr4_emif_mem_mem_ck,           //    ddr4_emif_mem.mem_ck
		output wire [0:0]  ddr4_emif_mem_mem_ck_n,         //                 .mem_ck_n
		output wire [16:0] ddr4_emif_mem_mem_a,            //                 .mem_a
		output wire [0:0]  ddr4_emif_mem_mem_act_n,        //                 .mem_act_n
		output wire [1:0]  ddr4_emif_mem_mem_ba,           //                 .mem_ba
		output wire [0:0]  ddr4_emif_mem_mem_bg,           //                 .mem_bg
		output wire [0:0]  ddr4_emif_mem_mem_cke,          //                 .mem_cke
		output wire [0:0]  ddr4_emif_mem_mem_cs_n,         //                 .mem_cs_n
		output wire [0:0]  ddr4_emif_mem_mem_odt,          //                 .mem_odt
		output wire [0:0]  ddr4_emif_mem_mem_reset_n,      //                 .mem_reset_n
		output wire [0:0]  ddr4_emif_mem_mem_par,          //                 .mem_par
		input  wire [0:0]  ddr4_emif_mem_mem_alert_n,      //                 .mem_alert_n
		inout  wire [7:0]  ddr4_emif_mem_mem_dqs,          //                 .mem_dqs
		inout  wire [7:0]  ddr4_emif_mem_mem_dqs_n,        //                 .mem_dqs_n
		inout  wire [63:0] ddr4_emif_mem_mem_dq,           //                 .mem_dq
		inout  wire [7:0]  ddr4_emif_mem_mem_dbi_n,        //                 .mem_dbi_n
		output wire        sdram_status_local_cal_success, //     sdram_status.local_cal_success
		output wire        sdram_status_local_cal_fail,    //                 .local_cal_fail
		input  wire        clk_clk,                        //              clk.clk
		output wire        pr_freeze_freeze,               //        pr_freeze.freeze
		input  wire        reset_reset                     //            reset.reset
	);
endmodule

