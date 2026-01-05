	PR_Test_Top u0 (
		.sdram_pll_locked_pll_locked    (_connected_to_sdram_pll_locked_pll_locked_),    //  output,   width = 1, sdram_pll_locked.pll_locked
		.ddr4_emif_oct_oct_rzqin        (_connected_to_ddr4_emif_oct_oct_rzqin_),        //   input,   width = 1,    ddr4_emif_oct.oct_rzqin
		.ddr4_emif_mem_mem_ck           (_connected_to_ddr4_emif_mem_mem_ck_),           //  output,   width = 1,    ddr4_emif_mem.mem_ck
		.ddr4_emif_mem_mem_ck_n         (_connected_to_ddr4_emif_mem_mem_ck_n_),         //  output,   width = 1,                 .mem_ck_n
		.ddr4_emif_mem_mem_a            (_connected_to_ddr4_emif_mem_mem_a_),            //  output,  width = 17,                 .mem_a
		.ddr4_emif_mem_mem_act_n        (_connected_to_ddr4_emif_mem_mem_act_n_),        //  output,   width = 1,                 .mem_act_n
		.ddr4_emif_mem_mem_ba           (_connected_to_ddr4_emif_mem_mem_ba_),           //  output,   width = 2,                 .mem_ba
		.ddr4_emif_mem_mem_bg           (_connected_to_ddr4_emif_mem_mem_bg_),           //  output,   width = 1,                 .mem_bg
		.ddr4_emif_mem_mem_cke          (_connected_to_ddr4_emif_mem_mem_cke_),          //  output,   width = 1,                 .mem_cke
		.ddr4_emif_mem_mem_cs_n         (_connected_to_ddr4_emif_mem_mem_cs_n_),         //  output,   width = 1,                 .mem_cs_n
		.ddr4_emif_mem_mem_odt          (_connected_to_ddr4_emif_mem_mem_odt_),          //  output,   width = 1,                 .mem_odt
		.ddr4_emif_mem_mem_reset_n      (_connected_to_ddr4_emif_mem_mem_reset_n_),      //  output,   width = 1,                 .mem_reset_n
		.ddr4_emif_mem_mem_par          (_connected_to_ddr4_emif_mem_mem_par_),          //  output,   width = 1,                 .mem_par
		.ddr4_emif_mem_mem_alert_n      (_connected_to_ddr4_emif_mem_mem_alert_n_),      //   input,   width = 1,                 .mem_alert_n
		.ddr4_emif_mem_mem_dqs          (_connected_to_ddr4_emif_mem_mem_dqs_),          //   inout,   width = 8,                 .mem_dqs
		.ddr4_emif_mem_mem_dqs_n        (_connected_to_ddr4_emif_mem_mem_dqs_n_),        //   inout,   width = 8,                 .mem_dqs_n
		.ddr4_emif_mem_mem_dq           (_connected_to_ddr4_emif_mem_mem_dq_),           //   inout,  width = 64,                 .mem_dq
		.ddr4_emif_mem_mem_dbi_n        (_connected_to_ddr4_emif_mem_mem_dbi_n_),        //   inout,   width = 8,                 .mem_dbi_n
		.sdram_status_local_cal_success (_connected_to_sdram_status_local_cal_success_), //  output,   width = 1,     sdram_status.local_cal_success
		.sdram_status_local_cal_fail    (_connected_to_sdram_status_local_cal_fail_),    //  output,   width = 1,                 .local_cal_fail
		.clk_clk                        (_connected_to_clk_clk_),                        //   input,   width = 1,              clk.clk
		.pr_freeze_freeze               (_connected_to_pr_freeze_freeze_),               //  output,   width = 1,        pr_freeze.freeze
		.reset_reset                    (_connected_to_reset_reset_)                     //   input,   width = 1,            reset.reset
	);

