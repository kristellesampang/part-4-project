	component PR_Test_Top is
		port (
			sdram_pll_locked_pll_locked    : out   std_logic;                                        -- pll_locked
			ddr4_emif_oct_oct_rzqin        : in    std_logic                     := 'X';             -- oct_rzqin
			ddr4_emif_mem_mem_ck           : out   std_logic_vector(0 downto 0);                     -- mem_ck
			ddr4_emif_mem_mem_ck_n         : out   std_logic_vector(0 downto 0);                     -- mem_ck_n
			ddr4_emif_mem_mem_a            : out   std_logic_vector(16 downto 0);                    -- mem_a
			ddr4_emif_mem_mem_act_n        : out   std_logic_vector(0 downto 0);                     -- mem_act_n
			ddr4_emif_mem_mem_ba           : out   std_logic_vector(1 downto 0);                     -- mem_ba
			ddr4_emif_mem_mem_bg           : out   std_logic_vector(0 downto 0);                     -- mem_bg
			ddr4_emif_mem_mem_cke          : out   std_logic_vector(0 downto 0);                     -- mem_cke
			ddr4_emif_mem_mem_cs_n         : out   std_logic_vector(0 downto 0);                     -- mem_cs_n
			ddr4_emif_mem_mem_odt          : out   std_logic_vector(0 downto 0);                     -- mem_odt
			ddr4_emif_mem_mem_reset_n      : out   std_logic_vector(0 downto 0);                     -- mem_reset_n
			ddr4_emif_mem_mem_par          : out   std_logic_vector(0 downto 0);                     -- mem_par
			ddr4_emif_mem_mem_alert_n      : in    std_logic_vector(0 downto 0)  := (others => 'X'); -- mem_alert_n
			ddr4_emif_mem_mem_dqs          : inout std_logic_vector(7 downto 0)  := (others => 'X'); -- mem_dqs
			ddr4_emif_mem_mem_dqs_n        : inout std_logic_vector(7 downto 0)  := (others => 'X'); -- mem_dqs_n
			ddr4_emif_mem_mem_dq           : inout std_logic_vector(63 downto 0) := (others => 'X'); -- mem_dq
			ddr4_emif_mem_mem_dbi_n        : inout std_logic_vector(7 downto 0)  := (others => 'X'); -- mem_dbi_n
			sdram_status_local_cal_success : out   std_logic;                                        -- local_cal_success
			sdram_status_local_cal_fail    : out   std_logic;                                        -- local_cal_fail
			clk_clk                        : in    std_logic                     := 'X';             -- clk
			pr_freeze_freeze               : out   std_logic;                                        -- freeze
			reset_reset                    : in    std_logic                     := 'X'              -- reset
		);
	end component PR_Test_Top;

	u0 : component PR_Test_Top
		port map (
			sdram_pll_locked_pll_locked    => CONNECTED_TO_sdram_pll_locked_pll_locked,    -- sdram_pll_locked.pll_locked
			ddr4_emif_oct_oct_rzqin        => CONNECTED_TO_ddr4_emif_oct_oct_rzqin,        --    ddr4_emif_oct.oct_rzqin
			ddr4_emif_mem_mem_ck           => CONNECTED_TO_ddr4_emif_mem_mem_ck,           --    ddr4_emif_mem.mem_ck
			ddr4_emif_mem_mem_ck_n         => CONNECTED_TO_ddr4_emif_mem_mem_ck_n,         --                 .mem_ck_n
			ddr4_emif_mem_mem_a            => CONNECTED_TO_ddr4_emif_mem_mem_a,            --                 .mem_a
			ddr4_emif_mem_mem_act_n        => CONNECTED_TO_ddr4_emif_mem_mem_act_n,        --                 .mem_act_n
			ddr4_emif_mem_mem_ba           => CONNECTED_TO_ddr4_emif_mem_mem_ba,           --                 .mem_ba
			ddr4_emif_mem_mem_bg           => CONNECTED_TO_ddr4_emif_mem_mem_bg,           --                 .mem_bg
			ddr4_emif_mem_mem_cke          => CONNECTED_TO_ddr4_emif_mem_mem_cke,          --                 .mem_cke
			ddr4_emif_mem_mem_cs_n         => CONNECTED_TO_ddr4_emif_mem_mem_cs_n,         --                 .mem_cs_n
			ddr4_emif_mem_mem_odt          => CONNECTED_TO_ddr4_emif_mem_mem_odt,          --                 .mem_odt
			ddr4_emif_mem_mem_reset_n      => CONNECTED_TO_ddr4_emif_mem_mem_reset_n,      --                 .mem_reset_n
			ddr4_emif_mem_mem_par          => CONNECTED_TO_ddr4_emif_mem_mem_par,          --                 .mem_par
			ddr4_emif_mem_mem_alert_n      => CONNECTED_TO_ddr4_emif_mem_mem_alert_n,      --                 .mem_alert_n
			ddr4_emif_mem_mem_dqs          => CONNECTED_TO_ddr4_emif_mem_mem_dqs,          --                 .mem_dqs
			ddr4_emif_mem_mem_dqs_n        => CONNECTED_TO_ddr4_emif_mem_mem_dqs_n,        --                 .mem_dqs_n
			ddr4_emif_mem_mem_dq           => CONNECTED_TO_ddr4_emif_mem_mem_dq,           --                 .mem_dq
			ddr4_emif_mem_mem_dbi_n        => CONNECTED_TO_ddr4_emif_mem_mem_dbi_n,        --                 .mem_dbi_n
			sdram_status_local_cal_success => CONNECTED_TO_sdram_status_local_cal_success, --     sdram_status.local_cal_success
			sdram_status_local_cal_fail    => CONNECTED_TO_sdram_status_local_cal_fail,    --                 .local_cal_fail
			clk_clk                        => CONNECTED_TO_clk_clk,                        --              clk.clk
			pr_freeze_freeze               => CONNECTED_TO_pr_freeze_freeze,               --        pr_freeze.freeze
			reset_reset                    => CONNECTED_TO_reset_reset                     --            reset.reset
		);

