	component NonDPR is
		port (
			clk_clk     : in std_logic := 'X'; -- clk
			reset_reset : in std_logic := 'X'  -- reset
		);
	end component NonDPR;

	u0 : component NonDPR
		port map (
			clk_clk     => CONNECTED_TO_clk_clk,     --   clk.clk
			reset_reset => CONNECTED_TO_reset_reset  -- reset.reset
		);

