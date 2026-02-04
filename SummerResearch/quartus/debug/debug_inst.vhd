	component debug is
		port (
			source     : out std_logic_vector(0 downto 0);                    -- source
			source_clk : in  std_logic                    := 'X';             -- clk
			probe      : in  std_logic_vector(0 downto 0) := (others => 'X')  -- probe
		);
	end component debug;

	u0 : component debug
		port map (
			source     => CONNECTED_TO_source,     --    sources.source
			source_clk => CONNECTED_TO_source_clk, -- source_clk.clk
			probe      => CONNECTED_TO_probe       --     probes.probe
		);

