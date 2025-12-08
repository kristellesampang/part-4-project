	component DataBuffer2 is
		port (
			data_a    : in  std_logic_vector(15 downto 0) := (others => 'X'); -- datain_a
			q_a       : out std_logic_vector(15 downto 0);                    -- dataout_a
			data_b    : in  std_logic_vector(15 downto 0) := (others => 'X'); -- datain_b
			q_b       : out std_logic_vector(15 downto 0);                    -- dataout_b
			address_a : in  std_logic_vector(9 downto 0)  := (others => 'X'); -- address_a
			address_b : in  std_logic_vector(9 downto 0)  := (others => 'X'); -- address_b
			wren_a    : in  std_logic                     := 'X';             -- wren_a
			wren_b    : in  std_logic                     := 'X';             -- wren_b
			clock     : in  std_logic                     := 'X';             -- clk
			freeze    : in  std_logic                     := 'X';             -- freeze
			enable    : in  std_logic                     := 'X';             -- enable
			aclr      : in  std_logic                     := 'X'              -- reset
		);
	end component DataBuffer2;

	u0 : component DataBuffer2
		port map (
			data_a    => CONNECTED_TO_data_a,    --    data_a.datain_a
			q_a       => CONNECTED_TO_q_a,       --       q_a.dataout_a
			data_b    => CONNECTED_TO_data_b,    --    data_b.datain_b
			q_b       => CONNECTED_TO_q_b,       --       q_b.dataout_b
			address_a => CONNECTED_TO_address_a, -- address_a.address_a
			address_b => CONNECTED_TO_address_b, -- address_b.address_b
			wren_a    => CONNECTED_TO_wren_a,    --    wren_a.wren_a
			wren_b    => CONNECTED_TO_wren_b,    --    wren_b.wren_b
			clock     => CONNECTED_TO_clock,     --     clock.clk
			freeze    => CONNECTED_TO_freeze,    --    freeze.freeze
			enable    => CONNECTED_TO_enable,    --    enable.enable
			aclr      => CONNECTED_TO_aclr       --      aclr.reset
		);

