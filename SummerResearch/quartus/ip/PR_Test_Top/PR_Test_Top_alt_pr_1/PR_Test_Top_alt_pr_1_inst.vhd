	component PR_Test_Top_alt_pr_1 is
		port (
			clk                    : in  std_logic                     := 'X';             -- clk
			nreset                 : in  std_logic                     := 'X';             -- reset_n
			freeze                 : out std_logic;                                        -- freeze
			avmm_slave_address     : in  std_logic_vector(3 downto 0)  := (others => 'X'); -- address
			avmm_slave_read        : in  std_logic                     := 'X';             -- read
			avmm_slave_writedata   : in  std_logic_vector(31 downto 0) := (others => 'X'); -- writedata
			avmm_slave_write       : in  std_logic                     := 'X';             -- write
			avmm_slave_readdata    : out std_logic_vector(31 downto 0);                    -- readdata
			avmm_slave_waitrequest : out std_logic;                                        -- waitrequest
			irq                    : out std_logic                                         -- irq
		);
	end component PR_Test_Top_alt_pr_1;

	u0 : component PR_Test_Top_alt_pr_1
		port map (
			clk                    => CONNECTED_TO_clk,                    --        clk.clk
			nreset                 => CONNECTED_TO_nreset,                 --     nreset.reset_n
			freeze                 => CONNECTED_TO_freeze,                 --     freeze.freeze
			avmm_slave_address     => CONNECTED_TO_avmm_slave_address,     -- avmm_slave.address
			avmm_slave_read        => CONNECTED_TO_avmm_slave_read,        --           .read
			avmm_slave_writedata   => CONNECTED_TO_avmm_slave_writedata,   --           .writedata
			avmm_slave_write       => CONNECTED_TO_avmm_slave_write,       --           .write
			avmm_slave_readdata    => CONNECTED_TO_avmm_slave_readdata,    --           .readdata
			avmm_slave_waitrequest => CONNECTED_TO_avmm_slave_waitrequest, --           .waitrequest
			irq                    => CONNECTED_TO_irq                     --  interrupt.irq
		);

