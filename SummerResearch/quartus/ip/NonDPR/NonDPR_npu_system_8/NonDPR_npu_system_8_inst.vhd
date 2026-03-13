	component NonDPR_npu_system_8 is
		port (
			clk                 : in  std_logic                     := 'X';             -- clk
			reset_n             : in  std_logic                     := 'X';             -- reset_n
			avs_address         : in  std_logic_vector(4 downto 0)  := (others => 'X'); -- address
			avs_write           : in  std_logic                     := 'X';             -- write
			avs_writedata       : in  std_logic_vector(31 downto 0) := (others => 'X'); -- writedata
			avs_read            : in  std_logic                     := 'X';             -- read
			avs_readdata        : out std_logic_vector(31 downto 0);                    -- readdata
			avs_waitrequest     : out std_logic;                                        -- waitrequest
			avm_act_address     : out std_logic_vector(31 downto 0);                    -- address
			avm_act_read        : out std_logic;                                        -- read
			avm_act_readdata    : in  std_logic_vector(31 downto 0) := (others => 'X'); -- readdata
			avm_act_waitreq     : in  std_logic                     := 'X';             -- waitrequest
			avm_weight_address  : out std_logic_vector(31 downto 0);                    -- address
			avm_weight_read     : out std_logic;                                        -- read
			avm_weight_readdata : in  std_logic_vector(31 downto 0) := (others => 'X'); -- readdata
			avm_weight_waitreq  : in  std_logic                     := 'X';             -- waitrequest
			avm_out_address     : out std_logic_vector(31 downto 0);                    -- address
			avm_out_write       : out std_logic;                                        -- write
			avm_out_writedata   : out std_logic_vector(31 downto 0);                    -- writedata
			avm_out_waitreq     : in  std_logic                     := 'X'              -- waitrequest
		);
	end component NonDPR_npu_system_8;

	u0 : component NonDPR_npu_system_8
		port map (
			clk                 => CONNECTED_TO_clk,                 --          clock.clk
			reset_n             => CONNECTED_TO_reset_n,             --          reset.reset_n
			avs_address         => CONNECTED_TO_avs_address,         -- avalon_slave_0.address
			avs_write           => CONNECTED_TO_avs_write,           --               .write
			avs_writedata       => CONNECTED_TO_avs_writedata,       --               .writedata
			avs_read            => CONNECTED_TO_avs_read,            --               .read
			avs_readdata        => CONNECTED_TO_avs_readdata,        --               .readdata
			avs_waitrequest     => CONNECTED_TO_avs_waitrequest,     --               .waitrequest
			avm_act_address     => CONNECTED_TO_avm_act_address,     --            act.address
			avm_act_read        => CONNECTED_TO_avm_act_read,        --               .read
			avm_act_readdata    => CONNECTED_TO_avm_act_readdata,    --               .readdata
			avm_act_waitreq     => CONNECTED_TO_avm_act_waitreq,     --               .waitrequest
			avm_weight_address  => CONNECTED_TO_avm_weight_address,  --         weight.address
			avm_weight_read     => CONNECTED_TO_avm_weight_read,     --               .read
			avm_weight_readdata => CONNECTED_TO_avm_weight_readdata, --               .readdata
			avm_weight_waitreq  => CONNECTED_TO_avm_weight_waitreq,  --               .waitrequest
			avm_out_address     => CONNECTED_TO_avm_out_address,     --            out.address
			avm_out_write       => CONNECTED_TO_avm_out_write,       --               .write
			avm_out_writedata   => CONNECTED_TO_avm_out_writedata,   --               .writedata
			avm_out_waitreq     => CONNECTED_TO_avm_out_waitreq      --               .waitrequest
		);

