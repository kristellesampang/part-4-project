library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.custom_types.all;

entity tb_npu_wrapper is
end tb_npu_wrapper;

architecture sim of tb_npu_wrapper is
    signal clk           : std_logic := '0';
    signal reset_n       : std_logic := '0';

    -- Avalon-MM Slave Interface
    signal avs_address   : std_logic_vector(4 downto 0) := (others => '0');
    signal avs_write     : std_logic := '0';
    signal avs_writedata : std_logic_vector(31 downto 0) := (others => '0');
    signal avs_read      : std_logic := '0';
    signal avs_readdata  : std_logic_vector(31 downto 0);
    signal avs_waitrequest : std_logic;

    -- Avalon-MM Master Interfaces
    signal avm_act_address    : std_logic_vector(31 downto 0);
    signal avm_act_read       : std_logic;
    signal avm_act_readdata   : std_logic_vector(31 downto 0);
    signal avm_act_waitreq    : std_logic := '0';

    signal avm_weight_address    : std_logic_vector(31 downto 0);
    signal avm_weight_read       : std_logic;
    signal avm_weight_readdata   : std_logic_vector(31 downto 0);
    signal avm_weight_waitreq    : std_logic := '0';

    signal avm_out_address    : std_logic_vector(31 downto 0);
    signal avm_out_write      : std_logic;
    signal avm_out_writedata  : std_logic_vector(31 downto 0);
    signal avm_out_waitreq    : std_logic := '0';

    constant clk_period : time := 10 ns;

begin

    DUT: entity work.npu_system_wrapper
        port map (
            clk => clk,
            reset_n => reset_n,
            avs_address => avs_address,
            avs_write => avs_write,
            avs_writedata => avs_writedata,
            avs_read => avs_read,
            avs_readdata => avs_readdata,
            avs_waitrequest => avs_waitrequest,
            avm_act_address => avm_act_address,
            avm_act_read => avm_act_read,
            avm_act_readdata => avm_act_readdata,
            avm_act_waitreq => avm_act_waitreq,
            avm_weight_address => avm_weight_address,
            avm_weight_read => avm_weight_read,
            avm_weight_readdata => avm_weight_readdata,
            avm_weight_waitreq => avm_weight_waitreq,
            avm_out_address => avm_out_address,
            avm_out_write => avm_out_write,
            avm_out_writedata => avm_out_writedata,
            avm_out_waitreq => avm_out_waitreq
        );

    -- Clock generation
    clk <= not clk after clk_period / 2;

    process
    begin
        -- 1. Reset
        reset_n <= '0';
        wait for 50 ns;
        reset_n <= '1';
        wait for 50 ns;

        -- 2. Write Config Registers (Using Byte Offsets for SYMBOL addressing)
        -- Address 0x4 = M = 4
        avs_address <= "00100"; avs_writedata <= x"00000004"; avs_write <= '1';
        wait until rising_edge(clk);
        -- Address 0x8 = N = 4
        avs_address <= "01000"; avs_writedata <= x"00000004"; avs_write <= '1';
        wait until rising_edge(clk);
        -- Address 0xC = K = 1
        avs_address <= "01100"; avs_writedata <= x"00000001"; avs_write <= '1';
        wait until rising_edge(clk);
        avs_write <= '0';
        
        wait for 20 ns;

        -- 3. Trigger NPU (Start bit at 0x0)
        avs_address <= "00000"; avs_writedata <= x"00000001"; avs_write <= '1';
        wait until rising_edge(clk);
        avs_write <= '0';

        -- 4. Monitor Loop
        wait for 2000 ns; -- Observe waves for FETCH and EXECUTE transitions
        
        report "Simulation Complete. Check wave for register updates and FSM state.";
        wait;
    end process;

    -- Simulate a RAM response without the RAM component for simplicity
    process(clk)
    begin
        if rising_edge(clk) then
            if avm_act_read = '1' then
                avm_act_readdata <= avm_act_address;
            end if;
            if avm_weight_read = '1' then
                avm_weight_readdata <= x"0000000A";
            end if;
        end if;
    end process;
end architecture;