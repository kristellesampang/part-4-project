library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.custom_types.all;

entity tb_npu_memory_system is
end entity tb_npu_memory_system;

architecture test of tb_npu_memory_system is

    signal clk_tb       : std_logic := '0';
    signal reset_tb     : std_logic := '1';
    signal start_transfer_cmd_tb : std_logic := '0';

    signal npu_data_read_tb : bit_16;
    signal npu_weight_read_tb : bit_16;
    signal buffer_is_ping_sig : std_logic;

    signal npu_read_sig : bit_1 := '0';
    signal sa_output_sig : systolic_array_matrix_output;
    signal sa_cycle_count : integer;

    constant CLK_PERIOD : time := 10 ns;
    constant TEST_N : integer := 32;

    function to_bit(s : std_logic) return bit_1 is
    begin
        return '1' when s = '1' else '0';
    end function;

    for DUT_Controller : BRAM_Controller_Test use entity work.BRAM_Controller_Test;
    for NPU_Core : top_level_systolic_array use entity work.top_level_systolic_array;

begin
    clk_tb <= not clk_tb after CLK_PERIOD / 2;

    DUT_Controller : entity work.BRAM_Controller_Test
        port map (
            clk_in => clk_tb,
            reset_in => reset_tb,
            start_transfer_cmd_in => start_transfer_cmd_tb,
            npu_data_read => npu_data_read_tb,
            npu_weight_read => npu_weight_read_tb,
            current_buffer_is_ping => buffer_is_ping_sig
        );

    NPU_Core : entity work.top_level_systolic_array
        port map (
            clk => to_bit(clk_tb),
            reset => to_bit(reset_tb),
            ready => npu_read_sig,
            matrix_data => (others => (others => (others => '0'))),
            matrix_weight => (others => (others => (others => '0'))),
            active_rows => TEST_N,
            active_cols => TEST_N,
            active_k => TEST_N,
            output => sa_output_sig,
            cycle_count => sa_cycle_count
        );

    test_proc : process
        constant M_INIT : bit_16 := X"0001";
        constant M_NEXT : bit_16 := X"0002";
        constant W_INIT : bit_16 := X"0003";
    begin

        wait for CLK_PERIOD * 2;
        reset_tb <= '0';
        wait for CLK_PERIOD * 5;
        reset_tb <= '1';
        wait for CLK_PERIOD * 5;

        npu_ready_sig <= '1';
        wait for CLK_PERIOD;

        wait until sa_cycle_count = 10;
        assert (npu_data_read_sig = M_INIT)
            report "Error: NPU failed to read initial PING data (Expected 1)" severity error;
        assert (npu_weight_read_sig = W_INIT)
            report "Error: NPU failed to read initial WEIGHT data (Expected 3)" severity error;
        
        start_transfer_cmd_tb <= '1';
        wait for CLK_PERIOD;
        start_transfer_cmd_tb <= '0';

        wait until sa_cycle_count = 1024;

        wait for CLK_PERIOD * 2;

        wait until sa_cycle_count = 1030;
        assert (npu_data_read_sig = M_NEXT)
            report "Error: NPU failed to read next PONG data (Expected 2)" severity error;
        report "Memory System Test Passed: Double Buffering Verified" severity note;
        wait;
    end process test_proc;

end architecture test;