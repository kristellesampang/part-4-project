library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.custom_types.all;

entity tb_npu_memory_system is
end entity tb_npu_memory_system;

architecture test of tb_npu_memory_system is

    -- Signals declared for the testbench
    signal clk_tb       : std_logic := '0';
    signal reset_tb     : std_logic := '1';
    signal start_transfer_cmd_tb : std_logic := '0';

    signal dma_write_data_16_sig : bit_16 := (others => '0');
    signal dma_write_data_64_sig : std_logic_vector(63 downto 0) := (others => '0');

    -- These are the names we MUST use in the process
    signal npu_data_read_tb : bit_16;
    signal npu_weight_read_tb : bit_16;
    signal buffer_is_ping_sig : std_logic;

    signal current_state_sig : state_t;

    signal npu_read_sig : bit_1 := '0';
    signal sa_output_sig : systolic_array_matrix_output;
    signal sa_cycle_count : integer;

    constant CLK_PERIOD : time := 10 ns;
    constant TEST_N : integer := 32;

    -- Conversion function for std_logic to bit_1
    function to_bit(s : std_logic) return bit_1 is
    begin
        if s = '1' then return '1'; else return '0'; end if;
    end function;

    -- Configuration specifications
    -- for DUT_Controller : BRAM_Controller_Test use entity work.BRAM_Controller_Test;
    -- for NPU_Core : top_level_systolic_array use entity work.top_level_systolic_array;



begin
    -- Clock generation
    clk_tb <= not clk_tb after CLK_PERIOD / 2;

    -- Component Instantiation: Controller
    DUT_Controller : entity work.BRAM_Controller_Test
        port map (
            clk_in => clk_tb,
            reset_in => reset_tb,
            start_transfer_cmd_in => start_transfer_cmd_tb,

            dma_write_data_16 => dma_write_data_16_sig,
            dma_write_data_64 => dma_write_data_64_sig,

            npu_data_read => npu_data_read_tb,
            npu_weight_read => npu_weight_read_tb,
            current_buffer_is_ping => buffer_is_ping_sig,

            current_state_out => current_state_sig
    );

    -- Component Instantiation: NPU Core
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

    dma_writer: process(clk_tb, reset_tb)
        variable write_data_64 : std_logic_vector(63 downto 0) := X"0000000000000001";
        variable write_data_16 : bit_16 := X"FFFF";
    begin
        if reset_tb = '1' then
            dma_write_data_16_sig <= (others => '0');
            dma_write_data_64_sig <= (others => '0');
            write_data_64 := X"0000000000000001";
            write_data_16 := X"FFFF";
        elsif rising_edge(clk_tb) then
            if current_state_sig /= S_IDLE then
                write_data_64 := std_logic_vector(unsigned(write_data_64) + 1);
                write_data_16 := bit_16(std_logic_vector(unsigned(std_logic_vector(write_data_16)) + 1));

                dma_write_data_64_sig <= write_data_64;
                dma_write_data_16_sig <= write_data_16;
            end if;
        end if;
    end process dma_writer;


    -- Test Procedure
    test_proc : process
        constant M_INIT : bit_16 := X"0001";
        constant M_NEXT : bit_16 := X"0002";
        constant W_INIT : bit_16 := X"0003";
    begin
        -- 1. Reset Pulse
        reset_tb <= '1';
        npu_read_sig <= '0';
        start_transfer_cmd_tb <= '0';

        wait for CLK_PERIOD * 2;

        reset_tb <= '0';
        wait for CLK_PERIOD * 5;

        npu_read_sig <= '1'; -- Signal NPU to start reading

        start_transfer_cmd_tb <= '1';
        wait for CLK_PERIOD;
        start_transfer_cmd_tb <= '0';

        -- 3. Verify PING data (Using correctly named _tb signals)
        wait until sa_cycle_count = 10;
        wait for CLK_PERIOD; -- Ensure data is stable
        assert (npu_data_read_tb = M_INIT)
            report "Error: NPU failed to read initial PING data (Expected 1)" severity error;
        assert (npu_weight_read_tb = W_INIT)
            report "Error: NPU failed to read initial WEIGHT data (Expected 3)" severity error;
        

        -- 5. Wait for first tile completion
        wait until sa_cycle_count = 1024;
        wait for CLK_PERIOD * 2; -- Buffer switching time

        -- 4. Signal DMA transfer complete for the PONG buffer
        start_transfer_cmd_tb <= '1';
        wait for CLK_PERIOD;
        start_transfer_cmd_tb <= '0';

        -- 6. Verify PONG data after switch
        wait until sa_cycle_count = 1030;
        assert (npu_data_read_tb = M_NEXT)
            report "Error: NPU failed to read next PONG data (Expected 2)" severity error;

        report "Memory System Test Passed: Double Buffering Verified" severity note;
        wait;
    end process test_proc;

end architecture test;