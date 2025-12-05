library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.custom_types.all;

entity HW_Wrapper_Arria10 is
    port (
        CLK_50         : in  std_logic;
        BTN_RESET       : in  std_logic;
        BTW_START       : in  std_logic;

        LED_STATUS      : out std_logic_vector(7 downto 0)
        LED_DONE        : out std_logic
    );
end HW_Wrapper_Arria10;


architecture rtl of HW_Wrapper_Arria10 is

    component top_level_systolic_array is
    port (
        clk           : in  bit_1;
        reset         : in  bit_1;
        ready         : in bit_1;
        matrix_data   : in  systolic_array_matrix_input;
        matrix_weight : in  systolic_array_matrix_input;
        active_rows   : in integer;
        active_cols   : in integer;
        active_k      : in integer;
        output        : out systolic_array_matrix_output;
        cycle_count   : out integer
    );
    end component;

    -- internal signals
    signal internal_clk     : bit_1;
    signal internal_reset   : bit_1;
    signal internal_start   : bit_1;
    signal npu_ready_sig    : bit_1;


    -- hardcode a dense 32x32x32 input matrix pattern
    constant TEST_M : integer := 32;
    constant TEST_K : integer := 32;
    constant TEST_N : integer := 32;

    constant DUMMY_A_MATRIX : systolic_array_matrix_input := (others => (others => u16(1)));
    constant DUMMY_B_MATRIX : systolic_array_matrix_input := (others => (others => u16(1)));

    signal sa_output_sig : systolic_array_matrix_output;
    signal sa_cycle_count : integer;

    function to_bit(s : std_logic) return bit_1 is
    begin
        return '1' when s = '1' else '0';
    end function;

    function u16(x : integer) return bit_16 is
    begin
        return std_logic_vector(to_signed(x, 16));
    end function;

    signal start_latch : std_logic := '0';
begin
    internal_clk <= to_bit(CLK_50);
    internal_reset <= to_bit(not BTN_RESET);
    internal_start <= to_bit(BTN_START);
    
    process(CLK_50)
    begin
        if rising_edge(CLK_50) then
            if BTN_RESET = '0' then
                start_latch <= '0';
                npu_ready_sig <= '0';
            else 
                if BTN_START = '1' then
                    start_latch <= '1';
                end if;
            
                if start_latch = '1' then
                    npu_ready_sig <= '1';
                else
                    npu_ready_sig <= '0';
                end if;
            end if;
        end if;
    end process;

    -- Instantiate the top-level systolic array
    npu_core : component top_level_systolic_array
        port map (
            clk           => internal_clk,
            reset         => internal_reset,
            ready         => npu_ready_sig,
            matrix_data   => DUMMY_A_MATRIX,
            matrix_weight => DUMMY_B_MATRIX,
            active_rows   => TEST_M,
            active_cols   => TEST_N,
            active_k      => TEST_K,
            output        => sa_output_sig,
            cycle_count   => sa_cycle_count
        );
    -- Simple status LED logic
    LED_STATUS <= std_logic_vector(sa_cycle_count(0,0)(7 downto 0));
    
    LED_DONE <= '1' when sa_cycle_count > (TEST_M + TEST_N + TEST_K - 2) else '0';

end architecture rtl;