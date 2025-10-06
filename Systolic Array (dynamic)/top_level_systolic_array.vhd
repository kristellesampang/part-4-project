library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.custom_types.all;

entity top_level_systolic_array is
    port (
        clk               : in  std_logic;
        reset             : in  std_logic;
        start             : in  std_logic;
        done              : out std_logic;

        -- Control signals from the Testbench FSM
        active_rows       : in  integer;
        active_k          : in  integer;
        active_cols       : in  integer;

        -- Full matrices, provided by the Testbench FSM after loading from BRAM
        matrix_data_in    : in  systolic_array_matrix_input;
        matrix_weight_in  : in  systolic_array_matrix_input;

        -- The final result
        matrix_result_out : out systolic_array_matrix_output
    );
end entity top_level_systolic_array;

architecture rtl of top_level_systolic_array is

    -- Internal signals connecting the CU and the SA
    signal w_data_shift   : input_shift_matrix;
    signal w_weight_shift : input_shift_matrix;
    signal w_pe_mask      : enabled_PE_matrix;
    signal w_cycle_count  : integer;
    signal w_internal_reset : std_logic;

begin

    -- The internal reset for the CU and SA is triggered by the 'start' pulse
    -- combined with the global reset.
    w_internal_reset <= start or reset;

    -- Instantiate your Control Unit
    inst_cu : entity work.control_unit
        port map (
            clk             => clk,
            reset           => w_internal_reset,
            done            => open,
            matrix_data     => matrix_data_in,
            matrix_weight   => matrix_weight_in,
            active_rows        => active_rows,
            active_k        => active_k,
            active_cols        => active_cols,
            data_shift      => w_data_shift,
            weight_shift    => w_weight_shift,
            cycle_count     => open,        -- change back to w_cycle_count if needed
            PE_enabled_mask => w_pe_mask
        );

    -- Instantiate your Systolic Array
    inst_sa : entity work.systolic_array
        port map (
            clk          => clk,
            reset        => w_internal_reset,
            data_shift   => w_data_shift,
            weight_shift => w_weight_shift,
            enabled_PE   => w_pe_mask,
            output       => matrix_result_out
        );

    -- -- Done signal generation logic
    -- process(clk)
    -- begin
    --     if rising_edge(clk) then
    --         if (reset = '1') then
    --             done <= '0';
    --         elsif (start = '1') then
    --             done <= '0'; -- De-assert done when a new computation starts
    --         -- The computation is done when the cycle counter reaches the total latency
    --         elsif (w_cycle_count >= (active_rows + active_k + active_cols - 2)) then
    --             done <= '1';
    --         end if;
    --     end if;
    -- end process;

end architecture rtl;