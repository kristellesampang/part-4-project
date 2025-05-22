library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

use work.matrix_type.all;  -- for matrix_3x3_input, matrix_3x3_output

entity systolic_array_tb is
end entity;

architecture tb of systolic_array_tb is

    -- DUT signals
    signal clk             : std_logic := '0';
    signal reset           : std_logic := '1';
    signal matrix_data     : matrix_3x3_input;
    signal matrix_weight   : matrix_3x3_input;
    signal output_matrix   : matrix_3x3_output;
    signal cycle_counter_tb: integer := 0;  -- NEW: Expose cycle counter

    -- Clock period
    constant clk_period : time := 10 ns;

begin

    -- Instantiate DUT
    DUT: entity work.systolic_array
        port map (
            clk           => clk,
            reset         => reset,
            matrix_data   => matrix_data,
            matrix_weight => matrix_weight,
            output        => output_matrix,
            cycle_counter => cycle_counter_tb  -- NEW: connect internal counter
        );

    -- Clock generation
    clk_process: process
    begin
        while true loop
            clk <= '0';
            wait for clk_period / 2;
            clk <= '1';
            wait for clk_period / 2;
        end loop;
    end process;

    -- Stimulus
    stimulus: process
    begin
        -- Reset system
        reset <= '1';
        wait for 2 * clk_period;
        reset <= '0';

        -- Provide test input matrix A (matrix_data)
        -- [1 2 3]
        -- [4 5 6]
        -- [7 8 9]
        matrix_data(0,0) <= to_unsigned(1, 8);
        matrix_data(0,1) <= to_unsigned(2, 8);
        matrix_data(0,2) <= to_unsigned(3, 8);
        matrix_data(1,0) <= to_unsigned(4, 8);
        matrix_data(1,1) <= to_unsigned(5, 8);
        matrix_data(1,2) <= to_unsigned(6, 8);
        matrix_data(2,0) <= to_unsigned(7, 8);
        matrix_data(2,1) <= to_unsigned(8, 8);
        matrix_data(2,2) <= to_unsigned(9, 8);

        -- Provide test input matrix B (matrix_weight)
        -- [9 8 7]
        -- [6 5 4]
        -- [3 2 1]
        matrix_weight(0,0) <= to_unsigned(9, 8);
        matrix_weight(0,1) <= to_unsigned(8, 8);
        matrix_weight(0,2) <= to_unsigned(7, 8);
        matrix_weight(1,0) <= to_unsigned(6, 8);
        matrix_weight(1,1) <= to_unsigned(5, 8);
        matrix_weight(1,2) <= to_unsigned(4, 8);
        matrix_weight(2,0) <= to_unsigned(3, 8);
        matrix_weight(2,1) <= to_unsigned(2, 8);
        matrix_weight(2,2) <= to_unsigned(1, 8);

        -- Wait for array to process (e.g., 20+ cycles to see full propagation)
        wait for 500 ns;

        wait;
    end process;

end architecture;
