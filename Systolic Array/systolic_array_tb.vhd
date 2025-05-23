library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

use work.matrix_type.all;  -- matrix_3x3_input and matrix_3x3_output now use std_logic_vector

entity systolic_array_tb is
end entity;

architecture tb of systolic_array_tb is

    -- DUT signals
    signal clk           : std_logic := '0';
    signal reset         : std_logic := '1';
    signal matrix_data   : matrix_3x3_input;
    signal matrix_weight : matrix_3x3_input;
    signal output_matrix : matrix_3x3_output;

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
            output        => output_matrix
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
        -- Apply reset
        reset <= '1';

        -- initiliase to 0
        -- Matrix A (data)
        matrix_data(0,0) <= std_logic_vector(to_unsigned(0, 8));
        matrix_data(0,1) <= std_logic_vector(to_unsigned(0, 8));
        matrix_data(0,2) <= std_logic_vector(to_unsigned(0, 8));
        matrix_data(1,0) <= std_logic_vector(to_unsigned(0, 8));
        matrix_data(1,1) <= std_logic_vector(to_unsigned(0, 8));
        matrix_data(1,2) <= std_logic_vector(to_unsigned(0, 8));
        matrix_data(2,0) <= std_logic_vector(to_unsigned(0, 8));
        matrix_data(2,1) <= std_logic_vector(to_unsigned(0, 8));
        matrix_data(2,2) <= std_logic_vector(to_unsigned(0, 8));

        -- Matrix B (weight)
        matrix_weight(0,0) <= std_logic_vector(to_unsigned(0, 8));
        matrix_weight(0,1) <= std_logic_vector(to_unsigned(0, 8));
        matrix_weight(0,2) <= std_logic_vector(to_unsigned(0, 8));
        matrix_weight(1,0) <= std_logic_vector(to_unsigned(0, 8));
        matrix_weight(1,1) <= std_logic_vector(to_unsigned(0, 8));
        matrix_weight(1,2) <= std_logic_vector(to_unsigned(0, 8));
        matrix_weight(2,0) <= std_logic_vector(to_unsigned(0, 8));
        matrix_weight(2,1) <= std_logic_vector(to_unsigned(0, 8));
        matrix_weight(2,2) <= std_logic_vector(to_unsigned(0, 8));
        

        wait for 2 * clk_period;
        reset <= '0';

        -- Matrix A (data): [1 2 3; 4 5 6; 7 8 9]
        matrix_data(0,0) <= std_logic_vector(to_unsigned(1, 8));
        matrix_data(0,1) <= std_logic_vector(to_unsigned(2, 8));
        matrix_data(0,2) <= std_logic_vector(to_unsigned(3, 8));
        matrix_data(1,0) <= std_logic_vector(to_unsigned(4, 8));
        matrix_data(1,1) <= std_logic_vector(to_unsigned(5, 8));
        matrix_data(1,2) <= std_logic_vector(to_unsigned(6, 8));
        matrix_data(2,0) <= std_logic_vector(to_unsigned(7, 8));
        matrix_data(2,1) <= std_logic_vector(to_unsigned(8, 8));
        matrix_data(2,2) <= std_logic_vector(to_unsigned(9, 8));

        -- Matrix B (weight): [9 8 7; 6 5 4; 3 2 1]
        matrix_weight(0,0) <= std_logic_vector(to_unsigned(9, 8));
        matrix_weight(0,1) <= std_logic_vector(to_unsigned(8, 8));
        matrix_weight(0,2) <= std_logic_vector(to_unsigned(7, 8));
        matrix_weight(1,0) <= std_logic_vector(to_unsigned(6, 8));
        matrix_weight(1,1) <= std_logic_vector(to_unsigned(5, 8));
        matrix_weight(1,2) <= std_logic_vector(to_unsigned(4, 8));
        matrix_weight(2,0) <= std_logic_vector(to_unsigned(3, 8));
        matrix_weight(2,1) <= std_logic_vector(to_unsigned(2, 8));
        matrix_weight(2,2) <= std_logic_vector(to_unsigned(1, 8));
        -- Let systolic array run for enough time
        wait for 300 ns;
        wait;
    end process;

end architecture;
