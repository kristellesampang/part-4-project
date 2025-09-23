library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.custom_types.all;

entity tb_top_level_systolic_array is
end tb_top_level_systolic_array;

architecture sim of tb_top_level_systolic_array is
    signal clk           : bit_1 := '0';
    signal reset         : bit_1 := '1';
    signal matrix_A      : systolic_array_matrix_input := (others => (others => (others => '0')));
    signal matrix_B      : systolic_array_matrix_input := (others => (others => (others => '0')));
    signal result_C      : systolic_array_matrix_output;
    signal cycle_counter : integer;
    
    constant CLK_PER : time := 20 ns;
begin
    -- Clock generation
    clk <= not clk after CLK_PER / 2;

    -- Instantiate DUT
    DUT: entity work.top_level_systolic_array
    port map (
        clk           => clk,
        reset         => reset,
        matrix_data   => matrix_A,
        matrix_weight => matrix_B,
        output        => result_C,
        cycle_count   => cycle_counter
    );

    -- Test process for matrix sizes 1x1 to 8x8
    process
        variable val_a : integer;
        variable val_b : integer;
    begin
        for size in 1 to 8 loop
            -- Reset values for each size (to match MATLAB)
            val_a := 1;
            val_b := 10;

            -- Load matrix_A and matrix_B with sample values
            for i in 0 to size-1 loop
                for j in 0 to size-1 loop
                    matrix_A(i,j) <= std_logic_vector(to_unsigned(val_a, 8));
                    matrix_B(i,j) <= std_logic_vector(to_unsigned(val_b, 8));
                    val_a := val_a + 1;
                    val_b := val_b + 1;
                end loop;
            end loop;

            -- Reset
            reset <= '1';
            wait for 2 * CLK_PER;
            reset <= '0';

            -- Wait long enough for computation to complete
            wait for (2 * size + size + 2) * CLK_PER;


            wait for 5 * CLK_PER;
        end loop;

        wait;
    end process;

end sim;