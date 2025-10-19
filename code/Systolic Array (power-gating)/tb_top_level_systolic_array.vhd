library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.custom_types.all;

entity tb_top_level_systolic_array is
end tb_top_level_systolic_array;

architecture sim of tb_top_level_systolic_array is
    signal clock           : bit_1 := '0';
    signal reset         : bit_1 := '1';
    signal matrix_A      : systolic_array_matrix_input := (others => (others => (others => '0')));
    signal matrix_B      : systolic_array_matrix_input := (others => (others => (others => '0')));
    signal result_C      : systolic_array_matrix_output;
    signal cycle_counter : integer;
    signal size_tb       : integer;  -- Test size variable
    signal done          : bit_1;
    signal sim_finished : bit_1 := '0';
    
    constant CLK_PER : time := 20 ns;
begin
    -- Clock generation - simple concurrent assignment
    clock <= not clock after CLK_PER / 2;

    -- Instantiate DUT
    DUT: entity work.top_level_systolic_array
    port map (
        clk           => clock,
        reset         => reset,
        size          => size_tb,
        matrix_data   => matrix_A,
        matrix_weight => matrix_B,
        output        => result_C,
        cycle_count   => cycle_counter,
        done          => done
    );

    -- Test process for matrix sizes 1x1 to 8x8
    stimulus_process: process
        variable val_a : integer;
        variable val_b : integer;
    begin
        -- Initial wait for clock to stabilize
        wait for 10 * CLK_PER;
        
        for size in 1 to 8 loop
            size_tb <= size;  -- Update size signal for DUT
            -- Reset values for each size (to match MATLAB)
            val_a := 1;
            val_b := 10;

            -- Clear all matrix positions first
            for i in 0 to N-1 loop
                for j in 0 to N-1 loop
                    matrix_A(i,j) <= (others => '0');
                    matrix_B(i,j) <= (others => '0');
                end loop;
            end loop;

            -- Load matrix_A with a variable number of rows (size x N)
            for i in 0 to size-1 loop
                for j in 0 to N-1 loop
                    matrix_A(i,j) <= std_logic_vector(to_unsigned(val_a, 8));
                    val_a := val_a + 1;
                end loop;
            end loop;

            -- Load matrix_B as a full 8x8 matrix in every test
            for i in 0 to N-1 loop
                for j in 0 to N-1 loop
                    matrix_B(i,j) <= std_logic_vector(to_unsigned(val_b, 8));
                    val_b := val_b + 1;
                end loop;
            end loop;

            -- Apply reset
            reset <= '1';
            wait for 5 * CLK_PER;
            reset <= '0';
            
            -- Wait for matrix loading and computation to complete
            -- Control unit needs time to process and systolic array needs computation time
            wait for (3 * N) * CLK_PER;
            sim_finished <= '1';

            -- Report results for current size
            report "=== Results for size " & integer'image(size) & "x" & integer'image(size) & " ===";
            for i in 0 to size-1 loop
                for j in 0 to size-1 loop
                    report "Result[" & integer'image(i) & "," & integer'image(j) & "] = " &
                           integer'image(to_integer(unsigned(result_C(i,j))));
                end loop;
            end loop;

            wait for 5 * CLK_PER;
            sim_finished <= '0';
            
        end loop;
        

        report "Simulation completed successfully";
        -- sim_finished <= true;  -- Stop the clock
        wait;
    end process;

end sim;
