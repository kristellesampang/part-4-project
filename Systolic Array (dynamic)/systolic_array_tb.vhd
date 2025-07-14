library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.custom_types.all;

entity tb_systolic_array is
end tb_systolic_array;

architecture sim of tb_systolic_array is
    signal clk           : bit_1 := '0';
    signal reset         : bit_1 := '1';
    signal matrix_A      : matrix_3x3;
    signal matrix_B      : matrix_3x3;
    signal result_C      : matrix_3x3_output;
    signal cycle_counter : integer;

    constant CLK_PER : time := 20 ns;
begin
    -- Clock generation
    clk <= not clk after CLK_PER / 2;

    -- Instantiate DUT
    DUT: entity work.systolic_array
    port map (
        clk           => clk,
        reset         => reset,
        matrix_data   => matrix_A,
        matrix_weight => matrix_B,
        output        => result_C,
        cycle_counter => cycle_counter
    );

    -- Test process
    process
    begin
        -- Load A = [0 1 2; 3 4 5; 6 7 8]
        -- Load D = [69 44 26; 72 96 15; 89 22 11]
        matrix_A(0,0) <= std_logic_vector(to_unsigned(69, 8)); 
        matrix_A(0,1) <= std_logic_vector(to_unsigned(44, 8)); 
        matrix_A(0,2) <= std_logic_vector(to_unsigned(26, 8));

        matrix_A(1,0) <= std_logic_vector(to_unsigned(72, 8)); 
        matrix_A(1,1) <= std_logic_vector(to_unsigned(96, 8)); 
        matrix_A(1,2) <= std_logic_vector(to_unsigned(15, 8));

        matrix_A(2,0) <= std_logic_vector(to_unsigned(89, 8)); 
        matrix_A(2,1) <= std_logic_vector(to_unsigned(22, 8)); 
        matrix_A(2,2) <= std_logic_vector(to_unsigned(11, 8));

        -- Load E = [93 45 12; 75 37 40; 53 3 36]
        matrix_B(0,0) <= std_logic_vector(to_unsigned(93, 8)); 
        matrix_B(0,1) <= std_logic_vector(to_unsigned(45, 8)); 
        matrix_B(0,2) <= std_logic_vector(to_unsigned(12, 8));

        matrix_B(1,0) <= std_logic_vector(to_unsigned(75, 8)); 
        matrix_B(1,1) <= std_logic_vector(to_unsigned(37, 8)); 
        matrix_B(1,2) <= std_logic_vector(to_unsigned(40, 8));

        matrix_B(2,0) <= std_logic_vector(to_unsigned(53, 8)); 
        matrix_B(2,1) <= std_logic_vector(to_unsigned(3, 8)); 
        matrix_B(2,2) <= std_logic_vector(to_unsigned(36, 8));


        -- Hold reset for 2 clock cycles
        wait for 2 * CLK_PER;
        reset <= '0';

        -- Wait for systolic array to propagate and compute
        wait for 10 * CLK_PER;

        -- -- Print result
        -- report "PE(0,0) = " & integer'image(to_integer(unsigned(result_C(0,0))));
        -- report "PE(0,1) = " & integer'image(to_integer(unsigned(result_C(0,1))));
        -- report "PE(0,2) = " & integer'image(to_integer(unsigned(result_C(0,2))));
        -- report "PE(1,0) = " & integer'image(to_integer(unsigned(result_C(1,0))));
        -- report "PE(1,1) = " & integer'image(to_integer(unsigned(result_C(1,1))));
        -- report "PE(1,2) = " & integer'image(to_integer(unsigned(result_C(1,2))));
        -- report "PE(2,0) = " & integer'image(to_integer(unsigned(result_C(2,0))));
        -- report "PE(2,1) = " & integer'image(to_integer(unsigned(result_C(2,1))));
        -- report "PE(2,2) = " & integer'image(to_integer(unsigned(result_C(2,2))));

        wait;
    end process;
end sim;
