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

    -- Test process for sparse 4x4 matrix
    process
    begin
        -- Reset all values first
        for i in 0 to 3 loop
            for j in 0 to 3 loop
                matrix_A(i,j) <= (others => '0');
                matrix_B(i,j) <= (others => '0');
            end loop;
        end loop;

        -- Load sparse A =
        -- [1 0 0 2;
        --  0 0 3 0;
        --  4 0 0 0;
        --  0 5 0 0]
        matrix_A(0,0) <= std_logic_vector(to_unsigned(1, 8));
        matrix_A(0,3) <= std_logic_vector(to_unsigned(2, 8));
        matrix_A(1,2) <= std_logic_vector(to_unsigned(3, 8));
        matrix_A(2,0) <= std_logic_vector(to_unsigned(4, 8));
        matrix_A(3,1) <= std_logic_vector(to_unsigned(5, 8));

        -- Load sparse B =
        -- [0 6 0 0;
        --  7 0 0 0;
        --  0 0 8 0;
        --  0 0 0 9]
        matrix_B(0,1) <= std_logic_vector(to_unsigned(6, 8));
        matrix_B(1,0) <= std_logic_vector(to_unsigned(7, 8));
        matrix_B(2,2) <= std_logic_vector(to_unsigned(8, 8));
        matrix_B(3,3) <= std_logic_vector(to_unsigned(9, 8));

        -- Reset pulse
        reset <= '1';
        wait for 2 * CLK_PER;
        reset <= '0';

        -- Wait long enough for computation to complete
        wait for 40 * CLK_PER;

        wait;
    end process;

end sim;

-- architecture sim of tb_top_level_systolic_array is
--   constant CLK_PER : time := 20 ns;

--   signal clk           : bit_1 := '0';
--   signal reset         : bit_1 := '1';
--   signal matrix_A      : systolic_array_matrix_input := (others => (others => (others => '0')));
--   signal matrix_B      : systolic_array_matrix_input := (others => (others => (others => '0')));
--   signal result_C      : systolic_array_matrix_output;
--   signal cycle_counter : integer;

--   -- helper to make 8-bit vectors quickly
--   function u8(i : integer) return bit_8 is
--   begin
--     return std_logic_vector(to_unsigned(i, 8));
--   end function;

--   -- Expected 4x4 product
--   type exp4_t is array (0 to 3, 0 to 3) of integer;
--   constant C_exp : exp4_t :=
--     (( 0,  6,  0, 18),
--      ( 0,  0, 24,  0),
--      ( 0, 24,  0,  0),
--      (35,  0,  0,  0));
-- begin
--   --------------------------------------------------------------------
--   -- DUT
--   --------------------------------------------------------------------
--   clk <= not clk after CLK_PER/2;

--   DUT: entity work.top_level_systolic_array
--     port map (
--       clk           => clk,
--       reset         => reset,
--       matrix_data   => matrix_A,
--       matrix_weight => matrix_B,
--       output        => result_C,
--       cycle_count   => cycle_counter
--     );

--   --------------------------------------------------------------------
--   -- Drive the exact 4x4 example into top-left of the 8x8 matrices
--   --------------------------------------------------------------------
--   stim: process
--     variable line_txt : string(1 to 100);
--     variable cursor   : integer := 1;
--     variable v        : integer;
--   begin
--     -- Clear all (already cleared by init); then place A and B 4x4 blocks.

--     -- A =
--     -- [1 0 0 2;
--     --  0 0 3 0;
--     --  4 0 0 0;
--     --  0 5 0 0]
--     matrix_A(0,0) <= u8(1);
--     matrix_A(0,3) <= u8(2);

--     matrix_A(1,2) <= u8(3);

--     matrix_A(2,0) <= u8(4);

--     matrix_A(3,1) <= u8(5);
--     -- everything else remains 0 (both inside 4x4 and outside)

--     -- B =
--     -- [0 6 0 0;
--     --  7 0 0 0;
--     --  0 0 8 0;
--     --  0 0 0 9]
--     matrix_B(0,1) <= u8(6);
--     matrix_B(1,0) <= u8(7);
--     matrix_B(2,2) <= u8(8);
--     matrix_B(3,3) <= u8(9);

--     -- Reset pulse
--     reset <= '1';
--     wait for 2*CLK_PER;
--     reset <= '0';

--     -- Wait long enough for 8x8 pipeline (3N-2 = 22 cycles). Add a cushion.
--     wait for 40*CLK_PER;

--     -- Print top-left 4x4 of C and check vs expected
--     report "Top-left 4x4 of C = A*B";
--     for i in 0 to 3 loop
--       line_txt := (others => ' ');
--       for j in 0 to 3 loop
--         v := to_integer(unsigned(result_C(i,j)));
--         report "C(" & integer'image(i) & "," & integer'image(j) & ") = " & integer'image(v);
--         -- simple assert check
--         assert v = C_exp(i,j)
--           report "Mismatch at C(" & integer'image(i) & "," & integer'image(j) &
--                  "): got " & integer'image(v) &
--                  " expected " & integer'image(C_exp(i,j))
--           severity warning;
--       end loop;
--     end loop;

--     wait;
--   end process;

-- end architecture;

-- architecture sim of tb_top_level_systolic_array is
--     signal clk           : bit_1 := '0';
--     signal reset         : bit_1 := '1';
--     signal matrix_A      : systolic_array_matrix_input := (others => (others => (others => '0')));
--     signal matrix_B      : systolic_array_matrix_input := (others => (others => (others => '0')));
--     signal result_C      : systolic_array_matrix_output;
--     signal cycle_counter : integer;
    
--     constant CLK_PER : time := 20 ns;
-- begin
--     -- Clock generation
--     clk <= not clk after CLK_PER / 2;

--     -- Instantiate DUT
--     DUT: entity work.top_level_systolic_array
--     port map (
--         clk           => clk,
--         reset         => reset,
--         matrix_data   => matrix_A,
--         matrix_weight => matrix_B,
--         output        => result_C,
--         cycle_count   => cycle_counter
--     );

--     -- Test process for matrix sizes 1x1 to 8x8
--     process
--         variable val_a : integer;
--         variable val_b : integer;
--     begin
--         for size in 1 to 8 loop
--             -- Reset values for each size (to match MATLAB)
--             val_a := 1;
--             val_b := 10;

--             -- Load matrix_A and matrix_B with sample values
--             for i in 0 to size-1 loop
--                 for j in 0 to size-1 loop
--                     matrix_A(i,j) <= std_logic_vector(to_unsigned(val_a, 8));
--                     matrix_B(i,j) <= std_logic_vector(to_unsigned(val_b, 8));
--                     val_a := val_a + 1;
--                     val_b := val_b + 1;
--                 end loop;
--             end loop;

--             -- Reset
--             reset <= '1';
--             wait for 2 * CLK_PER;
--             reset <= '0';

--             -- Wait long enough for computation to complete
--             wait for (2 * size + size + 2) * CLK_PER;


--             wait for 5 * CLK_PER;
--         end loop;

--         wait;
--     end process;

-- end sim;
