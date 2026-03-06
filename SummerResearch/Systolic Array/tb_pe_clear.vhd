library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity tb_pe_clear is
end tb_pe_clear;

architecture sim of tb_pe_clear is
    signal clk    : std_logic := '0';
    signal reset  : std_logic := '0';
    signal en     : std_logic := '0';
    signal in_data   : std_logic_vector(15 downto 0) := (others => '0');
    signal in_weight : std_logic_vector(15 downto 0) := (others => '0');
    signal out_data   : std_logic_vector(15 downto 0);
    signal out_weight : std_logic_vector(15 downto 0);
    signal result : std_logic_vector(63 downto 0);

    constant CLK_PER : time := 10 ns;
begin

    DUT: entity work.processing_element
    port map(
        clk => clk, reset => reset, en => en,
        in_data => in_data, in_weight => in_weight,
        out_data => out_data, out_weight => out_weight,
        result_register => result
    );

    clk <= not clk after CLK_PER/2;

    process
    begin
        -- Reset
        reset <= '1'; wait for 20 ns;
        reset <= '0'; wait for 20 ns;

        -- RUN 1: feed 3 x 4 = 12, expect result = 12
        en <= '1';
        in_data   <= std_logic_vector(to_signed(3, 16));
        in_weight <= std_logic_vector(to_signed(4, 16));
        wait until rising_edge(clk);
        in_data   <= (others => '0');
        in_weight <= (others => '0');
        wait until rising_edge(clk); -- mult registered
        wait until rising_edge(clk); -- acc registered
        en <= '0';
        wait for 20 ns;
        report "RUN 1 result = " & integer'image(to_integer(signed(result))) & " (expect 12)";

        -- clear cycle
        wait for 40 ns;

        -- RUN 2: same inputs, should also give 12 not 24
        en <= '1';
        in_data   <= std_logic_vector(to_signed(3, 16));
        in_weight <= std_logic_vector(to_signed(4, 16));
        wait until rising_edge(clk);
        in_data   <= (others => '0');
        in_weight <= (others => '0');
        wait until rising_edge(clk);
        wait until rising_edge(clk);
        en <= '0';
        wait for 20 ns;
        report "RUN 2 result = " & integer'image(to_integer(signed(result))) & " (expect 12, not 24)";

        report "DONE";
        wait;
    end process;
end architecture;