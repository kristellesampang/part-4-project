library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.custom_types.all;

entity tb_buffer is
end tb_buffer;

architecture sim of tb_buffer is

    constant clk_period : time := 20 ns;

    signal clk          : std_logic := '0';
    signal reset        : std_logic := '0';
    signal output_mat   : systolic_array_matrix_output;
    signal cycle_count  : integer;

    -- debug signals from inside DUT
    signal dbg_db_rdaddr    : std_logic_vector(10 downto 0);
    signal dbg_db_data_out  : std_logic_vector(7 downto 0);
    signal dbg_wb_rdaddr    : std_logic_vector(10 downto 0);
    signal dbg_wb_data_out  : std_logic_vector(7 downto 0);
    signal dbg_matrix_data  : systolic_array_matrix_input;
    signal dbg_matrix_weight: systolic_array_matrix_input;

begin
    
    clk <= not clk after clk_period / 2;

    -- Instantiate DUT with extra debug ports
    DUT: entity work.top_level_systolic_array
    port map (
        clk               => clk,
        reset             => reset,
        output            => output_mat,
        cycle_count       => cycle_count,
        dbg_db_rdaddr     => dbg_db_rdaddr,
        dbg_db_data_out   => dbg_db_data_out,
        dbg_wb_rdaddr     => dbg_wb_rdaddr,
        dbg_wb_data_out   => dbg_wb_data_out,
        dbg_matrix_data   => dbg_matrix_data,
        dbg_matrix_weight => dbg_matrix_weight
    );


    process(clk)
    begin
        if rising_edge(clk) then
            report "db_rdaddr=" & integer'image(to_integer(unsigned(dbg_db_rdaddr))) &
                   " db_data_out=" & integer'image(to_integer(unsigned(dbg_db_data_out))) &
                   " wb_rdaddr=" & integer'image(to_integer(unsigned(dbg_wb_rdaddr))) &
                   " wb_data_out=" & integer'image(to_integer(unsigned(dbg_wb_data_out)));
        end if;
    end process;

    process
    begin
        -- Hold reset for a bit so that buffers are loaded
        wait for 5 * clk_period;
        reset <= '0';

        -- wait for enough cycles to complete load + computation
        wait for clk_period * (N * N + (3 * N - 2) + 5);

        -- show final matrix in transcript
        for i in 0 to N-1 loop
            for j in 0 to N-1 loop
                report "Output(" & integer'image(i) & "," & integer'image(j) & ") = " &
                        integer'image(to_integer(unsigned(output_mat(i, j))));
            end loop;
        end loop;

        wait;
    end process;

end sim;
