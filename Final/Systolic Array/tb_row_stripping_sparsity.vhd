library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.custom_types.all;

entity tb_row_stripping_sparsity is
end tb_row_stripping_sparsity;

architecture sim of tb_row_stripping_sparsity is

    -- Helper function to convert integers to 8-bit std_logic_vectors 
    function u8(x : integer) return bit_8 is
    begin
        return std_logic_vector(to_unsigned(x, 8));
    end function;


-- VHDL stimulus for data matrix
constant ACTIVE_ROWS_DATA : integer := 2;
constant ACTIVE_COLS_DATA : integer := 4;
constant MATRIX_DATA_STIMULUS : systolic_array_matrix_input := (
    (u8(3), u8(5), u8(1), u8(2), u8(0), u8(0), u8(0), u8(0)),
    (u8(8), u8(0), u8(6), u8(0), u8(0), u8(0), u8(0), u8(0)),
    others => (others => u8(0))
);

-- VHDL stimulus for weight matrix
constant ACTIVE_ROWS_WEIGHT : integer := 4;
constant ACTIVE_COLS_WEIGHT : integer := 3;
constant MATRIX_WEIGHT_STIMULUS : systolic_array_matrix_input := (
    (u8(9), u8(1), u8(2), u8(0), u8(0), u8(0), u8(0), u8(0)),
    (u8(7), u8(0), u8(0), u8(0), u8(0), u8(0), u8(0), u8(0)),
    (u8(0), u8(4), u8(0), u8(0), u8(0), u8(0), u8(0), u8(0)),
    (u8(1), u8(0), u8(5), u8(0), u8(0), u8(0), u8(0), u8(0)),
    others => (others => u8(0))
);

    -- System signals
    signal clk   : bit_1 := '0';
    signal reset : bit_1 := '1';
    constant CLK_PER : time := 20 ns;

    -- Signals to connect to the top level
    signal matrix_data_sig   : systolic_array_matrix_input;
    signal matrix_weight_sig : systolic_array_matrix_input;
    signal result_matrix_sig : systolic_array_matrix_output;

    -- Performance measurement signals
    --The running cycle counter
    signal latency_counter : natural := 0; -- 
    -- Control flag to start/stop the counter, ts dont work properly bruh
    signal run_counter     : boolean := false; 

begin
    clk <= not clk after CLK_PER / 2;

    -- Instantiate the Design Under Test (DUT).
    DUT: entity work.top_level_systolic_array
    port map (
        clk           => clk,
        reset         => reset,
        matrix_data   => matrix_data_sig,
        matrix_weight => matrix_weight_sig,
        output        => result_matrix_sig,
        cycle_count   => open, 
        active_rows   => ACTIVE_ROWS_DATA, 
        active_cols   => ACTIVE_COLS_DATA
    );


    LatencyTest: process
        -- Calculate latency until last result
        -- Formula (rows-1) for vertical travel + (cols-1) for horizontal travel + (cols) for stream duration + 1 for final PE register stage.
        variable last_result_latency : integer := (ACTIVE_ROWS_DATA - 1) + (ACTIVE_COLS_DATA - 1) + ACTIVE_COLS_DATA + 1;
        variable final_latency_value : natural := 0;
    begin
        -- Load the matrices
        matrix_data_sig   <= MATRIX_DATA_STIMULUS;
        matrix_weight_sig <= MATRIX_WEIGHT_STIMULUS;

        -- Apply reset pulse
        reset <= '1';
        wait for 2*CLK_PER;
        reset <= '0';
        
        -- Start the performance counter on the first cycle after reset.
        run_counter <= true;
        
        -- Wait for the exact number of cycles until the calculation is complete.
        wait for (last_result_latency) * CLK_PER;

        -- Capture the counter's value and immediately stop it.
        final_latency_value := latency_counter;
        run_counter <= false;
        
        -- Wait a few more cycles to allow signals to settle before ending simulation.
        wait for 5 * CLK_PER;

        -- Report the final performance results to the console.
        report "--- PERFORMANCE REPORT ---" severity note;
        report "Active Dimensions (Data): " & integer'image(ACTIVE_ROWS_DATA) & "x" & integer'image(ACTIVE_COLS_DATA) severity note;
        report "Measured Latency: " & integer'image(final_latency_value) & " cycles." severity note;
        report "--------------------------" severity note;
        
        wait; 
    end process;

    --  latency counter.
    Count_Cycles: process(clk)
    begin
        if rising_edge(clk) then
            if reset = '1' then
                latency_counter <= 0;
            -- Only increment the counter when the run_counter flag is true.
            elsif run_counter = true then
                latency_counter <= latency_counter + 1;
            end if;
        end if;
    end process;

end sim;
