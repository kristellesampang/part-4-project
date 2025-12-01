library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.custom_types.all;

entity tb_dense_test is
end tb_dense_test;

architecture sim of tb_dense_test is

    -- Helper function must be here
    function u16(x : integer) return bit_16 is
        begin
            return std_logic_vector(to_signed(x, 16));
    end function;
    
    -- *************************************************************************
    -- *** 1. CONSTANTS AND SIGNAL DECLARATIONS (MUST BE BEFORE 'BEGIN') ***
    -- *************************************************************************
    

    -- Signals to connect to the DUT/Control Logic
    signal clk           : bit_1 := '0';
    signal reset         : bit_1 := '1';
    signal ready         : bit_1 := '0';
    signal matrix_data_sig   : systolic_array_matrix_input;
    signal matrix_weight_sig : systolic_array_matrix_input;
    signal result_matrix_sig : systolic_array_matrix_output;
    signal run_cycles_sig    : integer;

-- VHDL CONSTANTS FOR 3x3x3 DENSE TEST
    constant ACTIVE_ROWS : integer := 3;
    constant ACTIVE_K : integer := 3;
    constant ACTIVE_COLS : integer := 3;
    constant EXPECTED_LATENCY : integer := 7; -- 3 + 3 + 3 - 2 = 7 Cycles
    constant CLK_PER     : time  := 20 ns; 

    -- Expected final result value (1^2 + 2^2 + 3^2 = 14)
    constant EXPECTED_FINAL_RESULT : bit_64 := std_logic_vector(to_signed(14, 64)); 
    
    -- Matrix A: Rows [1, 2, 3] in all 3 rows (Padded to 32 elements)
    constant MATRIX_DATA_STIMULUS : systolic_array_matrix_input := (
        (u16(1), u16(2), u16(3), u16(0), u16(0), u16(0), u16(0), u16(0), u16(0), u16(0), u16(0), u16(0), u16(0), u16(0), u16(0), u16(0), u16(0), u16(0), u16(0), u16(0), u16(0), u16(0), u16(0), u16(0), u16(0), u16(0), u16(0), u16(0), u16(0), u16(0), u16(0), u16(0)),
        (u16(1), u16(2), u16(3), u16(0), u16(0), u16(0), u16(0), u16(0), u16(0), u16(0), u16(0), u16(0), u16(0), u16(0), u16(0), u16(0), u16(0), u16(0), u16(0), u16(0), u16(0), u16(0), u16(0), u16(0), u16(0), u16(0), u16(0), u16(0), u16(0), u16(0), u16(0), u16(0)),
        (u16(1), u16(2), u16(3), u16(0), u16(0), u16(0), u16(0), u16(0), u16(0), u16(0), u16(0), u16(0), u16(0), u16(0), u16(0), u16(0), u16(0), u16(0), u16(0), u16(0), u16(0), u16(0), u16(0), u16(0), u16(0), u16(0), u16(0), u16(0), u16(0), u16(0), u16(0), u16(0)),
        others => (others => u16(0))
    );

    -- Matrix B: Columns [1, 2, 3] in all 3 columns (Padded to 32 elements)
    constant MATRIX_WEIGHT_STIMULUS : systolic_array_matrix_input := (
        (u16(1), u16(1), u16(1), u16(0), u16(0), u16(0), u16(0), u16(0), u16(0), u16(0), u16(0), u16(0), u16(0), u16(0), u16(0), u16(0), u16(0), u16(0), u16(0), u16(0), u16(0), u16(0), u16(0), u16(0), u16(0), u16(0), u16(0), u16(0), u16(0), u16(0), u16(0), u16(0)),
        (u16(2), u16(2), u16(2), u16(0), u16(0), u16(0), u16(0), u16(0), u16(0), u16(0), u16(0), u16(0), u16(0), u16(0), u16(0), u16(0), u16(0), u16(0), u16(0), u16(0), u16(0), u16(0), u16(0), u16(0), u16(0), u16(0), u16(0), u16(0), u16(0), u16(0), u16(0), u16(0)),
        (u16(3), u16(3), u16(3), u16(0), u16(0), u16(0), u16(0), u16(0), u16(0), u16(0), u16(0), u16(0), u16(0), u16(0), u16(0), u16(0), u16(0), u16(0), u16(0), u16(0), u16(0), u16(0), u16(0), u16(0), u16(0), u16(0), u16(0), u16(0), u16(0), u16(0), u16(0), u16(0)),
        others => (others => u16(0))
    );
    
begin -- Executable statements begin here

    -- Concurrent Clock Generation (Needs to be outside a process to run continuously)
    clk <= not clk after CLK_PER / 2;

    -- Instantiate the Design Under Test (DUT)
    DUT: entity work.top_level_systolic_array
    port map (
        clk           => clk,
        reset         => reset,
        ready         => ready,
        matrix_data   => matrix_data_sig,
        matrix_weight => matrix_weight_sig,
        active_rows   => ACTIVE_ROWS,
        active_cols   => ACTIVE_COLS,
        active_k      => ACTIVE_K,
        output        => result_matrix_sig,
        cycle_count   => run_cycles_sig 
    );

    -- Simulation Stimulus and Verification Logic (Sequential Process)
    Stimulus: process
    begin
        -- 1. Load Matrices (Signals must be driven here)
        matrix_data_sig   <= MATRIX_DATA_STIMULUS;
        matrix_weight_sig <= MATRIX_WEIGHT_STIMULUS;

        -- 2. Apply Reset
        reset <= '1';
        wait for 3 * CLK_PER;
        reset <= '0';
        wait for CLK_PER;

        -- 3. Start Calculation (Ready Pulse)
        report "--- STARTING DENSE TEST ---" severity note;
        ready <= '1';
        
        -- Wait the full input duration (K=32 cycles)
        wait for ACTIVE_K * CLK_PER; 
        
        ready <= '0'; 
        
        -- 4. Wait for Completion (Pipeline flush)
        wait for (EXPECTED_LATENCY - ACTIVE_K + 5) * CLK_PER; 

        -- 5. Verification
        report "--- VERIFICATION REPORT ---" severity note;
        
        -- Check 1: Latency Check
        if run_cycles_sig = EXPECTED_LATENCY then
            report "PASS: Latency matches expected value of " & integer'image(EXPECTED_LATENCY) & " cycles." severity note;
        else
            report "FAIL: Latency Mismatch! Expected: " & integer'image(EXPECTED_LATENCY) & " cycles, Got: " & integer'image(run_cycles_sig) severity error;
        end if;
        
        -- Check 2: Final Result Check
        if result_matrix_sig(ACTIVE_ROWS-1, ACTIVE_COLS-1) = EXPECTED_FINAL_RESULT then
            report "PASS: Final cell C[31][31] matches expected value of 11440." severity note;
        else
            report "FAIL: Final cell C[31][31] result incorrect. Check waveforms." severity error;
        end if;
        
        wait;
    end process Stimulus;

end sim;