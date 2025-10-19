-- Project #43 (2025) - Top-level integration of Control Unit and Systolic Array
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.custom_types.all;

entity top_level_systolic_array is
    port (
        clk           : in  bit_1;
        reset         : in  bit_1;
        size          : in integer;  -- size of the matrices (1 to N)

        -- Inputs to feed matrices
        matrix_data   : in  systolic_array_matrix_input;
        matrix_weight : in  systolic_array_matrix_input;

        -- Outputs from the systolic array
        output        : out systolic_array_matrix_output;
        cycle_count   : out integer;
        done         : out bit_1  -- signal to indicate completion of operation

    );
end top_level_systolic_array;

architecture structure of top_level_systolic_array is

    -- Internal signals to connect control unit and systolic array
    signal enabled_PE_mask   : enabled_PE_matrix;
    signal data_shift_sig    : input_shift_matrix;
    signal weight_shift_sig  : input_shift_matrix;

    -- matrices fed to control unit
    signal matrix_data_sig  : systolic_array_matrix_input := (others => (others => (others => '0')));
    signal matrix_weight_sig  : systolic_array_matrix_input := (others => (others => (others => '0')));

    --Indices for counting through row and column
    signal row, col : integer range 0 to N-1 := 0;
    signal loadingCompleteFlag : boolean := false;

    signal start_sig : bit_1 := '0';
    
    
begin
    -- Connect input matrices to internal signals
    matrix_data_sig <= matrix_data;
    matrix_weight_sig <= matrix_weight;
    
    -- Start signal logic - start after reset is released
    process(clk, reset)
    begin
        if reset = '1' then
            start_sig <= '0';
        elsif rising_edge(clk) then
            start_sig <= '1';  -- Always start after reset
        end if;
    end process;

    -- Instantiate the Control Unit
    control_unit: entity work.control_unit
        port map (
            clk              => clk,
            reset            => reset,
            start            => start_sig,
            matrix_data      => matrix_data_sig,
            matrix_weight    => matrix_weight_sig,
            data_shift       => data_shift_sig,
            weight_shift     => weight_shift_sig,
            cycle_count      => cycle_count,
            PE_enabled_mask  => enabled_PE_mask,
            done             => done
        );

    -- Instantiate the Systolic Array
    systolic_array: entity work.systolic_array
        port map (
            clk         => clk,
            reset       => reset,
            data_shift  => data_shift_sig,
            weight_shift=> weight_shift_sig,
            enabled_PE  => enabled_PE_mask,
            output      => output
        );


end architecture;