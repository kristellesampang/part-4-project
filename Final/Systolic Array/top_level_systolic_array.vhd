-- Project #43 (2025) - Top-level integration of Control Unit and Systolic Array
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.custom_types.all;

entity top_level_systolic_array is
    port (
        clk           : in  bit_1;
        reset         : in  bit_1;
        ready         : in bit_1;

        -- Inputs to feed matrices
        matrix_data   : in  systolic_array_matrix_input;
        matrix_weight : in  systolic_array_matrix_input;

        active_rows   : in integer;
        active_cols   : in integer;
        active_k      : in integer;

        -- Outputs from the systolic array
        completed     : out bit_1;
        output        : out systolic_array_matrix_output;
        cycle_count   : out integer
    );
end top_level_systolic_array;

architecture structure of top_level_systolic_array is

    -- Internal signals to connect control unit and systolic array
    signal data_shift_sig    : input_shift_matrix;
    signal weight_shift_sig  : input_shift_matrix;
    signal enabled_PE_mask   : enabled_PE_matrix;

begin

    -- Instantiate the Control Unit
    control_unit: entity work.control_unit
        port map (
            clk              => clk,
            reset            => reset,
            ready            => ready,
            completed        => completed,
            matrix_data      => matrix_data,
            matrix_weight    => matrix_weight,
            data_shift       => data_shift_sig,
            weight_shift     => weight_shift_sig,
            cycle_count      => cycle_count,
            PE_enabled_mask  => enabled_PE_mask,
            active_rows      => active_rows,
            active_cols      => active_cols,
            active_k        => active_k
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
