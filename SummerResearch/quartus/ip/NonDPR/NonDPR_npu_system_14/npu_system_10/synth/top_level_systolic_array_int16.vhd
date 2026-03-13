-- Project #43 (2025) - Top-level integration of Control Unit and Systolic Array
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.custom_types_int16.all;

entity top_level_systolic_array_int16 is
    port (
        clk           : in  bit_1;
        reset         : in  bit_1;
        ready         : in bit_1;

        -- Inputs to feed matrices
        matrix_data   : in  systolic_array_matrix_input_int16;
        matrix_weight : in  systolic_array_matrix_input_int16;

        active_rows   : in integer;
        active_cols   : in integer;
        active_k      : in integer;

        -- Outputs from the systolic array
        done          : out bit_1;
        output        : out systolic_array_matrix_output_int16;
        cycle_count   : out integer
    );
end top_level_systolic_array_int16;

architecture structure of top_level_systolic_array_int16 is

    -- Internal signals to connect control unit and systolic array
    signal data_shift_sig    : input_shift_matrix_int16;
    signal weight_shift_sig  : input_shift_matrix_int16;
    signal enabled_PE_mask   : enabled_PE_matrix_int16;

begin

    -- Instantiate the Control Unit
    control_unit: entity work.control_unit_int16
        port map (
            clk              => clk,
            reset            => reset,
            ready            => ready,
            completed        => done,
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
    systolic_array: entity work.systolic_array_int16
        port map (
            clk         => clk,
            reset       => reset,
            data_shift  => data_shift_sig,
            weight_shift=> weight_shift_sig,
            enabled_PE  => enabled_PE_mask,
            output      => output
        );
end architecture;
