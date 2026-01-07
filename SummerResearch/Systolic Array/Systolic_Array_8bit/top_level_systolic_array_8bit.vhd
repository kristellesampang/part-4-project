-- Project #43 (2025) - Top-level integration of Control Unit and Systolic Array
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.custom_types.all;
use work.custom_types_8bit.all;

entity top_level_systolic_array_8bit is
    port (
        clk           : in  bit_1;
        reset         : in  bit_1;
        ready         : in bit_1;

        -- Inputs to feed matrices
        matrix_data   : in  work.custom_types.systolic_array_matrix_input;
        matrix_weight : in  work.custom_types.systolic_array_matrix_input;

        active_rows   : in integer;
        active_cols   : in integer;
        active_k      : in integer;

        -- Outputs from the systolic array
        -- completed     : out bit_1;
        output        : out work.custom_types.systolic_array_matrix_output;
        cycle_count   : out integer
    );
end top_level_systolic_array_8bit;

architecture structure of top_level_systolic_array_8bit is

    -- Internal signals to connect control unit and systolic array
    signal data_shift_sig    : work.custom_types_8bit.input_shift_matrix;
    signal weight_shift_sig  : work.custom_types_8bit.input_shift_matrix;
    signal enabled_PE_mask   : work.custom_types_8bit.enabled_PE_matrix;

    -- intermediate signals to handle bit-width conversion
    signal matrix_data_8bit   : work.custom_types_8bit.systolic_array_matrix_input;
    signal matrix_weight_8bit : work.custom_types_8bit.systolic_array_matrix_input;
    signal output_8bit        : work.custom_types_8bit.systolic_array_matrix_output;

begin

    -- bit stripping
    process(matrix_data, matrix_weight)
    begin
        -- Strip the upper bits from the input matrices
        for i in 0 to N-1 loop
            for j in 0 to N-1 loop
                -- Assuming matrix_data and matrix_weight are already 8-bit, no stripping needed
                -- If they were wider, we would strip here
                matrix_data_8bit(i,j) <= matrix_data(i,j)(7 downto 0);
                matrix_weight_8bit(i,j) <= matrix_weight(i,j)(7 downto 0);
            end loop;
        end loop;
    end process;

    -- Instantiate the Control Unit
    control_unit: entity work.control_unit_8bit
        port map (
            clk              => clk,
            reset            => reset,
            ready            => ready,
            -- completed        => completed,
            matrix_data      => matrix_data_8bit,
            matrix_weight    => matrix_weight_8bit,
            data_shift       => data_shift_sig,
            weight_shift     => weight_shift_sig,
            cycle_count      => cycle_count,
            PE_enabled_mask  => enabled_PE_mask,
            active_rows      => active_rows,
            active_cols      => active_cols,
            active_k        => active_k
        );

    -- Instantiate the Systolic Array
    systolic_array: entity work.systolic_array_8bit
        port map (
            clk         => clk,
            reset       => reset,
            data_shift  => data_shift_sig,
            weight_shift=> weight_shift_sig,
            enabled_PE  => enabled_PE_mask,
            output      => output_8bit
        );

    -- Convert back to original bit-width if needed
    process(output_8bit)
    begin
        for i in 0 to N-1 loop
            for j in 0 to N-1 loop
                
                output(i,j) <= "00000000" & output_8bit(i,j);
            end loop;
        end loop;
    end process;

    
end architecture;
