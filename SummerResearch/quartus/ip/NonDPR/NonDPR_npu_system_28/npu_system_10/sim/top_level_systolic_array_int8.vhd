-- Top Level Systolic Array INT8 -- Project #43 (2025)
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.custom_types_int8.all;

entity top_level_systolic_array_int8 is
port(
    clk           : in  bit_1;
    reset         : in  bit_1;
    ready         : in  bit_1;

    matrix_data   : in  systolic_array_matrix_input_int8;
    matrix_weight : in  systolic_array_matrix_input_int8;

    active_rows   : in  integer;
    active_cols   : in  integer;
    active_k      : in  integer;

    done          : out bit_1;
    output        : out systolic_array_matrix_output_int8;
    cycle_count   : out integer
);
end top_level_systolic_array_int8;

architecture structure of top_level_systolic_array_int8 is
    signal data_shift_sig   : input_shift_matrix_int8;
    signal weight_shift_sig : input_shift_matrix_int8;
    signal enabled_PE_mask  : enabled_PE_matrix_int8;
begin

    control_unit_inst: entity work.control_unit_int8
    port map (
        clk             => clk,
        reset           => reset,
        ready           => ready,
        completed       => done,
        matrix_data     => matrix_data,
        matrix_weight   => matrix_weight,
        data_shift      => data_shift_sig,
        weight_shift    => weight_shift_sig,
        cycle_count     => cycle_count,
        PE_enabled_mask => enabled_PE_mask,
        active_rows     => active_rows,
        active_cols     => active_cols,
        active_k        => active_k
    );

    systolic_array_inst: entity work.systolic_array_int8
    port map (
        clk          => clk,
        reset        => reset,
        data_shift   => data_shift_sig,
        weight_shift => weight_shift_sig,
        enabled_PE   => enabled_PE_mask,
        output       => output
    );

end architecture;