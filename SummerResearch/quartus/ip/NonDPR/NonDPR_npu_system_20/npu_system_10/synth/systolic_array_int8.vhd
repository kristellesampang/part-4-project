-- Systolic Array INT8 -- Project #43 (2025)
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.custom_types_int8.all;

entity systolic_array_int8 is
port(
    clk          : in  bit_1;
    reset        : in  bit_1;
    data_shift   : in  input_shift_matrix_int8;
    weight_shift : in  input_shift_matrix_int8;
    enabled_PE   : in  enabled_PE_matrix_int8;
    output       : out systolic_array_matrix_output_int8
);
end systolic_array_int8;

architecture behaviour of systolic_array_int8 is
    signal data_bus   : data_bus_matrix_int8   := (others => (others => (others => '0')));
    signal weight_bus : weight_bus_matrix_int8 := (others => (others => (others => '0')));
    signal results    : result_matrix_int8     := (others => (others => (others => '0')));
begin

    feed_data_edge: for i in 0 to N16-1 generate
        data_bus(i, 0) <= data_shift(i);
    end generate feed_data_edge;

    feed_weight_edge: for i in 0 to N16-1 generate
        weight_bus(0, i) <= weight_shift(i);
    end generate feed_weight_edge;

    gen_PE_array : for i in 0 to N16-1 generate
    pe_col : for j in 0 to N16-1 generate
    begin
        PE_inst : entity work.processing_element_int8
        port map (
            clk             => clk,
            reset           => reset,
            en              => enabled_PE(i, j),
            in_data         => data_bus(i, j),
            in_weight       => weight_bus(i, j),
            out_data        => data_bus(i, j+1),
            out_weight      => weight_bus(i+1, j),
            result_register => results(i, j)
        );
    end generate pe_col;
    end generate gen_PE_array;

    gen_output_assign : for i in 0 to N16-1 generate
    gen_output_col : for j in 0 to N16-1 generate
    begin
        output(i, j) <= results(i, j) when enabled_PE(i,j) = '1' else (others => '0');
    end generate gen_output_col;
    end generate gen_output_assign;

end behaviour;