-- Dynamic Systolic Array 
-- Project #43 (2025)
library ieee; 
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.custom_types.all;

entity systolic_array is 
port(
    
    -- clock and reset signals
    clk : in bit_1; -- synchronous 
    reset : in bit_1; -- reset 

    -- inputs for the MAC operation
    data_shift    : in input_shift_matrix;
    weight_shift  : in input_shift_matrix;
    enabled_PE    : in enabled_PE_matrix;

    -- outputs for the MAC operation
    output : out systolic_array_matrix_output
);
end systolic_array;
    
architecture behaviour of systolic_array is

    -- bus signals to connect all PEs
    signal data_bus   : data_bus_matrix := (others => (others => (others => '0')));
    signal weight_bus : weight_bus_matrix := (others => (others => (others => '0')));
    signal results    : result_matrix := (others => (others => (others => '0')));

begin

    -- Dynamically assign inputs to left/top edges
    feed_data_edge: for i in 0 to N-1 generate
        data_bus(i, 0) <= data_shift(i); 
    end generate feed_data_edge;

    feed_weight_edge: for i in 0 to N-1 generate
        weight_bus(0, i) <= weight_shift(i); 
    end generate feed_weight_edge;

    -- Generate block for 8x8 PEs
    gen_PE_array : for i in 0 to N-1 generate
    pe_col : for j in 0 to N-1 generate        
    begin
        -- instantiates a PE
        PE_inst : entity work.processing_element
        port map (
            clk             => clk,
            reset           => reset,
            en              => enabled_PE(i, j), -- based off the enabled PE mask
            in_data         => data_bus(i, j), -- input data at the PE location
            in_weight       => weight_bus(i, j), -- input weihgt at the PE location
            out_data        => data_bus(i, j+1), -- passes the data to the right 
            out_weight      => weight_bus(i+1, j), -- passes the weight downwards
            result_register => results(i, j) -- holds the accumlated value 
        );
    end generate pe_col;
end generate gen_PE_array;

    -- Assign results to output
    gen_output_assign : for i in 0 to N-1 generate
        gen_output_col : for j in 0 to N-1 generate
        begin
            --  only assign a result if the PE is enabled, otherwise it is 0
            output(i, j) <= results(i, j) when enabled_PE(i,j) = '1' else (others => '0');
        end generate gen_output_col;
    end generate gen_output_assign;


    


end behaviour;