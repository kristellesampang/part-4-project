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
    matrix_data : in systolic_array_matrix_input; -- data 
    matrix_weight : in systolic_array_matrix_input; -- weight 
    enabled_PE : in enabled_PE_matrix; -- enabled PE
    array_size : in integer; -- N size of the matrix input 

    -- outputs for the MAC operation
    output : out systolic_array_matrix_output;
    cycle_counter : out integer
);
end systolic_array;
    
architecture behaviour of systolic_array is

    
    -- shift registers for feeding data and weights
    signal data_shift : input_shift_matrix := (others => (others => '0')); -- 1xN size
    signal weight_shift : input_shift_matrix := (others => (others => '0')); -- 1xN size

    -- bus signals to connect all PEs
    signal data_bus   : data_bus_matrix := (others => (others => (others => '0')));
    signal weight_bus : weight_bus_matrix := (others => (others => (others => '0')));
    signal results    : result_matrix := (others => (others => (others => '0')));

    -- use this to manually input to the systolic array
    signal cycle_count : integer := 0;

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

    cycle_counter <= cycle_count;

    -- acts like the control unit of how the data is fed into the systolic array in a staggered manner
    feeder : process(clk, reset)
    begin
        -- resets all the values to 0 
        if reset = '1' then
            cycle_count <= 0;

            -- clear all shift registers
            for i in 0 to N-1 loop
                data_shift(i)   <= (others => '0');
                weight_shift(i) <= (others => '0');
            end loop;

        elsif rising_edge(clk) then
            -- increment clock cycle count at every rising edge 
            cycle_count <= cycle_count + 1;

            -- feeds the data (the left most side)
            for i in 0 to array_size-1 loop
                if cycle_count >= i and cycle_count < i + array_size then
                    -- matrix_data(row = i, col = cycle‑i)
                    data_shift(i) <= matrix_data(i, cycle_count - i);
                else
                    data_shift(i) <= (others => '0');
                end if;
            end loop;
            
            -- feeds the weight (from the top to bottom)
            for j in 0 to array_size-1 loop
                if cycle_count >= j and cycle_count < j + array_size then
                    -- matrix_weight(row = cycle‑j, col = j)
                    weight_shift(j) <= matrix_weight(cycle_count - j, j);
                else
                    weight_shift(j) <= (others => '0');
                end if;
            end loop;
        end if;
    end process feeder;


end behaviour;