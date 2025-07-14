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
    matrix_data : in matrix_3x3; -- data 
    matrix_weight : in matrix_3x3; -- weight 

    -- outputs for the MAC operation
    output : out matrix_3x3_output;
    cycle_counter : out integer
);
end systolic_array;
    
architecture behaviour of systolic_array is

    
    -- shift registers for feeding data and weights
    signal data_shift : input_shift_matrix := (others => (others => '0'));
    signal weight_shift : input_shift_matrix := (others => (others => '0'));

    -- placeholder for the PE enable bits
    signal enabled_PE : PE_en_3x3 := (others => (others => '1'));
    

    -- bus signals to connect all PEs
    signal data_bus   : data_bus_matrix := (others => (others => (others => '0')));
    signal weight_bus : weight_bus_matrix := (others => (others => (others => '0')));
    signal results    : result_matrix := (others => (others => (others => '0')));

    -- use this to manually input to the systolic array
    signal cycle_count : integer := 0;

begin

    -- Dynamically assign inputs to left/top edges using generate
    feed_data_edge: for i in 0 to 2 generate
        data_bus(i, 0) <= data_shift(i);
    end generate feed_data_edge;

    feed_weight_edge: for i in 0 to 2 generate
        weight_bus(0, i) <= weight_shift(i);
    end generate feed_weight_edge;

    -- Generate block for 3x3 PEs
    gen_PE_array : for i in 0 to 2 generate
    pe_col : for j in 0 to 2 generate        
    begin
        PE_inst : entity work.processing_element
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

     -- Assign results to output
    output(0,0) <= results(0,0);
    output(0,1) <= results(0,1);
    output(0,2) <= results(0,2);
    output(1,0) <= results(1,0);
    output(1,1) <= results(1,1);
    output(1,2) <= results(1,2);
    output(2,0) <= results(2,0);
    output(2,1) <= results(2,1);
    output(2,2) <= results(2,2);
    cycle_counter <= cycle_count;


-- Feeder Process
    process(clk, reset) 
    begin
        if reset = '1' then 
            cycle_count <= 0;

            data_shift(0) <= (others => '0');
            data_shift(1) <= (others => '0');
            data_shift(2) <= (others => '0');

            weight_shift(0) <= (others => '0');
            weight_shift(1) <= (others => '0');
            weight_shift(2) <= (others => '0');

        elsif rising_edge(clk) then    	
            cycle_count <= cycle_count + 1;

            -- Feed matrix_data row-wise with staggered delays
            if cycle_count < 3 then
                data_shift(0) <= matrix_data(0, cycle_count);
            else
                data_shift(0) <= (others => '0');
            end if;

            if cycle_count >= 1 and cycle_count < 4 then
                data_shift(1) <= matrix_data(1, cycle_count - 1);
            else
                data_shift(1) <= (others => '0');
            end if;

            if cycle_count >= 2 and cycle_count < 5 then
                data_shift(2) <= matrix_data(2, cycle_count - 2);
            else
                data_shift(2) <= (others => '0');
            end if;

            -- Feed matrix_weight column-wise with staggered delays
            if cycle_count < 3 then
                weight_shift(0) <= matrix_weight(cycle_count, 0);
            else
                weight_shift(0) <= (others => '0');
            end if;

            if cycle_count >= 1 and cycle_count < 4 then
                weight_shift(1) <= matrix_weight(cycle_count - 1, 1);
            else
                weight_shift(1) <= (others => '0');
            end if;

            if cycle_count >= 2 and cycle_count < 5 then
                weight_shift(2) <= matrix_weight(cycle_count - 2, 2);
            else
                weight_shift(2) <= (others => '0');
            end if;
        end if;
    end process;


end behaviour;