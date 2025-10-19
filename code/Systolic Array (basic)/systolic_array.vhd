-- 3x3 Systolic Arrays 
-- Project #43 (2025)


library ieee; 
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.matrix_type.all;


entity systolic_array is 
port(
    
    clk : in std_logic; -- synchronous 
    reset : in std_logic; -- reset 

    -- inputs for the MAC operation
    matrix_data : in matrix_3x3_input; -- data 
    matrix_weight : in matrix_3x3_input; -- weight 

    -- outputs for the MAC operation
    output : out matrix_3x3_output;
    cycle_counter : out integer
);
end systolic_array;
    
architecture behaviour of systolic_array is

    -- just a random name for a random custom type to hold a row or column of elements 
    type some_array_name is array (0 to 2) of std_logic_vector(7 downto 0);

    -- shift registers for feeding data and weights
    signal data_shift : some_array_name := (others => (others => '0'));
    signal weight_shift : some_array_name := (others => (others => '0'));
    

    -- Horizontal connections: data flows left to right
    signal PE00_2_PE01_data, PE01_2_PE02_data : std_logic_vector(7 downto 0) := (others => '0');
    signal PE10_2_PE11_data, PE11_2_PE12_data : std_logic_vector(7 downto 0) := (others => '0');
    signal PE20_2_PE21_data, PE21_2_PE22_data : std_logic_vector(7 downto 0) := (others => '0');

    -- Vertical connections: data flows top to bottom
    signal PE00_2_PE10_weight, PE10_2_PE20_weight : std_logic_vector(7 downto 0) := (others => '0');
    signal PE01_2_PE11_weight, PE11_2_PE21_weight : std_logic_vector(7 downto 0) := (others => '0');
    signal PE02_2_PE12_weight, PE12_2_PE22_weight : std_logic_vector(7 downto 0) := (others => '0');
    
    -- Result register of eahh PE
    signal PE00_result, PE01_result, PE02_result : std_logic_vector(31 downto 0) := (others => '0');
    signal PE10_result, PE11_result, PE12_result : std_logic_vector(31 downto 0) := (others => '0');
    signal PE20_result, PE21_result, PE22_result : std_logic_vector(31 downto 0) := (others => '0');

    -- use this to manually input to the systolic array
    signal cycle_count : integer := 0;

begin
    
    -- instantiate all PEs
    --Each PE will take data from left and weight from top and generate it's own sum

    -- Row 0
    PE00: entity work.processing_element
    port map (
        clk             => clk,
        reset           => reset,
        in_data         => data_shift(0),
        in_weight       => weight_shift(0),
        --in_accumulator  => PE00_accumulator,
        out_data        => PE00_2_PE01_data,
        out_weight      => PE00_2_PE10_weight,
        result_register => PE00_result
    );

    PE01: entity work.processing_element
    port map (
        clk             => clk,
        reset           => reset,
        in_data         => PE00_2_PE01_data,
        in_weight       => weight_shift(1),
        --in_accumulator  => PE01_accumulator,
        out_data        => PE01_2_PE02_data,
        out_weight      => PE01_2_PE11_weight,
        result_register => PE01_result
    );


    PE02: entity work.processing_element
    port map (
        clk             => clk,
        reset           => reset,
        in_data         => PE01_2_PE02_data,
        in_weight       => weight_shift(2),
         --in_accumulator  => PE02_accumulator,
        out_data        => open,
        out_weight      => PE02_2_PE12_weight,
        result_register => PE02_result
    );

    -- Row 1
    PE10: entity work.processing_element
    port map (
        clk             => clk,
        reset           => reset,
        in_data         => data_shift(1),
        in_weight       => PE00_2_PE10_weight,
        --in_accumulator  => PE10_accumulator,
        out_data        => PE10_2_PE11_data,
        out_weight      => PE10_2_PE20_weight,
        result_register => PE10_result
    );

    PE11: entity work.processing_element
    port map (
        clk             => clk,
        reset           => reset,
        in_data         => PE10_2_PE11_data,
        in_weight       => PE01_2_PE11_weight,
        --in_accumulator  => PE11_accumulator,
        out_data        => PE11_2_PE12_data,
        out_weight      => PE11_2_PE21_weight,
        result_register => PE11_result 
    );

    PE12: entity work.processing_element
    port map (
        clk             => clk,
        reset           => reset,
        in_data         => PE11_2_PE12_data,
        in_weight       => PE02_2_PE12_weight,
        --in_accumulator  => PE12_accumulator,
        out_data        => open,
        out_weight      => PE12_2_PE22_weight,
        result_register => PE12_result
    );

    -- Row 2
    PE20: entity work.processing_element
    port map (
        clk             => clk,
        reset           => reset,
        in_data         => data_shift(2),
        in_weight       => PE10_2_PE20_weight,
        --in_accumulator  => PE20_accumulator,
        out_data        => PE20_2_PE21_data,
        out_weight      => open,
        result_register => PE20_result
    );

    PE21: entity work.processing_element
    port map (
        clk             => clk,
        reset           => reset,
        in_data         => PE20_2_PE21_data,
        in_weight       => PE11_2_PE21_weight,
        --in_accumulator  => PE21_accumulator,
        out_data        => PE21_2_PE22_data,
        out_weight      => open,
        result_register => PE21_result
    );

    PE22: entity work.processing_element
    port map (
        clk             => clk,
        reset           => reset,
        in_data         => PE21_2_PE22_data,
        in_weight       => PE12_2_PE22_weight,
        --in_accumulator  => PE22_accumulator,
        out_data        => open,
        out_weight      => open,
        result_register => PE22_result
    );

-- Process that feeds in the inputs (will probably turn into a control unit later on as per morteza's suggestion)
process(clk, reset) 
begin
    if reset = '1' then 
        cycle_count <= 0;

        -- Clear all shift registers on reset
        data_shift(0)  <= (others => '0');
        data_shift(1)  <= (others => '0');
        data_shift(2)  <= (others => '0');

        weight_shift(0) <= (others => '0');
        weight_shift(1) <= (others => '0');
        weight_shift(2) <= (others => '0');

    elsif rising_edge(clk) then    	
        cycle_count <= cycle_count + 1;

        -- Feed row 0 of A (data), one element per cycle
        if cycle_count < 3 then
            data_shift(0) <= matrix_data(0, cycle_count);
        else
            data_shift(0) <= (others => '0');
        end if;

        -- Feed row 1 of A with 1-cycle delay
        if cycle_count >= 1 and cycle_count < 4 then
            data_shift(1) <= matrix_data(1, cycle_count-1);
        else
            data_shift(1) <= (others => '0');
        end if;

        -- Feed row 2 of A with 2-cycle delay
        if cycle_count >= 2 and cycle_count < 5 then
            data_shift(2) <= matrix_data(2, cycle_count-2);
        else
            data_shift(2) <= (others => '0');
        end if;

        -- feed col 0 of B (weight), one element per cycle
        if cycle_count < 3 then
            weight_shift(0) <= matrix_weight(cycle_count, 0);
        else
            weight_shift(0) <= (others => '0');
        end if;

        -- Feed col 1 of B with 1-cycle delay
        if cycle_count >= 1 and cycle_count < 4 then
            weight_shift(1) <= matrix_weight(cycle_count-1, 1);
        else
            weight_shift(1) <= (others => '0');
        end if;

        -- Feed col 2 of B with 2-cycle delay
        if cycle_count >= 2 and cycle_count < 5 then
            weight_shift(2) <= matrix_weight(cycle_count-2, 2);
        else
            weight_shift(2) <= (others => '0');
        end if;
    end if;
end process;

        -- assign them result registers of each PE to the output matrix

    -- output assignment, which is the the result registers
    output(0,0) <= (PE00_result);
    output(0,1) <= (PE01_result);
    output(0,2) <= (PE02_result);
    output(1,0) <= (PE10_result);
    output(1,1) <= (PE11_result);
    output(1,2) <= (PE12_result);
    output(2,0) <= (PE20_result);
    output(2,1) <= (PE21_result);
    output(2,2) <= (PE22_result);
    cycle_counter <= cycle_count;

end behaviour;