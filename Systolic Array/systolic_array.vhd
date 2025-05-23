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
    -- initialise signals and vairables
    -- Horizontal connections: data flows left to right
    signal PE00_2_PE01_data        : std_logic_vector(7 downto 0) := (others => '0');
    signal PE01_2_PE02_data        : std_logic_vector(7 downto 0) := (others => '0');
    signal PE10_2_PE11_data        : std_logic_vector(7 downto 0) := (others => '0');
    signal PE11_2_PE12_data        : std_logic_vector(7 downto 0) := (others => '0');
    signal PE20_2_PE21_data        : std_logic_vector(7 downto 0) := (others => '0');
    signal PE21_2_PE22_data        : std_logic_vector(7 downto 0) := (others => '0');

    -- Vertical connections: data flows top to bottom
    signal PE00_2_PE10_weight      : std_logic_vector(7 downto 0) := (others => '0');
    signal PE10_2_PE20_weight      : std_logic_vector(7 downto 0) := (others => '0');
    signal PE01_2_PE11_weight      : std_logic_vector(7 downto 0) := (others => '0');
    signal PE11_2_PE21_weight      : std_logic_vector(7 downto 0) := (others => '0');
    signal PE02_2_PE12_weight      : std_logic_vector(7 downto 0) := (others => '0');
    signal PE12_2_PE22_weight      : std_logic_vector(7 downto 0) := (others => '0');
    
    -- Initialise the inputs from the matrix
    signal PE00_data : std_logic_vector(7 downto 0) := (others => '0');
    signal PE10_data : std_logic_vector(7 downto 0) := (others => '0');
    signal PE20_data : std_logic_vector(7 downto 0) := (others => '0');
    signal PE00_weight : std_logic_vector(7 downto 0) := (others => '0');
    signal PE01_weight : std_logic_vector(7 downto 0) := (others => '0');
    signal PE02_weight : std_logic_vector(7 downto 0) := (others => '0');

    -- Result register of eahh PE
    signal PE00_result : std_logic_vector(31 downto 0) := (others => '0');
    signal PE01_result : std_logic_vector(31 downto 0) := (others => '0');
    signal PE02_result : std_logic_vector(31 downto 0) := (others => '0');
    signal PE10_result : std_logic_vector(31 downto 0) := (others => '0');
    signal PE11_result : std_logic_vector(31 downto 0) := (others => '0');
    signal PE12_result : std_logic_vector(31 downto 0) := (others => '0');
    signal PE20_result : std_logic_vector(31 downto 0) := (others => '0');
    signal PE21_result : std_logic_vector(31 downto 0) := (others => '0');
    signal PE22_result : std_logic_vector(31 downto 0) := (others => '0');

    -- Accumulator of each PE
    signal PE00_accumulator : std_logic_vector(31 downto 0) := (others => '0');
    signal PE01_accumulator : std_logic_vector(31 downto 0) := (others => '0');
    signal PE02_accumulator : std_logic_vector(31 downto 0) := (others => '0');
    signal PE10_accumulator : std_logic_vector(31 downto 0) := (others => '0');
    signal PE11_accumulator : std_logic_vector(31 downto 0) := (others => '0');
    signal PE12_accumulator : std_logic_vector(31 downto 0) := (others => '0');
    signal PE20_accumulator : std_logic_vector(31 downto 0) := (others => '0');
    signal PE21_accumulator : std_logic_vector(31 downto 0) := (others => '0');
    signal PE22_accumulator : std_logic_vector(31 downto 0) := (others => '0');

    -- Dummy signals to move data and weight outside the systolic array
    signal PE02_2_dummy : std_logic_vector(7 downto 0) := (others => '0');
    signal PE12_2_dummy : std_logic_vector(7 downto 0) := (others => '0');
    signal PE22_2_dummy_data : std_logic_vector(7 downto 0) := (others => '0');
    signal PE22_2_dummy_weight : std_logic_vector(7 downto 0) := (others => '0');
    signal PE20_2_dummy : std_logic_vector(7 downto 0) := (others => '0');
    signal PE21_2_dummy : std_logic_vector(7 downto 0) := (others => '0');

    -- use this to manually input to the systolic array
    signal cycle_count : natural := 0;

    -- Flattened matrix A (data)
    signal a1, a2, a3 : std_logic_vector(7 downto 0):= (others => '0');
    signal a4, a5, a6 : std_logic_vector(7 downto 0):= (others => '0');
    signal a7, a8, a9 : std_logic_vector(7 downto 0):= (others => '0');

    -- Flattened matrix B (weight)
    signal b1, b2, b3 : std_logic_vector(7 downto 0):= (others => '0');
    signal b4, b5, b6 : std_logic_vector(7 downto 0):= (others => '0');
    signal b7, b8, b9 : std_logic_vector(7 downto 0):= (others => '0');


begin
    
    -- A: row-major
    a1 <= matrix_data(0,0);
    a2 <= matrix_data(0,1);
    a3 <= matrix_data(0,2);
    a4 <=matrix_data(1,0);
    a5 <=matrix_data(1,1);
    a6 <= matrix_data(1,2);
    a7 <=matrix_data(2,0);
    a8 <= matrix_data(2,1);
    a9 <= matrix_data(2,2);

    -- B: column-major
    b1 <= matrix_weight(0,0);
    b2 <= matrix_weight(1,0);
    b3 <=matrix_weight(2,0);
    b4 <= matrix_weight(0,1);
    b5 <= matrix_weight(1,1);
    b6 <= matrix_weight(2,1);
    b7 <= matrix_weight(0,2);
    b8 <= matrix_weight(1,2);
    b9 <= matrix_weight(2,2);

    -- instantiate all PEs
    PE00: entity work.processing_element
    port map (
        clk             => clk,
        reset           => reset,
        in_data         => PE00_data,
        in_weight       => PE00_weight,
        in_accumulator  => PE00_accumulator,
        out_data        => PE00_2_PE01_data,
        out_weight      => PE00_2_PE10_weight,
        result_register => PE00_result
    );

    PE01: entity work.processing_element
    port map (
        clk             => clk,
        reset           => reset,
        in_data         => PE00_2_PE01_data,
        in_weight       => PE01_weight,
        in_accumulator  => PE01_accumulator,
        out_data        => PE01_2_PE02_data,
        out_weight      => PE01_2_PE11_weight,
        result_register => PE01_result
    );


    PE02: entity work.processing_element
    port map (
        clk             => clk,
        reset           => reset,
        in_data         => PE01_2_PE02_data,
        in_weight       => PE02_weight,
        in_accumulator  => PE02_accumulator,
        out_data        => PE02_2_dummy,
        out_weight      => PE02_2_PE12_weight,
        result_register => PE02_result
    );

    PE10: entity work.processing_element
    port map (
        clk             => clk,
        reset           => reset,
        in_data         => PE10_data,
        in_weight       => PE00_2_PE10_weight,
        in_accumulator  => PE10_accumulator,
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
        in_accumulator  => PE11_accumulator,
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
        in_accumulator  => PE12_accumulator,
        out_data        => PE12_2_dummy,
        out_weight      => PE12_2_PE22_weight,
        result_register => PE12_result
    );

    PE20: entity work.processing_element
    port map (
        clk             => clk,
        reset           => reset,
        in_data         => PE20_data,
        in_weight       => PE10_2_PE20_weight,
        in_accumulator  => PE20_accumulator,
        out_data        => PE20_2_PE21_data,
        out_weight      => PE20_2_dummy,
        result_register => PE20_result
    );

    PE21: entity work.processing_element
    port map (
        clk             => clk,
        reset           => reset,
        in_data         => PE20_2_PE21_data,
        in_weight       => PE11_2_PE21_weight,
        in_accumulator  => PE21_accumulator,
        out_data        => PE21_2_PE22_data,
        out_weight      => PE21_2_dummy,
        result_register => PE21_result
    );

    PE22: entity work.processing_element
    port map (
        clk             => clk,
        reset           => reset,
        in_data         => PE21_2_PE22_data,
        in_weight       => PE12_2_PE22_weight,
        in_accumulator  => PE22_accumulator,
        out_data        => PE22_2_dummy_data,
        out_weight      => PE22_2_dummy_weight,
        result_register => PE22_result
    );

    process(clk, reset) 
        begin
            if reset = '1' then 
                cycle_count <= 0;
                PE00_data   <= a1;
                PE10_data   <= a4;
                PE20_data   <= a7;
                PE00_weight <= b1;
                PE01_weight <= b4;
                PE02_weight <= b7;

            elsif rising_edge(clk) then    	
                if cycle_count < 1000 then
                    cycle_count <= cycle_count + 1;
                end if;
                case cycle_count is
                    -- when cycle count is 0, it is the init clock cycle
                    when 1 =>
                        PE00_data   <= (matrix_data(0,0));
                        PE00_weight <= (matrix_weight(0,0));

                    when 2 =>
                        PE00_data   <= (matrix_data(0,1));
                        PE10_data   <= (matrix_data(1,0));
                        PE00_weight <= (matrix_weight(1,0));
                        PE01_weight <= (matrix_weight(0,1));

                    when 3 =>
                        PE00_data   <= (matrix_data(0,2));
                        PE10_data   <= (matrix_data(1,1));
                        PE20_data   <= (matrix_data(2,0));
                        PE00_weight <= (matrix_weight(2,0));
                        PE01_weight <= (matrix_weight(1,1));
                        PE02_weight <= (matrix_weight(0,2));

                    when 4 =>
                        PE10_data   <= (matrix_data(1,2));
                        PE20_data   <= (matrix_data(2,1));
                        PE01_weight <= (matrix_weight(2,1));
                        PE02_weight <= (matrix_weight(1,2));

                    when 5 =>
                        PE20_data   <= (matrix_data(2,2));
                        PE02_weight <= (matrix_weight(2,2));


                    when others =>
                        -- no new data/weight injection
                        PE00_data   <= (others => '0');
                        PE10_data   <= (others => '0');
                        PE20_data   <= (others => '0');
                        PE00_weight <= (others => '0');
                        PE01_weight <= (others => '0');
                        PE02_weight <= (others => '0');
                end case;
                PE00_accumulator <= PE00_result;
                PE01_accumulator <= PE01_result;
                PE02_accumulator <= PE02_result;
                PE10_accumulator <= PE10_result;
                PE11_accumulator <= PE11_result;
                PE12_accumulator <= PE12_result;
                PE20_accumulator <= PE20_result;
                PE21_accumulator <= PE21_result;
                PE22_accumulator <= PE22_result;
                
            end if;
           
    end process;

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