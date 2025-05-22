-- 4x4 Systolic Arrays 
-- Project #43 (2025)


library ieee; 
use ieee.std_logic_1164;
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
)
end systolic_array;
    
architecture behaviour of systolic_array is
    -- initialise signals and vairables
    -- Horizontal connections: data flows left to right
    signal PE00_2_PE01_data        : std_logic_vector(7 downto 0);
    signal PE01_2_PE02_data        : std_logic_vector(7 downto 0);
    signal PE10_2_PE11_data        : std_logic_vector(7 downto 0);
    signal PE11_2_PE12_data        : std_logic_vector(7 downto 0);
    signal PE20_2_PE21_data        : std_logic_vector(7 downto 0);
    signal PE21_2_PE22_data        : std_logic_vector(7 downto 0);

    -- Vertical connections: data flows top to bottom
    signal PE00_2_PE10_weight      : std_logic_vector(7 downto 0);
    signal PE10_2_PE20_weight      : std_logic_vector(7 downto 0);
    signal PE01_2_PE11_weight      : std_logic_vector(7 downto 0);
    signal PE11_2_PE21_weight      : std_logic_vector(7 downto 0);
    signal PE02_2_PE12_weight      : std_logic_vector(7 downto 0);
    signal PE12_2_PE22_weight      : std_logic_vector(7 downto 0);
    
    -- Initialise the inputs from the matrix
    signal PE00_data : std_logic_vector(7 downto 0);
    signal PE10_data : std_logic_vector(7 downto 0);
    signal PE20_data : std_logic_vector(7 downto 0);
    signal PE00_weight : std_logic_vector(7 downto 0);
    signal PE01_weight : std_logic_vector(7 downto 0);
    signal PE02_weight : std_logic_vector(7 downto 0);

    -- Result register of eahh PE
    signal PE00_result : std_logic_vector(31 downto 0);
    signal PE01_result : std_logic_vector(31 downto 0);
    signal PE02_result : std_logic_vector(31 downto 0);
    signal PE10_result : std_logic_vector(31 downto 0);
    signal PE11_result : std_logic_vector(31 downto 0);
    signal PE12_result : std_logic_vector(31 downto 0);
    signal PE20_result : std_logic_vector(31 downto 0);
    signal PE21_result : std_logic_vector(31 downto 0);
    signal PE22_result : std_logic_vector(31 downto 0);

    -- Dummy signals to move data and weight outside the systolic array
    signal PE02_2_dummy : std_logic_vector(7 downto 0);
    signal PE12_2_dummy : std_logic_vector(7 downto 0);
    signal PE22_2_dummy_data : std_logic_vector(7 downto 0);
    signal PE22_2_dummy_weight : std_logic_vector(7 downto 0);
    signal PE20_2_dummy : std_logic_vector(7 downto 0);
    signal PE21_2_dummy : std_logic_vector(7 downto 0);


    

begin
    PE00_data <= matrix_data(0,0);
    PE10_data <= matrix_data(1,0);
    PE20_data <= matrix_data(2,0);
    PE00_weight <= matrix_weight(0,0);
    PE01_weight <= matrix_weight(0,1);
    PE02_weight <= matrix_weight(0,2);

    -- instantiate all PEs
    PE00: entity work.processing_element
    port map (
        clk             => clk,
        reset           => reset,
        in_data         => PE00_data,
        in_weight       => PE00_weight,
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
        out_data        => PE22_2_dummy_data,
        out_weight      => PE22_2_dummy_weight,
        result_register => PE22_result
    );

    process(clk) 
        begin
            -- no need for reset in top level 
            if rising_edge(clk) then
                -- do stuff
            end if;
           
    end process;

    -- output assignment, which is the the result registers
    output(0,0) <= unsigned(PE00_result);
    output(0,1) <= unsigned(PE01_result);
    output(0,2) <= unsigned(PE02_result);
    output(1,0) <= unsigned(PE10_result);
    output(1,1) <= unsigned(PE11_result);
    output(1,2) <= unsigned(PE12_result);
    output(2,0) <= unsigned(PE20_result);
    output(2,1) <= unsigned(PE21_result);
    output(2,2) <= unsigned(PE22_result);
    
end behaviour;