-- Processing Element that functions as an ALU to execute MAC (Multiply-and-Accumulate) Operations for CNN 
-- Project #43 (2025)

library ieee; 
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;


entity processing_element is 
port(
    
    clk : in std_logic; -- synchronous 
    reset : in std_logic; -- reset 

    -- inputs for the MAC operation
    in_data : in std_logic_vector(7 downto 0); -- can be data input or activation 
    in_weight : in std_logic_vector(7 downto 0); -- weight
    --in_accumulator : in std_logic_vector(31 downto 0); -- accumulated value 
    
    -- outputs for the MAC operation
    out_data : out std_logic_vector(7 downto 0);
    out_weight : out std_logic_vector(7 downto 0);
    
    -- The accumulated MAC result
    result_register : out std_logic_vector(31 downto 0) -- this stays within the PE, the systolic array does not do anything with this 
);
end processing_element;
    
architecture behaviour of processing_element is

    -- -- initialise signals and vairables for the internal registers
    signal data, weight : std_logic_vector(7 downto 0) := (others => '0');
    signal accumulator : unsigned(31 downto 0) := (others => '0');


begin
    process(clk, reset)
        variable multiplication : unsigned(31 downto 0);
    begin
        if reset = '1' then
            data <= (others => '0');
            weight <= (others => '0');
            accumulator <= (others => '0');

        elsif rising_edge(clk) then
            -- Receive the incoming data and weight
            data <= in_data;
            weight <= in_weight;

            if in_data /= x"00" and in_weight /= x"00" then
                -- Compute the multplication result (the product)
                multiplication := resize(unsigned(in_data) * unsigned(in_weight), 32);

                -- Add product to accumulator
                accumulator <= accumulator + multiplication;
            end if;
        end if;
    end process;

    -- output assignment
    out_data <= data;
    out_weight <= weight;
    result_register <= std_logic_vector(accumulator);
    -- result_register <= std_logic_vector(multiplication);
end behaviour;