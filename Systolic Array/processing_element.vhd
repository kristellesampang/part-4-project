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
    
    -- outputs for the MAC operation
    out_data : out std_logic_vector(7 downto 0);
    out_weight : out std_logic_vector(7 downto 0);
    result_register : out std_logic_vector(31 downto 0); -- this stays within the PE, the systolic array does not do anything with this 
)
end processing_element;
    
architecture behaviour of processing_element is
    -- -- initialise signals and vairables
    signal data, weight : std_logic_vector(7 downto 0) := (others => '0');
    signal accumulator : std_logic_vector(31 downto 0) := (others => '0');

begin
    -- instantiate all the processing elements
    process(clk, reset) 
        begin
            -- reset all elements
            if reset = '1' then
                data <= (others => '0');
                weight <= (others => '0');
                accumulator <= (others => '0');
            elseif rising_edge (clk) then
                data <= in_data;
                weight <= in_weight;
                accumulator <= accumulator + resize(signed(in_data), 32) * resize(signed(in_weight), 32); -- MAC operation

            end if;
    end process;

    -- output assignment
    out_data <= data;
    out_weight <= weight;
    result_register <= std_logic_vector(accumulator);

end behaviour;