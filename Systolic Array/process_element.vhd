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
    in_accumulated : in std_logic_vector(31 downto 0); -- accumulated value from other PEs, must be 16 bits if we are doing 8x8

    -- outputs for the MAC operation
    out_data : out std_logic_vector(7 downto 0);
    out_weight : out std_logic_vector(7 downto 0);
    out_accumulated : out std_logic_vector(31 downto 0);
)
end processing_element;
    
architecture behaviour of processing_element is
    -- initialise signals and vairables
    signal data, weight : std_logic_vector(7 downto 0) := (others => '0');
    signal acc : std_logic_vector(31 downto 0) := (others => '0');

begin
    process(clk, reset) 
        begin
            -- reset all elements
            if reset = '1' then
                data <= (others => '0');
                weight <= (others => '0');
                acc <= (others => '0');
            elseif rising_edge (clk) then
                data <= in_data;
                weight <= in_weight;
                acc <= std_logic_vector(resize(signed(in_data), 32) * resize(signed(in_weight), 32) + in_accumulated);

            end if;
    end process;

    -- output assignment
    out_data <= data;
    out_weight <= weight;
    out_accumulated <= acc;
end behaviour;