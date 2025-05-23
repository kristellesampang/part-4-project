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
    in_accumulator : in std_logic_vector(31 downto 0); -- accumulated value 
    
    -- outputs for the MAC operation
    out_data : out std_logic_vector(7 downto 0);
    out_weight : out std_logic_vector(7 downto 0);
    result_register : out std_logic_vector(31 downto 0) -- this stays within the PE, the systolic array does not do anything with this 
);
end processing_element;
    
architecture behaviour of processing_element is
    -- -- initialise signals and vairables
    signal data, weight : std_logic_vector(7 downto 0) := (others => '0');
    signal accumulator : unsigned(31 downto 0) := (others => '0');
    -- signal multiplication : unsigned(31 downto 0) := (others => '0');
    signal result_valid : std_logic := '0';


    -- Function to check for undefined 'X' bits
    function contains_X(signal_vec: std_logic_vector) return boolean is
    begin
        for i in signal_vec'range loop
            if signal_vec(i) = 'X' then
                return true;
            end if;
        end loop;
        return false;
    end function;

begin
    -- instantiate all the processing elements
    
    process(clk, reset)
        variable multiplication : unsigned(31 downto 0) := (others => '0');
    begin
        if reset = '1' then
            data <= (others => '0');
            weight <= (others => '0');
            accumulator <= (others => '0');
            result_valid <= '0';
        elsif rising_edge(clk) then
            data <= in_data;
            weight <= in_weight;
            
            if result_valid = '0' then
                if not contains_X(in_data) and not contains_X(in_weight) and not contains_X(in_accumulator) then
                    multiplication := resize(unsigned(in_data) * unsigned(in_weight), 32);
                    accumulator <= multiplication + unsigned(in_accumulator);
                    if multiplication /= 0 then  -- crude way to detect valid data passed
                        result_valid <= '1';     -- lock once valid result seen
                    end if;
                end if;
            end if;
        end if;
    end process;

    -- output assignment
    out_data <= data;
    out_weight <= weight;
    result_register <= std_logic_vector(accumulator);
    -- result_register <= std_logic_vector(multiplication);

end behaviour;