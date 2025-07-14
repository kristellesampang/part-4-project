-- Processing Element that functions as an ALU to execute MAC (Multiply-and-Accumulate) Operations for CNN 
-- Project #43 (2025)

library ieee; 
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.custom_types.all;


entity processing_element is 
port(
    clk : in bit_1; -- synchronous 
    reset : in bit_1; -- reset 
    en : in bit_1; -- enables PE

    -- inputs for the MAC operation
    in_data : in bit_8; -- can be data input or activation 
    in_weight : in bit_8; -- weight
    
    -- outputs for the MAC operation
    out_data : out bit_8;
    out_weight : out bit_8;
    result_register : out bit_32
);
end processing_element;
    
architecture behaviour of processing_element is

    -- -- initialise signals and vairables for the internal registers
    signal data, weight : bit_8 := (others => '0');
    signal accumulator : unsigned(31 downto 0) := (others => '0'); 


begin
    process(clk, reset)
        variable multiplication : unsigned(31 downto 0);
    begin
        if reset = '1' then
            data <= (others => '0');
            weight <= (others => '0');
            accumulator <= (others => '0');

        -- Only process data if PE is enabled 
        elsif rising_edge(clk) then
            -- Synchronous 
            if en = '1' then
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
        else 
            -- do nothing
        end if;
        
    end process;

    -- output assignment
    out_data <= data;
    out_weight <= weight;
    result_register <= std_logic_vector(accumulator);
end behaviour;