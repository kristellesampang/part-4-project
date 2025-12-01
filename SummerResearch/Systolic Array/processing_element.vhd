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
    in_data : in bit_16; -- can be data input or activation 
    in_weight : in bit_16; -- weight
    
    -- outputs for the MAC operation
    out_data : out bit_16;
    out_weight : out bit_16;
    result_register : out bit_64
);
end processing_element;
    
architecture behaviour of processing_element is
    -- Internal signals for pipelining
    signal data_reg        : signed(15 downto 0) := (others => '0');
    signal weight_reg      : signed(15 downto 0) := (others => '0');
    signal mult_result_reg : signed(31 downto 0) := (others => '0');  --16-bit x 16-bit = 32-bit
    signal accumulator_reg : signed(63 downto 0) := (others => '0');

begin
    process(clk)
    begin
        if rising_edge(clk) then
            if reset = '1' then
                data_reg        <= (others => '0');
                weight_reg      <= (others => '0');
                mult_result_reg <= (others => '0');
                accumulator_reg <= (others => '0');
            elsif en = '1' then
                -- Register inputs and perform multiplication
                data_reg        <= signed(in_data);
                weight_reg      <= signed(in_weight);
                mult_result_reg <= data_reg * weight_reg;

                -- Add the result from the previous cycle's multiplication
                accumulator_reg <= accumulator_reg + resize(mult_result_reg, accumulator_reg'length);
            end if;
        end if;
    end process;

    -- Output assignment (pass registered values through)
    out_data        <= std_logic_vector(data_reg);
    out_weight      <= std_logic_vector(weight_reg);
    result_register <= std_logic_vector(accumulator_reg); 

end behaviour;