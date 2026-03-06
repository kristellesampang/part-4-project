-- Processing Element INT8 (LUT-based) -- Project #43 (2025)
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.custom_types_int8.all;

entity processing_element_int8 is
port(
    clk          : in  std_logic;
    reset        : in  std_logic;
    en           : in  std_logic;
    in_data      : in  std_logic_vector(7 downto 0);
    in_weight    : in  std_logic_vector(7 downto 0);
    out_data     : out std_logic_vector(7 downto 0);
    out_weight   : out std_logic_vector(7 downto 0);
    result_register : out std_logic_vector(31 downto 0)
);
end processing_element_int8;

architecture behaviour of processing_element_int8 is
    signal data_reg        : signed(7 downto 0) := (others => '0');
    signal weight_reg      : signed(7 downto 0) := (others => '0');
    signal mult_result_reg : signed(15 downto 0) := (others => '0');
    signal accumulator_reg : signed(31 downto 0) := (others => '0');

    -- Force LUT-based multiplication (no DSP blocks)
    attribute use_dsp : string;
    attribute use_dsp of mult_result_reg : signal is "no";
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
                data_reg        <= signed(in_data);
                weight_reg      <= signed(in_weight);
                mult_result_reg <= data_reg * weight_reg;
                accumulator_reg <= accumulator_reg + resize(mult_result_reg, 32);

            -- elsif en = '0' then
            --     accumulator_reg <= (others => '0'); -- Clear accumulator when disabled
            --     mult_result_reg <= (others => '0'); -- Clear multiplier result when disabled
            --     data_reg        <= (others => '0'); -- Clear data register when disabled
            --     weight_reg      <= (others => '0'); -- Clear weight register when disabled
            end if;
        end if;
    end process;

    out_data        <= std_logic_vector(data_reg);
    out_weight      <= std_logic_vector(weight_reg);
    result_register <= std_logic_vector(accumulator_reg);
end behaviour;