library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.custom_types.all;

entity processing_element is 
generic (
    DATA_WIDTH : integer := 16;
    ACC_WIDTH  : integer := 32
);
port(
    clk   : in std_logic; 
    reset : in std_logic; 
    en    : in std_logic; 

    in_data   : in std_logic_vector(DATA_WIDTH-1 downto 0);
    in_weight : in std_logic_vector(DATA_WIDTH-1 downto 0);
    
    out_data        : out std_logic_vector(DATA_WIDTH-1 downto 0);
    out_weight      : out std_logic_vector(DATA_WIDTH-1 downto 0);
    result_register : out std_logic_vector(ACC_WIDTH-1 downto 0)
);
end processing_element;
    
architecture behaviour of processing_element is
    signal data_reg        : signed(DATA_WIDTH-1 downto 0) := (others => '0');
    signal weight_reg      : signed(DATA_WIDTH-1 downto 0) := (others => '0');
    -- Multiplier output width is always 2x input width
    signal mult_result_reg : signed((DATA_WIDTH*2)-1 downto 0) := (others => '0');
    signal accumulator_reg : signed(ACC_WIDTH-1 downto 0) := (others => '0');

    -- Force DSP-based multiplication for higher bitwidths
    attribute use_dsp : string;
    attribute use_dsp of mult_result_reg : signal is "yes";

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

                -- The stripping algorithm logic resides in the Control Unit, 
                -- but the PE must be ready to accumulate at any width
                accumulator_reg <= accumulator_reg + resize(mult_result_reg, ACC_WIDTH);

            end if;
        end if;
    end process;

    out_data        <= std_logic_vector(data_reg);
    out_weight      <= std_logic_vector(weight_reg);
    result_register <= std_logic_vector(accumulator_reg); 
end behaviour;