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
    matrix_a : in matrix_4x4;
    matrix_b : in matrix_4x4;

    -- outputs for the MAC operation
    output : out matrix_4x4;
)
end systolic_array;
    
architecture behaviour of systolic_array is
    -- initialise signals and vairables


begin
    process(clk, reset) 
        begin
           
    end process;

    -- output assignment


    

end behaviour;