-- Project 43

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;  -- Required for 'unsigned'

package matrix_type is
    subtype byte is std_logic_vector(7 downto 0);
    subtype result_reg is std_logic_vector(31 downto 0);
    type matrix_4x4 is array (0 to 3, 0 to 3) of byte;
    type matrix_3x3_input is array (0 to 2, 0 to 2) of byte; 
    type matrix_3x3_output is array(0 to 2, 0 to 2) of result_reg;
end matrix_type;