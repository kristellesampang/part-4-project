-- Project 43

library ieee;
use ieee.std_logic_1164.all;

package matrix_type is
    subtype byte is unsigned(7 downto 0);
    type matrix_4x4 is array (0 to 3, 0 to 3) of byte;
    type matrix_3x3 is array (0 to 2, 0 to 2) of byte; 
end matrix_type; 