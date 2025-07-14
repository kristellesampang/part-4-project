library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity array_length_tb is
end entity;

architecture behavior of array_length_tb is

    -- Define an array type
    type my_array_type is array(0 to 9) of integer;

    -- Declare a signal of that type
    signal my_array : my_array_type := (others => 0);

begin

    process
        variable length_of_array : integer;
    begin
        -- Get the length using 'length
        length_of_array := my_array'length;

        -- Display the value to the simulator console
        report "Array length is " & integer'image(length_of_array);

        wait;  -- Stop simulation
    end process;

end architecture;
