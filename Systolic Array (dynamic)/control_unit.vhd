-- Control Unit for the Systolic Array
-- Project #43 (2025)
library ieee; 
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.custom_types.all;

entity control_unit is 
port(
    
    -- clock and reset signals
    clk : in bit_1; -- synchronous 
    reset : in bit_1; -- reset 

    -- inputs for the MAC operation
    matrix_data : in systolic_array_matrix_input; -- data 
    matrix_weight : in systolic_array_matrix_input; -- weight 

    -- outputs for the MAC operation
    data_shift    : out input_shift_matrix;
    weight_shift  : out input_shift_matrix;

    cycle_count   : out integer;

    -- Per-cycle enable mask for every PE
    PE_enabled_mask : out enabled_PE_matrix
);
end control_unit;

architecture behaviour of control_unit is 
    -- temp signals 
    signal data_reg   : input_shift_matrix := (others => (others => '0'));
    signal weight_reg : input_shift_matrix := (others => (others => '0'));
    signal count      : integer := 0;
    signal mask_internal  : enabled_PE_matrix := (others => (others => '0'));
    
begin

    process(clk, reset)
        variable array_size : integer;
    begin
        -- get the array size
        array_size := matrix_data'length;

        -- reset all signals 
        if reset = '1' then
            count <= 0;

            for i in 0 to N-1 loop
                data_reg(i)   <= (others => '0');
                weight_reg(i) <= (others => '0');
            end loop;

            for i in 0 to N-1 loop
                for j in 0 to N-1 loop
                    mask_internal(i,j) <= '0';
                end loop;
            end loop;
  
        elsif rising_edge(clk) then
            -- increment clock cycle count at every rising edge
            count <= count + 1;

            -- feeds the data into the systolic array depending on how the input matrices are 
            -- feeds the data (the left most side)
            for i in 0 to array_size - 1 loop
                if count >= i and count < i + array_size then
                    data_reg(i) <= matrix_data(i, count - i);
                else
                    data_reg(i) <= (others => '0');
                end if;
            end loop;

            -- feeds the weight (from the top to bottom)
            for j in 0 to array_size - 1 loop
                if count >= j and count < j + array_size then
                    weight_reg(j) <= matrix_weight(count - j, j);
                else
                    weight_reg(j) <= (others => '0');
                end if;
            end loop;


            -- Per-cycle enable mask
            for i in 0 to array_size-1 loop
                for j in 0 to array_size-1 loop
                    if (count >= i) and (count < i + array_size) and (count >= j) and (count < j + array_size) then
                        if (matrix_data(i, count - i) /= x"00") and (matrix_weight(count - j, j) /= x"00") then
                            mask_internal(i, j) <= '1';
                        else 
                            mask_internal(i, j) <= '0';
                        end if;
                    else
                        mask_internal(i, j) <= '0';
                    end if;
                end loop;
            end loop;
        end if;
    end process;

    PE_enabled_mask <= mask_internal;
    data_shift   <= data_reg;
    weight_shift <= weight_reg;
    PE_enabled_mask <= mask_internal;
    cycle_count  <= count;

end behaviour; 