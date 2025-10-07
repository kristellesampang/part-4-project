-- Control Unit for the Systolic Array
-- Project #43 (2025)
library ieee; 
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.custom_types.all;

entity control_unit is 
port(
    clk : in bit_1;
    reset : in bit_1;
    ready : in bit_1;
    -- completed : out bit_1;

    matrix_data   : in systolic_array_matrix_input;
    matrix_weight : in systolic_array_matrix_input;

    data_shift    : out input_shift_matrix;
    weight_shift  : out input_shift_matrix;
    cycle_count   : out integer;
    PE_enabled_mask : out enabled_PE_matrix;
    

    active_rows : in integer;
    active_cols : in integer;
    active_k : in integer
);
end control_unit;

architecture behaviour of control_unit is
    signal data_reg   : input_shift_matrix := (others => (others => '0'));
    signal weight_reg : input_shift_matrix := (others => (others => '0'));
    signal count      : integer := 0;
    signal mask_internal : enabled_PE_matrix := (others => (others => '0'));
begin
    process(clk, reset)
    begin
        if reset = '1' then
            count <= 0;
            -- Clearing all the elements of the registers upon reset
            for i in 0 to N-1 loop
                data_reg(i)   <= (others => '0');
                weight_reg(i) <= (others => '0');
            end loop;
            for i in 0 to N-1 loop
                for j in 0 to N-1 loop
                    mask_internal(i,j) <= '0';
                end loop;
            end loop;
        elsif rising_edge(clk) and ready = '1' then
            count <= count + 1;
            

            -- DATA (matrix A) (left->right)
            for i in 0 to active_rows-1 loop
                -- stagger and timing logic
                if (count >= i) and (count < i + active_k) then
                    data_reg(i) <= matrix_data(i, count - i);
                -- fill the rest with zeros
                else
                    data_reg(i) <= (others => '0');
                end if;
            end loop;


            -- WEIGHT (matrix B) -> (top->bottom)
            for j in 0 to active_cols-1 loop
                -- stagger and timing logic
                if (count >= j) and (count < j + active_k) then
                    weight_reg(j) <= matrix_weight(count - j, j);
                -- fill the rest with zeros
                else
                    weight_reg(j) <= (others => '0');
                end if;
            end loop;

            -- PE enable mask for power optimisation
            for i in 0 to N-1 loop
                for j in 0 to N-1 loop
                    if (i < active_rows) and (j < active_cols) then
                        mask_internal(i,j) <= '1';
                    else
                        mask_internal(i,j) <= '0';
                    end if;
                end loop;
            end loop;



        end if;
        
    end process;

    
    


    data_shift      <= data_reg;
    weight_shift    <= weight_reg;
    PE_enabled_mask <= mask_internal;
    cycle_count     <= count;
end behaviour;
