-- Control Unit INT8 -- Project #43 (2025)
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.custom_types_int16.all;

entity control_unit_int16 is
port(
    clk         : in  bit_1;
    reset       : in  bit_1;
    ready       : in  bit_1;

    matrix_data   : in  systolic_array_matrix_input_int16;
    matrix_weight : in  systolic_array_matrix_input_int16;

    data_shift      : out input_shift_matrix_int16;
    weight_shift    : out input_shift_matrix_int16;
    cycle_count     : out integer;
    PE_enabled_mask : out enabled_PE_matrix_int16;
    completed       : out bit_1;

    active_rows : in integer;
    active_cols : in integer;
    active_k    : in integer
);
end control_unit_int16;

architecture behaviour of control_unit_int16 is
    signal data_reg   : input_shift_matrix_int16 := (others => (others => '0'));
    signal weight_reg : input_shift_matrix_int16 := (others => (others => '0'));
    signal run_enable : bit_1 := '0';
    signal count      : integer := 0;
    signal mask_internal      : enabled_PE_matrix_int16 := (others => (others => '0'));
    signal completed_internal : bit_1 := '0';
    signal max_run_cycles     : integer := 0;
begin
    

    process(clk, reset)
    begin
        if reset = '1' then
            count <= 0;
            run_enable <= '0';
            completed_internal <= '0';
            
            for i in 0 to N16-1 loop
                data_reg(i)   <= (others => '0');
                weight_reg(i) <= (others => '0');
            end loop;
            for i in 0 to N16-1 loop
                for j in 0 to N16-1 loop
                    mask_internal(i,j) <= '0';
                end loop;
            end loop;

        elsif rising_edge(clk) then
            if ready = '1' then
                run_enable <= '1';
                completed_internal <= '0';
                count <= 0;
                max_run_cycles <= active_rows + active_cols + active_k - 2;
            end if;

            if run_enable = '1' then
                if count >= max_run_cycles then
                    run_enable <= '0';
                    completed_internal <= '1';
                else
                    count <= count + 1;
                end if;
            end if;

            for i in 0 to N16-1 loop
                if i < active_rows then
                    if run_enable = '1' and (count >= i) and (count < i + active_k) then
                        data_reg(i) <= matrix_data(i, count - i);
                    else
                        data_reg(i) <= (others => '0');
                    end if;
                else
                    data_reg(i) <= (others => '0');
                end if;
            end loop;

            for j in 0 to N16-1 loop
                if j < active_cols then
                    if run_enable = '1' and (count >= j) and (count < j + active_k) then
                        weight_reg(j) <= matrix_weight(count - j, j);
                    else
                        weight_reg(j) <= (others => '0');
                    end if;
                else
                    weight_reg(j) <= (others => '0');
                end if;
            end loop;

            if ready = '1' and count = 0 then
                for i in 0 to N16-1 loop
                    for j in 0 to N16-1 loop
                        if (i < active_rows) and (j < active_cols) then
                            mask_internal(i,j) <= '1';
                        else
                            mask_internal(i,j) <= '0';
                        end if;
                    end loop;
                end loop;
            end if;
        end if;
    end process;

    completed       <= completed_internal;
    data_shift      <= data_reg;
    weight_shift    <= weight_reg;
    PE_enabled_mask <= mask_internal;
    cycle_count     <= count;
end behaviour;