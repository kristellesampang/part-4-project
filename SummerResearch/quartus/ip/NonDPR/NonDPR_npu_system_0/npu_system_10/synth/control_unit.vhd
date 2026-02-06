library ieee; 
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.custom_types.all;

entity control_unit is 
port(
    clk : in bit_1;
    reset : in bit_1;
    ready : in bit_1; -- Retained as a master "START" signal

    matrix_data   : in systolic_array_matrix_input;
    matrix_weight : in systolic_array_matrix_input;

    data_shift    : out input_shift_matrix;
    weight_shift  : out input_shift_matrix;
    cycle_count   : out integer;
    PE_enabled_mask : out enabled_PE_matrix;
    completed     : out bit_1; -- Signal to indicate completion of the operation (optional, can be derived from cycle_count and max    


    active_rows : in integer;
    active_cols : in integer;
    active_k : in integer
);
end control_unit;

architecture behaviour of control_unit is
    signal data_reg   : input_shift_matrix := (others => (others => '0'));
    signal weight_reg : input_shift_matrix := (others => (others => '0'));
    -- Initial start signal (we set it high when ready is pulsed, and let it run until reset)
    signal run_enable : bit_1 := '0'; 
    signal count      : integer := 0;
    signal mask_internal : enabled_PE_matrix := (others => (others => '0'));
    signal completed_internal : bit_1 := '0';

    -- Pre-calculate the total cycles needed for one operation (M+N+K-2)
    -- This is the fixed runtime needed for the control logic.
    signal max_run_cycles : integer := 0; 
begin
    -- The maximum run cycle is calculated here, dynamically based on active inputs
    max_run_cycles <= active_rows + active_cols + active_k - 2;

    process(clk, reset)
    begin
        if reset = '1' then
            count <= 0;
            run_enable <= '0';
            completed_internal <= '0';
            
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
    
        -- When reset is released and 'ready' pulses, the internal state goes to RUN
        elsif rising_edge(clk) then
            if ready = '1' then
                run_enable <= '1';
                completed_internal <= '0';
            end if;

            -- Only execute the control logic if we are running and haven't hit the maximum cycle count
            if run_enable = '1' then
                if count >= max_run_cycles then
                    run_enable <= '0'; -- Stop the operation after the max cycles
                    completed_internal <= '1'; -- Signal completion
                else
                    count <= count + 1; -- Increment cycle count
                end if;
            end if;
            -- --- DATA (matrix A) (left->right) ---
            -- Logic relies on the external 'ready' pulse for the input window.
            -- When ready is '0', the loop naturally fills unused input cycles with zeros (u16(0)).
            for i in 0 to N-1 loop
                if i < active_rows then
                    -- stagger and timing logic
                    -- We use the internal 'count' for the timing offset.
                    if run_enable = '1' and (count >= i) and (count < i + active_k) then
                        data_reg(i) <= matrix_data(i, count - i);
                    -- fill the rest with zeros after the stream or if ready/active inputs stop
                    else
                        data_reg(i) <= (others => '0');
                    end if;
                else
                    data_reg(i) <= (others => '0');
                end if;
            end loop;


            -- --- WEIGHT (matrix B) -> (top->bottom) ---
            for j in 0 to N-1 loop
                if j < active_cols then
                    -- stagger and timing logic
                    if run_enable = '1' and (count >= j) and (count < j + active_k) then
                        weight_reg(j) <= matrix_weight(count - j, j);
                    -- fill the rest with zeros
                    else
                        weight_reg(j) <= (others => '0');
                    end if;
                else
                    weight_reg(j) <= (others => '0');
                end if;
            end loop;

            -- --- PE enable mask for power optimisation (Only set once on start) ---
            if ready = '1' and count = 0 then
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
        end if; -- end rising_edge(clk)
    end process;

    completed <= completed_internal;
    data_shift      <= data_reg;
    weight_shift    <= weight_reg;
    PE_enabled_mask <= mask_internal;
    cycle_count     <= count;
end behaviour;