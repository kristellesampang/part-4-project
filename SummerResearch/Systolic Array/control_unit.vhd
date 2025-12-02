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
    

    active_rows : in integer;
    active_cols : in integer;
    active_k : in integer
);
end control_unit;

architecture behaviour of control_unit is
    -- ... (signal declarations) ...

    signal run_enable : bit_1 := '0'; -- Internal flag for running state
    signal count      : integer := 0;
    -- ... (rest of declarations) ...

    -- Pre-calculate max_run_cycles concurrently
    signal max_run_cycles : integer := 0; 
begin
    max_run_cycles <= active_rows + active_cols + active_k - 2;

    process(clk, reset)
    begin
        if reset = '1' then
            count <= 0;
            run_enable <= '0';
            -- ... (clearing data_reg, weight_reg, mask_internal) ...
    
        elsif rising_edge(clk) then
            
            -- 1. STATE CONTROL: START, COUNT, and STOP Logic
            -- Start: Triggered by one external pulse
            if ready = '1' then
                run_enable <= '1';
            end if;

            -- Stop: Self-terminate after the required latency is reached
            if run_enable = '1' and count < max_run_cycles then
                count <= count + 1;
            elsif count = max_run_cycles then
                run_enable <= '0'; -- Turn off self-state once calculation is finished
            end if;


            -- 2. DATA SHIFTING (Data Input and Weight Input)
            -- Data and Weights flow ONLY during the input stream window (0 to active_k-1 cycles).
            for i in 0 to N-1 loop
                if i < active_rows then
                    -- Timing check relies on run_enable, NOT external ready
                    if run_enable = '1' AND (count >= i) AND (count < i + active_k) then
                        data_reg(i) <= matrix_data(i, count - i);
                    else
                        data_reg(i) <= (others => '0');
                    end if;
                -- ... (rest of data shifting/zero padding) ...
            end loop;

            -- ... (Weight shifting logic - apply similar run_enable guard) ...

            -- 3. PE MASK ACTIVATION (Triggered ONCE on start)
            -- We keep this triggered by the external 'ready' pulse for simplicity.
            if ready = '1' and count = 0 then
                for i in 0 to N-1 loop
                    for j in 0 to N-1 loop
                        -- **This logic must be correct to enable the 3x3 region**
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
-- ... (output assignments) ...
end behaviour;