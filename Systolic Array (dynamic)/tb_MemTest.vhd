library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.custom_types.all;

entity tb_MemTest is
end entity tb_MemTest;

architecture sim of tb_MemTest is

    constant C_ACTIVE_M : integer := 8;
    constant C_ACTIVE_K : integer := 8;
    constant C_ACTIVE_N : integer := 8;

    component data_bram is port ( 
        clock: in std_logic; 
        address: in std_logic_vector(3 downto 0); 
        data: in std_logic_vector(31 downto 0); 
        wren: in std_logic; 
        q: out std_logic_vector(31 downto 0) ); 
    end component;


    component weight_bram is port ( 
        clock: in std_logic; 
        address: in std_logic_vector(3 downto 0); 
        data: in std_logic_vector(31 downto 0); 
        wren: in std_logic; 
        q: out std_logic_vector(31 downto 0) ); 
    end component;

    signal clk   : std_logic := '0';
    signal reset : std_logic := '1';
    constant CLK_PERIOD : time := 20 ns;

    signal data_bram_addr, weight_bram_addr : std_logic_vector(3 downto 0);
    signal data_bram_q, weight_bram_q       : std_logic_vector(31 downto 0);
    
    signal npu_start            : std_logic := '0';
    signal npu_matrix_data_in   : systolic_array_matrix_input;
    signal npu_matrix_weight_in : systolic_array_matrix_input;
    signal npu_matrix_result    : systolic_array_matrix_output;

    type state_t is (IDLE, LOAD_DATA, START_COMPUTE, WAIT_DONE, CHECK_RESULTS);
    signal state        : state_t := IDLE;
    signal addr_counter : natural := 0;
    signal wait_counter : natural := 0;

begin
    clk <= not clk after CLK_PERIOD / 2;
    reset <= '0' after 5 * CLK_PERIOD;

    inst_data_bram : component data_bram port map ( 
        clock => clk, 
        address => data_bram_addr, 
        data => (others => '0'), 
        wren => '0', 
        q => data_bram_q );
    inst_weight_bram : component weight_bram port map ( 
        clock => clk, 
        address => weight_bram_addr,
        data => (others => '0'), 
        wren => '0', 
        q => weight_bram_q );

    inst_npu : entity work.top_level_systolic_array
        port map (
            clk               => clk,
            reset             => reset,
            start             => npu_start,
            done              => open, -- Done signal is not used
            active_rows       => C_ACTIVE_M,
            active_k          => C_ACTIVE_K,
            active_cols       => C_ACTIVE_N,
            matrix_data_in    => npu_matrix_data_in,
            matrix_weight_in  => npu_matrix_weight_in,
            matrix_result_out => npu_matrix_result
        );

    process(clk)
        variable reads_needed     : natural;
        variable latency_cycles   : natural;
        variable data_row, data_col, weight_row, weight_col : natural;
        -- Pipelined registers to handle 2-cycle latency
        variable data_q_d1, weight_q_d1 : std_logic_vector(31 downto 0);
    begin
        if rising_edge(clk) then
            if reset = '1' then
                state <= IDLE;
                addr_counter <= 0;
                wait_counter <= 0;
                npu_start <= '0';
            else
                npu_start <= '0';

                case state is
                    when IDLE =>
                        if reset = '0' then
                            addr_counter <= 0;
                            --report "FSM: Starting test. Moving to load data." severity note;
                            state <= LOAD_DATA;
                        end if;

                    -- we gon do this in one state so that its read together
                    when LOAD_DATA =>
                        reads_needed := (C_ACTIVE_M * C_ACTIVE_K + 3) / 4; -- 16 for 8x8
                        
                        -- Pipeline Stage 1: Register the BRAM output
                        data_q_d1   := data_bram_q;
                        weight_q_d1 := weight_bram_q;

                        if addr_counter < reads_needed then
                            -- Set the address for the next read
                            data_bram_addr   <= std_logic_vector(to_unsigned(addr_counter, 4));
                            weight_bram_addr <= std_logic_vector(to_unsigned(addr_counter, 4));

                            -- Latch the data that is now valid from TWO cycles ago
                            if addr_counter >= 1 then
                                for i in 0 to 3 loop
                                    data_row := ((addr_counter - 1) * 4 + i) / C_ACTIVE_K;
                                    data_col := ((addr_counter - 1) * 4 + i) mod C_ACTIVE_K;
                                    if data_row < C_ACTIVE_M then
                                        npu_matrix_data_in(data_row, data_col) <= data_q_d1((3-i)*8+7 downto (3-i)*8);
                                    end if;
                                    
                                    weight_row := ((addr_counter - 1) * 4 + i) / C_ACTIVE_N;
                                    weight_col := ((addr_counter - 1) * 4 + i) mod C_ACTIVE_N;
                                    if weight_row < C_ACTIVE_K then
                                        npu_matrix_weight_in(weight_row, weight_col) <= weight_q_d1((3-i)*8+7 downto (3-i)*8);
                                    end if;
                                end loop;
                            end if;
                            
                            addr_counter <= addr_counter + 1;
                        else
                            -- After the loop, we need one more cycle to latch the final data
                            for i in 0 to 3 loop
                                data_row := ((addr_counter - 1) * 4 + i) / C_ACTIVE_K;
                                data_col := ((addr_counter - 1) * 4 + i) mod C_ACTIVE_K;
                                if data_row < C_ACTIVE_M then
                                    npu_matrix_data_in(data_row, data_col) <= data_q_d1((3-i)*8+7 downto (3-i)*8);
                                end if;
                                
                                weight_row := ((addr_counter - 1) * 4 + i) / C_ACTIVE_N;
                                weight_col := ((addr_counter - 1) * 4 + i) mod C_ACTIVE_N;
                                if weight_row < C_ACTIVE_K then
                                    npu_matrix_weight_in(weight_row, weight_col) <= weight_q_d1((3-i)*8+7 downto (3-i)*8);
                                end if;
                            end loop;

                            --report "FSM: All matrices loaded. Moving to start compute." severity note;
                            state <= START_COMPUTE;
                        end if;

                    when START_COMPUTE =>
                        npu_start <= '1';
                        wait_counter <= 0;
                        --report "FSM: Starting NPU computation. Moving to WAIT_DONE." severity note;
                        state <= WAIT_DONE;
                        
                    when WAIT_DONE =>
                        latency_cycles := C_ACTIVE_M + C_ACTIVE_K + C_ACTIVE_N - 2;
                        if wait_counter < latency_cycles then
                            wait_counter <= wait_counter + 1;
                        else
                            --report "FSM: Computation time elapsed. Moving to CHECK_RESULTS." severity note;
                            state <= CHECK_RESULTS;
                        end if;

                    when CHECK_RESULTS =>
                        --report "FSM: TEST PASSED! Simulation finished." severity failure;
                        state <= CHECK_RESULTS;

                end case;
            end if;
        end if;
    end process;

end architecture sim;

