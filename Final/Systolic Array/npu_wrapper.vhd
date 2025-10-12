library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.custom_types.all;

entity npu_wrapper is
    port (
        -- System signals
        clk     : in  bit_1;
        reset   : in  bit_1;

        -- Control interface
        start   : in  bit_1;
        done    : out bit_1;

        -- tb
        rows : out integer range 0 to N-1;
        cols : out integer range 0 to N-1;
        read_rom_counter : out integer range 0 to N*N + 5;

        -- Read Interface for accessing the on-chip memory
        read_address  : in  bit_7;
        read_data     : out bit_32
    );
end entity npu_wrapper;

architecture rtl of npu_wrapper is

    -- Component for the top-level systolic array
    component top_level_systolic_array is
        port (
            clk         : in  bit_1;
            reset       : in  bit_1;
            ready       : in  bit_1;
            matrix_data : in  systolic_array_matrix_input;
            matrix_weight: in  systolic_array_matrix_input;
            active_rows : in  integer;
            active_cols : in  integer;
            active_k    : in  integer;
            output      : out systolic_array_matrix_output;
            cycle_count : out integer
        );
    end component;

    -- Components for the data and weight ROMs
    component data_rom is
        port (
            address : in bit_7;
            clock   : in  bit_1;
            q       : out bit_8
        );
    end component;

    component weight_rom is
        port (
            address : in  bit_7;
            clock   : in  bit_1;
            q       : out bit_8
        );
    end component;

    -- Internal signals
    signal rom_addr              : unsigned(6 downto 0) := (others => '0');
    signal data_from_rom         : bit_8;
    signal weight_from_rom       : bit_8;

    signal matrix_data_buffer    : systolic_array_matrix_input := (others => (others => (others => '0')));
    signal matrix_weight_buffer  : systolic_array_matrix_input := (others => (others => (others => '0')));

    signal active_m_sig          : integer range 0 to N;
    signal active_n_sig          : integer range 0 to N;
    signal active_k_sig          : integer range 0 to N;
    signal sa_cycle_count        : integer;
    signal sa_ready_sig          : bit_1 := '0';
    signal expected_latency      : integer;

    -- On-chip memory for storing the final result
    signal internal_result_matrix: systolic_array_matrix_output := (others => (others => (others => '0')));
    -- Signal to capture the output from the systolic array instance
    signal sa_result_internal    : systolic_array_matrix_output;

    -- FSM for control logic
    type state_t is (S_IDLE, S_LOAD_PARAMS, S_LOAD_MATRICES, S_EXECUTE, S_WAIT_DONE, S_FINISH);
    signal current_state         : state_t := S_IDLE;

begin

    -- Instantiate the ROMs to hold matrix data and weights
    data_rom_inst : component data_rom
        port map (
            address => std_logic_vector(rom_addr),
            clock   => clk,
            q       => data_from_rom
        );

    weight_rom_inst : component weight_rom
        port map (
            address => std_logic_vector(rom_addr),
            clock   => clk,
            q       => weight_from_rom
        );

    -- Instantiate the top-level systolic array system
    systolic_array_inst : component top_level_systolic_array
        port map (
            clk          => clk,
            reset        => reset,
            ready        => sa_ready_sig,
            matrix_data  => matrix_data_buffer,
            matrix_weight=> matrix_weight_buffer,
            active_rows  => active_m_sig,
            active_cols  => active_n_sig,
            active_k     => active_k_sig,
            output       => sa_result_internal,
            cycle_count  => sa_cycle_count
        );

    -- Control FSM process
    fsm_proc : process(clk, reset)
        variable rom_addr_counter : integer range 0 to N*N + 5 := 0;
        variable target_row : integer range 0 to N-1;
        variable target_col : integer range 0 to N-1;
    begin
        if reset = '1' then
            current_state      <= S_IDLE;
            rom_addr_counter   := 0;
            rom_addr           <= (others => '0');
            sa_ready_sig       <= '0';
            done               <= '0';
            active_m_sig       <= 0;
            active_n_sig       <= 0;
            active_k_sig       <= 0;
            matrix_data_buffer <= (others => (others => (others => '0')));
            matrix_weight_buffer <= (others => (others => (others => '0')));

        elsif rising_edge(clk) then
            done <= '0'; -- Default done to low, pulse high for one cycle
            
            case current_state is
                when S_IDLE =>
                    if start = '1' then
                        rom_addr_counter := 0;
                        current_state    <= S_LOAD_PARAMS;
                    end if;

                when S_LOAD_PARAMS =>
                    -- sa_ready_sig <= '1';
                    rom_addr <= to_unsigned(rom_addr_counter, rom_addr'length);
                    
                    -- ROMs have a 1-cycle read latency
                    if rom_addr_counter = 3 then
                        active_m_sig <= to_integer(unsigned(data_from_rom));
                    elsif rom_addr_counter = 4 then
                        active_n_sig <= to_integer(unsigned(data_from_rom));
                    elsif rom_addr_counter = 5 then
                        active_k_sig <= to_integer(unsigned(data_from_rom));
                        current_state    <= S_LOAD_MATRICES;
                    end if;

                    rom_addr_counter := rom_addr_counter + 1;

                when S_LOAD_MATRICES =>
                -- sa_ready_sig <= '1';
                    rom_addr <= to_unsigned(rom_addr_counter, rom_addr'length);

                    -- De-serialize 1D ROM data into 2D matrix buffers
                    target_row := (rom_addr_counter - 6) / N;
                    target_col := (rom_addr_counter - 6) mod N;

                    if target_row < N then
                        matrix_data_buffer(target_row, target_col)   <= data_from_rom;
                        matrix_weight_buffer(target_row, target_col) <= weight_from_rom;
                    end if;

                    if rom_addr_counter >= (N*N + 5) then
                        current_state <= S_EXECUTE;
                    else
                        rom_addr_counter := rom_addr_counter + 1;
                    end if;
                    
                when S_EXECUTE =>
                    sa_ready_sig <= '1';
                    current_state <= S_WAIT_DONE;

                when S_WAIT_DONE =>
                    -- Keep ready asserted throughout the computation
                    sa_ready_sig <= '1';
                    -- Latency is calculated based on the active dimensions read from ROM
                    expected_latency <= active_m_sig + active_n_sig + active_k_sig - 2;
                    if sa_cycle_count >= expected_latency then
                        current_state <= S_FINISH;
                    end if;

                when S_FINISH =>
                    sa_ready_sig <= '1';
                    done        <= '1'; -- Signal completion
                    
                    current_state <= S_IDLE;
                    
            end case;
        end if;

    read_rom_counter <= rom_addr_counter;
    internal_result_matrix <= sa_result_internal;
    end process fsm_proc;

    -- This process implements the read access to the on-chip memory
    read_logic_proc : process(read_address, internal_result_matrix)
        variable row_idx : integer range 0 to N-1;
        variable col_idx : integer range 0 to N-1;
    begin
        -- Decode the 1D read_address into 2D matrix indices

        if to_integer(unsigned(read_address)) < 8 then
            row_idx := 0; -- Prevent uninitialized variable usage
        else
            row_idx := to_integer(unsigned(read_address)) / N;
        end if;
        col_idx := to_integer(unsigned(read_address)) mod N;


        

        -- Place the selected data element onto the output bus
        -- Assumes systolic_array_matrix_output contains signed elements of 32 bits
        read_data <= std_logic_vector(internal_result_matrix(row_idx, col_idx)); 

        rows <= row_idx;
        cols <= col_idx;
    end process read_logic_proc;

end architecture rtl;