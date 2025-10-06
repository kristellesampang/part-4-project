library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.custom_types.all;

entity tb_memory is
end tb_memory;

architecture sim of tb_memory is

    -- Component Declarations
    component data_rom is
        port (address : in std_logic_vector(5 downto 0); clock : in std_logic; q : out std_logic_vector(7 downto 0));
    end component;
    component weight_rom is
        port (address : in std_logic_vector(5 downto 0); clock : in std_logic; q : out std_logic_vector(7 downto 0));
    end component;
    
    -- Constants
    constant MAX_ACTIVE_ROWS : integer := 8;
    constant MAX_ACTIVE_COLS : integer := 8;
    constant CLK_PER : time := 20 ns;

    -- Signals
    signal clk   : std_logic := '0';
    signal reset : std_logic;
    signal data_rom_addr     : std_logic_vector(5 downto 0);
    signal data_rom_q        : std_logic_vector(7 downto 0);
    signal weight_rom_addr   : std_logic_vector(5 downto 0);
    signal weight_rom_q      : std_logic_vector(7 downto 0);
    signal matrix_data_sig   : systolic_array_matrix_input := (others => (others => (others => '0')));
    signal matrix_weight_sig : systolic_array_matrix_input := (others => (others => (others => '0')));
    signal result_matrix_sig : systolic_array_matrix_output;
    signal cycle_count_sig   : integer;
    
begin
    clk <= not clk after CLK_PER / 2;

    -- Instantiate Components
    Data_ROM_inst : component data_rom port map (address => data_rom_addr, clock => clk, q => data_rom_q);
    Weight_ROM_inst : component weight_rom port map (address => weight_rom_addr, clock => clk, q => weight_rom_q);
    DUT: entity work.top_level_systolic_array
    port map (
        clk           => clk,
        reset         => reset,
        matrix_data   => matrix_data_sig,
        matrix_weight => matrix_weight_sig,
        output        => result_matrix_sig,
        cycle_count   => cycle_count_sig,
        active_rows   => MAX_ACTIVE_ROWS,
        active_cols   => MAX_ACTIVE_COLS
    );

    -- Main Test Process as an FSM
    LatencyTest_FSM: process
        -- Define states and variables here, before 'begin'
        type state_t is (IDLE, APPLY_RESET, LOAD_DATA, WAIT_FOR_COMPLETION);
        variable current_state : state_t := IDLE;
        variable addr_counter  : integer := 0;
        variable shifted_index : integer;
        variable target_row    : integer;
        variable target_col    : integer;
    begin
        case current_state is
            when IDLE =>
                wait until rising_edge(clk);
                current_state := APPLY_RESET;

            when APPLY_RESET =>
                reset <= '1';
                wait until rising_edge(clk);
                wait until rising_edge(clk);
                reset <= '0';
                current_state := LOAD_DATA;
                addr_counter := 0;

            when LOAD_DATA =>
                -- Present the address
                data_rom_addr   <= std_logic_vector(to_unsigned(addr_counter, data_rom_addr'length));
                weight_rom_addr <= std_logic_vector(to_unsigned(addr_counter, weight_rom_addr'length));
                
                -- Wait for the next clock edge
                wait until rising_edge(clk);
                
                -- Only start writing after the first two reads to create the shift
                if addr_counter >= 2 then
                    -- Calculate the shifted 1D index
                    shifted_index := addr_counter - 2;
                    target_row    := shifted_index / N;
                    target_col    := shifted_index mod N;
                    
                    -- Latch the data into the shifted position
                    matrix_data_sig(target_row, target_col)   <= data_rom_q;
                    matrix_weight_sig(target_row, target_col) <= weight_rom_q;
                end if;
                
                -- Check if loading is complete
                if addr_counter = (N*N+1) then
                    current_state := WAIT_FOR_COMPLETION;
                else
                    addr_counter := addr_counter + 1;
                end if;

            when WAIT_FOR_COMPLETION =>
                wait for ((MAX_ACTIVE_ROWS - 1) + (MAX_ACTIVE_COLS - 1) + MAX_ACTIVE_COLS + 5) * CLK_PER;
                
                -- report "Simulation  finished." severity note;
                -- std.env.finish;
                wait; 
        end case;
    end process;

end sim;