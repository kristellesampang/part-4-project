library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.custom_types.all;

entity npu_system_wrapper is
    port (
        clk           : in  std_logic;
        reset_n       : in  std_logic; -- Active Low from Platform Designer

        -- Avalon-MM Slave Interface (Laptop/JTAG Master)
        avs_address   : in  std_logic_vector(3 downto 0);
        avs_write     : in  std_logic;
        avs_writedata : in  std_logic_vector(31 downto 0);
        avs_read      : in  std_logic;
        avs_readdata  : out std_logic_vector(31 downto 0);
        avs_waitrequest : out std_logic := '0';

        -- Avalon-MM Master Interface for Data RAM
        avm_act_address    : out std_logic_vector(31 downto 0);
        avm_act_read       : out std_logic;
        avm_act_readdata   : in  std_logic_vector(31 downto 0);
        avm_act_waitreq    : in  std_logic;

        -- Avalon-MM Master Interface for Weight RAM
        avm_weight_address    : out std_logic_vector(31 downto 0);
        avm_weight_read       : out std_logic;
        avm_weight_readdata   : in  std_logic_vector(31 downto 0);
        avm_weight_waitreq    : in  std_logic
    );
end npu_system_wrapper;

architecture rtl of npu_system_wrapper is
    -- Internal Registers (Mapped to Avalon Slave)
    signal reg_ready       : std_logic := '0';
    signal reg_m           : integer := 0;
    signal reg_n           : integer := 0;
    signal reg_k           : integer := 0;

    -- Internal Signals for NPU Core
    signal n_cycle_count   : integer := 0;
    signal n_output        : systolic_array_matrix_output;
    signal n_reset         : std_logic;
    signal n_done          : std_logic; 

    signal n_matrix_data   : systolic_array_matrix_input := (others => (others => (others => '0')));
    signal n_matrix_weight : systolic_array_matrix_input := (others => (others => (others => '0')));

    -- FSM Signals
    signal fetch_counter   : integer range 0 to 1024 := 0;
    signal row_idx, col_idx : integer range 0 to 31 := 0;

    type state_type is (IDLE, FETCH_DATA, START_NPU, WAIT_FOR_DONE);
    signal state : state_type := IDLE;

begin
    -- Reset logic: Active Low to Active High conversion for the NPU core
    n_reset <= not reset_n;
    avs_waitrequest <= '0'; -- Always ready for JTAG master

    process(clk, reset_n)
    begin
        if reset_n = '0' then 
            reg_ready <= '0';
            state <= IDLE;
            avs_readdata <= (others => '0');
            avm_act_read <= '0';
            avm_weight_read <= '0';
            fetch_counter <= 0;
        elsif rising_edge(clk) then

            -- ---------------------------------------------------------
            -- 1. AVALON SLAVE INTERFACE (JTAG Master Communication)
            -- ---------------------------------------------------------
            avs_readdata <= (others => '0');

            if avs_write = '1' then 
                case avs_address is
                    when "0000" => reg_ready <= avs_writedata(0); -- Start Trigger
                    when "0001" => reg_m <= to_integer(unsigned(avs_writedata(15 downto 0)));
                    when "0010" => reg_n <= to_integer(unsigned(avs_writedata(15 downto 0)));
                    when "0011" => reg_k <= to_integer(unsigned(avs_writedata(15 downto 0)));
                    when others => null;
                end case;
            end if;

            if avs_read = '1' then
                case avs_address is
                    when "0000" => avs_readdata <= (31 downto 1 => '0') & reg_ready;
                    when "0001" => avs_readdata <= std_logic_vector(to_unsigned(reg_m, 32));
                    when "0010" => avs_readdata <= std_logic_vector(to_unsigned(reg_n, 32));
                    when "0011" => avs_readdata <= std_logic_vector(to_unsigned(reg_k, 32));
                    when "0100" => avs_readdata <= std_logic_vector(to_signed(n_cycle_count, 32));
                    when others => avs_readdata <= (others => '0');
                end case;
            end if;

            -- ---------------------------------------------------------
            -- 2. NPU STATE MACHINE (Control Plane)
            -- ---------------------------------------------------------
            case state is
                when IDLE =>
                    fetch_counter <= 0;
                    row_idx <= 0;
                    col_idx <= 0;
                    avm_act_read <= '0';
                    avm_weight_read <= '0';
                    -- Wait for JTAG Master to write '1' to address 0x0
                    if reg_ready = '1' then
                        state <= FETCH_DATA;
                    end if;

                when FETCH_DATA =>
                    -- Drive the Master addresses
                    avm_act_read    <= '1';
                    avm_weight_read <= '1';
                    avm_act_address <= std_logic_vector(to_unsigned(fetch_counter * 4, 32));
                    avm_weight_address <= std_logic_vector(to_unsigned(fetch_counter * 4, 32));

                    -- Wait for both memories to acknowledge the read
                    if avm_act_waitreq = '0' and avm_weight_waitreq = '0' then
                        -- Load internal systolic input buffers
                        n_matrix_data(row_idx, col_idx)   <= avm_act_readdata(15 downto 0);
                        n_matrix_weight(row_idx, col_idx) <= avm_weight_readdata(15 downto 0);

                        -- DYNAMIC FETCH: Check if we have fetched M*N elements
                        if fetch_counter < (reg_m * reg_n) - 1 then
                            fetch_counter <= fetch_counter + 1;
                            -- Update 2D array mapping
                            if col_idx = (reg_n - 1) then
                                col_idx <= 0;
                                row_idx <= row_idx + 1;
                            else
                                col_idx <= col_idx + 1;
                            end if;
                        else
                            state <= START_NPU;
                        end if;
                    end if;

                when START_NPU =>
                    avm_act_read    <= '0';
                    avm_weight_read <= '0';
                    -- Move to wait state. n_done will come from systolic array control unit.
                    state <= WAIT_FOR_DONE;

                when WAIT_FOR_DONE =>
                    -- HANDSHAKE: When the core is finished, we clear reg_ready
                    -- and return to IDLE. This stops the loop cycle.
                    if n_done = '1' then 
                        reg_ready <= '0'; -- Software can now see we are "Not Busy"
                        state <= IDLE;
                    end if;
            end case;         
        end if;
    end process;

    -- ---------------------------------------------------------
    -- 3. NPU CORE INSTANTIATION
    -- ---------------------------------------------------------
    NPU_CORE : entity work.top_level_systolic_array
    port map (
        clk           => clk,
        reset         => n_reset,
        ready         => reg_ready,  -- Drives 'run_enable' in your control unit
        matrix_data   => n_matrix_data,
        matrix_weight => n_matrix_weight,
        active_rows   => reg_m,
        active_cols   => reg_n,
        active_k      => reg_k,
        output        => n_output,
        done          => n_done,     -- This must be 'completed' from your CU
        cycle_count   => n_cycle_count 
    );

end architecture rtl;