library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.custom_types.all;

entity npu_system_wrapper is
    port (
        clk           : in  bit_1;
        reset_n       : in  bit_1;

        -- Avalon-MM Slave Interface
        avs_address   : in  std_logic_vector(3 downto 0);
        avs_write     : in  bit_1;
        avs_writedata : in  std_logic_vector(31 downto 0);
        avs_read      : in  bit_1;
        avs_readdata  : out std_logic_vector(31 downto 0);

        -- Avalon-MM Master Interface for Data/Activation
        avm_act_address    : out std_logic_vector(31 downto 0);
        avm_act_read       : out bit_1;
        avm_act_readdata   : in  std_logic_vector(31 downto 0);
        avm_act_waitreq    : in  bit_1;

        -- Avalon-MM Master Interface for Weights
        avm_weight_address    : out std_logic_vector(31 downto 0);
        avm_weight_read       : out bit_1;
        avm_weight_readdata   : in  std_logic_vector(31 downto 0);
        avm_weight_waitreq    : in  bit_1
    );
end npu_system_wrapper;

architecture rtl of npu_system_wrapper is
    signal reg_ready       : bit_1 := '0';
    signal reg_m, reg_n, reg_k : integer := 0;
    signal n_cycle_count   : integer := 0;
    signal n_output        : systolic_array_matrix_output;
    signal n_reset      : bit_1;

    signal n_matrix_data   : systolic_array_matrix_input := (others => (others => (others => '0')));
    signal n_matrix_weight : systolic_array_matrix_input := (others => (others => (others => '0')));

    -- internal signals to track fetching progress
    signal fetch_counter : integer range 0 to 1024 := 0;
    signal row_idx, col_idx : integer range 0 to 31 := 0;

    type state_type is (IDLE, FETCH_DATA, START_NPU);
    signal state : state_type := IDLE;

begin
    n_reset <= not reset_n;

    process(clk, reset_n)
    begin
        if reset_n = '0' then 
            reg_ready <= '0';
            avs_readdata <= (others => '0');
        elsif rising_edge(clk) then

            avs_readdata <= (others => '0');

            if avs_write = '1' then 
                case avs_address is
                    when "0000" => -- Control Register
                        reg_ready <= avs_writedata(0);
                    when "0001" => -- M Register
                        reg_m <= to_integer(unsigned(avs_writedata(15 downto 0)));
                    when "0010" => -- N Register
                        reg_n <= to_integer(unsigned(avs_writedata(15 downto 0)));
                    when "0011" => -- K Register
                        reg_k <= to_integer(unsigned(avs_writedata(15 downto 0)));
                    when others => null;
                end case;
            end if;

            if avs_read = '1' then
                case avs_address is
                    when x"4" => avs_readdata <= std_logic_vector(to_signed(n_cycle_count, 32));
                    when others => avs_readdata <= (others => '0');
                end case;
            end if;
            -- Inside  main process, update the state machine:
            case state is
                when IDLE =>
                    fetch_counter <= 0;
                    row_idx <= 0;
                    col_idx <= 0;
                    if reg_ready = '1' then
                        state <= FETCH_DATA;
                    end if;

                when FETCH_DATA =>
                    -- 1. Drive the Master Addresses to the RAMs
                    avm_act_read    <= '1';
                    avm_weight_read <= '1';
                    avm_act_address <= std_logic_vector(to_unsigned(fetch_counter * 4, 32)); -- Byte addressing
                    avm_weight_address <= std_logic_vector(to_unsigned(fetch_counter * 4, 32));

                    -- 2. Wait for RAM to be ready (Waitrequest = '0')
                    if avm_act_waitreq = '0' and avm_weight_waitreq = '0' then
                        -- Map the 32-bit RAM word to your 16-bit Matrix (taking lower 16 bits)
                        n_matrix_data(row_idx, col_idx)   <= avm_act_readdata(15 downto 0);
                        n_matrix_weight(row_idx, col_idx) <= avm_weight_readdata(15 downto 0);

                        -- 3. Increment indices
                        if fetch_counter < 1023 then
                            fetch_counter <= fetch_counter + 1;
                            -- Update row/col mapping for a 32x32 array
                            if col_idx = 31 then
                                col_idx <= 0;
                                row_idx <= row_idx + 1;
                            else
                                col_idx <= col_idx + 1;
                            end if;
                        else
                            state <= START_NPU; -- All 1024 elements fetched
                        end if;
                    end if;
                when START_NPU =>
                    -- Pulse the internal NPU core and go back to IDLE
                    avm_act_read    <= '0';
                    avm_weight_read <= '0';
                    reg_ready       <= '1'; -- This triggers n_reset/ready in the NPU_CORE
                    state           <= IDLE;
            end case;         
        end if;
    end process;

    NPU_CORE : entity work.top_level_systolic_array
    port map (
        clk           => clk,
        reset         => n_reset,
        ready         => reg_ready,
        matrix_data   => n_matrix_data,
        matrix_weight => n_matrix_weight,
        active_rows   => reg_m,
        active_cols   => reg_n,
        active_k      => reg_k,
        output        => n_output, -- Not connected
        cycle_count   => n_cycle_count  -- Not connected
    );
end rtl;