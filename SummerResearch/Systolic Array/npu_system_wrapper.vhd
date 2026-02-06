library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.custom_types.all;

entity npu_system_wrapper is
    port (
        clk           : in  std_logic;
        reset_n       : in  std_logic;

        -- Avalon-MM Slave Interface (Control/Status)
        avs_address   : in  std_logic_vector(3 downto 0);
        avs_write     : in  std_logic;
        avs_writedata : in  std_logic_vector(31 downto 0);
        avs_read      : in  std_logic;
        avs_readdata  : out std_logic_vector(31 downto 0);
        avs_waitrequest : out std_logic := '0';

        -- Avalon-MM Master Interface (Data RAM)
        avm_act_address    : out std_logic_vector(31 downto 0);
        avm_act_read       : out std_logic;
        avm_act_readdata   : in  std_logic_vector(31 downto 0);
        avm_act_waitreq    : in  std_logic;

        -- Avalon-MM Master Interface (Weight RAM)
        avm_weight_address    : out std_logic_vector(31 downto 0);
        avm_weight_read       : out std_logic;
        avm_weight_readdata   : in  std_logic_vector(31 downto 0);
        avm_weight_waitreq    : in  std_logic;

        -- NEW: Avalon-MM Master Interface (Output RAM)
        avm_out_address    : out std_logic_vector(31 downto 0);
        avm_out_write      : out std_logic;
        avm_out_writedata  : out std_logic_vector(31 downto 0);
        avm_out_waitreq    : in  std_logic
    );
end npu_system_wrapper;

architecture rtl of npu_system_wrapper is
    signal reg_ready       : std_logic := '0';
    signal reg_m, reg_n, reg_k : integer := 0;
    signal n_cycle_count   : integer := 0;
    signal n_output        : systolic_array_matrix_output;
    signal n_reset         : std_logic;
    signal n_done          : std_logic; 

    signal n_matrix_data   : systolic_array_matrix_input := (others => (others => (others => '0')));
    signal n_matrix_weight : systolic_array_matrix_input := (others => (others => (others => '0')));

    -- FSM Tracking
    signal fetch_counter   : integer range 0 to 1024 := 0;
    signal write_counter   : integer range 0 to 1024 := 0;
    signal row_idx, col_idx : integer range 0 to 31 := 0;
    signal w_row_idx, w_col_idx : integer range 0 to 31 := 0;

    signal current_output_val : signed(15 downto 0);

    -- Added WRITE_RESULTS state
    type state_type is (IDLE, FETCH_DATA, START_NPU, WAIT_FOR_DONE, WRITE_RESULTS);
    signal state : state_type := IDLE;

begin
    n_reset <= not reset_n;
    avs_waitrequest <= '0';

    process(clk, reset_n)
    begin
        if reset_n = '0' then 
            reg_ready <= '0';
            state <= IDLE;
            avs_readdata <= (others => '0');
            avm_act_read <= '0';
            avm_weight_read <= '0';
            avm_out_write <= '0';
        elsif rising_edge(clk) then
            avs_readdata <= (others => '0');

            -- Register Access
            if avs_write = '1' then 
                case avs_address is
                    when "0000" => reg_ready <= avs_writedata(0);
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

            case state is
                when IDLE =>
                    fetch_counter <= 0;
                    write_counter <= 0;
                    row_idx <= 0; col_idx <= 0;
                    w_row_idx <= 0; w_col_idx <= 0;
                    if reg_ready = '1' then
                        state <= FETCH_DATA;
                    end if;

                when FETCH_DATA =>
                    avm_act_read <= '1'; avm_weight_read <= '1';
                    avm_act_address <= std_logic_vector(to_unsigned(fetch_counter * 4, 32));
                    avm_weight_address <= std_logic_vector(to_unsigned(fetch_counter * 4, 32));

                    if avm_act_waitreq = '0' and avm_weight_waitreq = '0' then
                        n_matrix_data(row_idx, col_idx)   <= avm_act_readdata(15 downto 0);
                        n_matrix_weight(row_idx, col_idx) <= avm_weight_readdata(15 downto 0);
                        if fetch_counter < (reg_m * reg_n) - 1 then
                            fetch_counter <= fetch_counter + 1;
                            if col_idx = (reg_n - 1) then
                                col_idx <= 0; row_idx <= row_idx + 1;
                            else
                                col_idx <= col_idx + 1;
                            end if;
                        else
                            state <= START_NPU;
                        end if;
                    end if;

                when START_NPU =>
                    avm_act_read <= '0'; avm_weight_read <= '0';
                    state <= WAIT_FOR_DONE;

                when WAIT_FOR_DONE =>
                    if n_done = '1' then 
                        state <= WRITE_RESULTS;
                    end if;

                when WRITE_RESULTS =>
                    -- Drive results to the Output RAM
                    avm_out_write <= '1';
                    avm_out_address <= std_logic_vector(to_unsigned(write_counter * 4, 32));


                    current_output_val <= signed(n_output(w_row_idx, w_col_idx));

                    -- Casting the signed matrix output to 32-bit vector for Avalon
                    avm_out_writedata <= std_logic_vector(resize(current_output_val, 32));

                    if avm_out_waitreq = '0' then
                        if write_counter < (reg_m * reg_n) - 1 then
                            write_counter <= write_counter + 1;
                            if w_col_idx = (reg_n - 1) then
                                w_col_idx <= 0; 
                                w_row_idx <= w_row_idx + 1;
                            else
                                w_col_idx <= w_col_idx + 1;
                            end if;
                        else
                            avm_out_write <= '0';
                            reg_ready <= '0'; -- Handshake complete
                            state <= IDLE;
                        end if;
                    end if;
            end case;         
        end if;
    end process;

    NPU_CORE : entity work.top_level_systolic_array
    port map (
        clk => clk, reset => n_reset, ready => reg_ready,
        matrix_data => n_matrix_data, matrix_weight => n_matrix_weight,
        active_rows => reg_m, active_cols => reg_n, active_k => reg_k,
        output => n_output, done => n_done, cycle_count => n_cycle_count 
    );

end architecture rtl;