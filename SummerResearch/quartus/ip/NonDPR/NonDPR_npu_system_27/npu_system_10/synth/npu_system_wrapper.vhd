library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.custom_types.all;
use work.custom_types_int8.all;

entity npu_system_wrapper is
    port (
        clk           : in  std_logic;
        reset_n       : in  std_logic;
        avs_address   : in  std_logic_vector(4 downto 0);
        avs_write     : in  std_logic;
        avs_writedata : in  std_logic_vector(31 downto 0);
        avs_read      : in  std_logic;
        avs_readdata  : out std_logic_vector(31 downto 0);
        avs_waitrequest : out std_logic := '0';
        avm_act_address    : out std_logic_vector(31 downto 0);
        avm_act_read       : out std_logic;
        avm_act_readdata   : in  std_logic_vector(31 downto 0);
        avm_act_waitreq    : in  std_logic;
        avm_weight_address    : out std_logic_vector(31 downto 0);
        avm_weight_read       : out std_logic;
        avm_weight_readdata   : in  std_logic_vector(31 downto 0);
        avm_weight_waitreq    : in  std_logic;
        avm_out_address    : out std_logic_vector(31 downto 0);
        avm_out_write      : out std_logic;
        avm_out_writedata  : out std_logic_vector(31 downto 0);
        avm_out_waitreq    : in  std_logic
    );
end npu_system_wrapper;

architecture rtl of npu_system_wrapper is
    signal reg_ready  : std_logic := '0';
    signal reg_m, reg_n, reg_k : integer := 0;
    signal reg_config : std_logic := '0';
    signal n_cycle_count        : integer := 0;
    signal n_cycle_count_int8   : integer := 0;
    signal cycle_count_lat      : integer := 0;  -- latched at done, holds until next tile
    signal n_reset         : std_logic;

    -- Int16 32x32 signals
    signal n_output        : systolic_array_matrix_output;
    signal n_done          : std_logic;
    signal n_matrix_data   : systolic_array_matrix_input   := (others => (others => (others => '0')));
    signal n_matrix_weight : systolic_array_matrix_input   := (others => (others => (others => '0')));
    signal sa_start_trigger : std_logic := '0';

    -- Int8 16x16 signals
    signal n_output_int8        : systolic_array_matrix_output_int8;
    signal n_done_int8          : std_logic;
    signal n_matrix_data_int8   : systolic_array_matrix_input_int8   := (others => (others => (others => '0')));
    signal n_matrix_weight_int8 : systolic_array_matrix_input_int8   := (others => (others => (others => '0')));
    signal sa_start_trigger_int8 : std_logic := '0';

    signal n_done_mux : std_logic;

    signal fetch_counter   : integer range 0 to 1024 := 0;
    signal write_counter   : integer range 0 to 1024 := 0;
    signal row_idx, col_idx : integer range 0 to 31 := 0;
    signal w_row_idx, w_col_idx : integer range 0 to 31 := 0;

    signal npu_clear : std_logic := '0';

    signal w_row_idx_reg, w_col_idx_reg : integer range 0 to 31 := 0;

    -- max_fanout attributes to reduce logic depth on critical index paths
    attribute max_fanout : integer;
    attribute max_fanout of w_row_idx_reg : signal is 30;
    attribute max_fanout of w_col_idx_reg : signal is 30;

    signal out_data_reg : std_logic_vector(31 downto 0) := (others => '0');
    signal debug_state_reg : std_logic_vector(31 downto 0);

    type state_type is (IDLE,
                        FETCH_DATA_REQ, FETCH_DATA_LATENCY, FETCH_DATA_WAIT,
                        FETCH_WEIGHT_REQ, FETCH_WEIGHT_LATENCY, FETCH_WEIGHT_WAIT,
                        START_NPU, WAIT_FOR_DONE, WRITE_RESULTS, WRITE_AVALON, WRITE_FLUSH);
    signal state : state_type := IDLE;

begin
    n_reset <= (not reset_n) or npu_clear;

    avs_waitrequest <= '0';
    n_done_mux <= n_done_int8 when reg_config = '1' else n_done;

    debug_state_reg <= x"DEB00001" when state = IDLE else
                       x"DEB00002" when state = FETCH_DATA_REQ else
                       x"DEB00003" when state = FETCH_DATA_LATENCY else
                       x"DEB00004" when state = FETCH_DATA_WAIT else
                       x"DEB00005" when state = FETCH_WEIGHT_REQ else
                       x"DEB00006" when state = FETCH_WEIGHT_LATENCY else
                       x"DEB00007" when state = FETCH_WEIGHT_WAIT else
                       x"DEB00008" when state = START_NPU else
                       x"DEB00009" when state = WAIT_FOR_DONE else
                       x"DEB0000A" when state = WRITE_RESULTS else
                       x"DEB0000C" when state = WRITE_AVALON else
                       x"DEB0000B" when state = WRITE_FLUSH else
                       x"DEB0DEAD";

    avs_readdata <= (0 => reg_ready, others => '0')                when avs_address = "00000" else
                    std_logic_vector(to_unsigned(reg_m, 32))       when avs_address = "00100" else
                    std_logic_vector(to_unsigned(reg_n, 32))       when avs_address = "01000" else
                    std_logic_vector(to_unsigned(reg_k, 32))       when avs_address = "01100" else
                    (0 => reg_config, others => '0')               when avs_address = "10000" else
                    (0 => n_done_mux, others => '0')               when avs_address = "10100" else
                    debug_state_reg                                when avs_address = "11000" else
                    std_logic_vector(to_unsigned(cycle_count_lat, 32)) when avs_address = "11100" else
                    (others => '0');

    process(clk, reset_n)
        variable v_temp_out    : signed(31 downto 0);
        variable v_temp_out_i8 : signed(31 downto 0);
    begin
        if reset_n = '0' then
            reg_ready <= '0';
            state <= IDLE;
            avm_act_read <= '0';
            avm_weight_read <= '0';
            avm_out_write <= '0';
            sa_start_trigger <= '0';
            sa_start_trigger_int8 <= '0';
            cycle_count_lat <= 0;
        elsif rising_edge(clk) then

            if avs_write = '1' then
                case avs_address(4 downto 0) is
                    when "00000" => reg_ready  <= avs_writedata(0);
                    when "00100" => reg_m      <= to_integer(unsigned(avs_writedata(15 downto 0)));
                    when "01000" => reg_n      <= to_integer(unsigned(avs_writedata(15 downto 0)));
                    when "01100" => reg_k      <= to_integer(unsigned(avs_writedata(15 downto 0)));
                    when "10000" => reg_config <= avs_writedata(0);
                    when others => null;
                end case;
            end if;

            case state is
                when IDLE =>
                    sa_start_trigger      <= '0';
                    sa_start_trigger_int8 <= '0';
                    fetch_counter <= 0;
                    row_idx <= 0; col_idx <= 0;
                    w_row_idx     <= 0; w_col_idx     <= 0;
                    w_row_idx_reg <= 0; w_col_idx_reg <= 0;
                    npu_clear     <= '0';
                    if reg_ready = '1' then
                        state <= FETCH_DATA_REQ;
                    end if;

                when FETCH_DATA_REQ =>
                    avm_act_read <= '1';
                    avm_act_address <= std_logic_vector(to_unsigned(16#21000# + fetch_counter * 4, 32));
                    if avm_act_waitreq = '0' then
                        state <= FETCH_DATA_LATENCY;
                    end if;

                when FETCH_DATA_LATENCY =>
                    state <= FETCH_DATA_WAIT;

                when FETCH_DATA_WAIT =>
                    if reg_config = '1' then
                        n_matrix_data_int8(row_idx, col_idx) <= avm_act_readdata(7 downto 0);
                    else
                        n_matrix_data(row_idx, col_idx) <= avm_act_readdata(15 downto 0);
                    end if;

                    if fetch_counter < (reg_m * reg_k) - 1 then
                        fetch_counter <= fetch_counter + 1;
                        if col_idx = reg_k - 1 then
                            col_idx <= 0;
                            row_idx <= row_idx + 1;
                        else
                            col_idx <= col_idx + 1;
                        end if;
                        state <= FETCH_DATA_REQ;
                    else
                        fetch_counter <= 0;
                        row_idx <= 0; col_idx <= 0;
                        avm_act_read <= '0';
                        state <= FETCH_WEIGHT_REQ;
                    end if;

                when FETCH_WEIGHT_REQ =>
                    avm_weight_read <= '1';
                    avm_weight_address <= std_logic_vector(to_unsigned(16#22000# + fetch_counter * 4, 32));
                    if avm_weight_waitreq = '0' then
                        state <= FETCH_WEIGHT_LATENCY;
                    end if;

                when FETCH_WEIGHT_LATENCY =>
                    state <= FETCH_WEIGHT_WAIT;

                when FETCH_WEIGHT_WAIT =>
                    if reg_config = '1' then
                        n_matrix_weight_int8(row_idx, col_idx) <= avm_weight_readdata(7 downto 0);
                    else
                        n_matrix_weight(row_idx, col_idx) <= avm_weight_readdata(15 downto 0);
                    end if;

                    if fetch_counter < (reg_k * reg_n) - 1 then
                        fetch_counter <= fetch_counter + 1;
                        if col_idx = reg_n - 1 then
                            col_idx <= 0;
                            row_idx <= row_idx + 1;
                        else
                            col_idx <= col_idx + 1;
                        end if;
                        state <= FETCH_WEIGHT_REQ;
                    else
                        avm_weight_read <= '0';
                        state <= START_NPU;
                    end if;

                when START_NPU =>
                    if reg_config = '1' then
                        sa_start_trigger_int8 <= '1';
                    else
                        sa_start_trigger <= '1';
                    end if;
                    state <= WAIT_FOR_DONE;

                when WAIT_FOR_DONE =>
                    sa_start_trigger      <= '0';
                    sa_start_trigger_int8 <= '0';
                    if n_done_mux = '1' then
                        if reg_config = '1' then
                            cycle_count_lat <= n_cycle_count_int8;
                        else
                            cycle_count_lat <= n_cycle_count;
                        end if;
                        write_counter <= 0;
                        w_row_idx     <= 0; w_col_idx     <= 0;
                        w_row_idx_reg <= 0; w_col_idx_reg <= 0;
                        state <= WRITE_RESULTS;
                    end if;

                when WRITE_RESULTS =>
                    if reg_config = '1' then
                        v_temp_out_i8 := signed(n_output_int8(w_row_idx_reg, w_col_idx_reg));
                        out_data_reg  <= std_logic_vector(v_temp_out_i8);
                    else
                        v_temp_out   := signed(n_output(w_row_idx_reg, w_col_idx_reg));
                        out_data_reg <= std_logic_vector(resize(v_temp_out, 32));
                    end if;
                    state <= WRITE_AVALON;

                when WRITE_AVALON =>
                    avm_out_write     <= '1';
                    avm_out_address   <= std_logic_vector(to_unsigned(16#23000# + write_counter * 4, 32));
                    avm_out_writedata <= out_data_reg;
                    if avm_out_waitreq = '0' then
                        if write_counter < (reg_m * reg_n) - 1 then
                            write_counter <= write_counter + 1;
                            if w_col_idx = reg_n - 1 then
                                w_col_idx     <= 0;
                                w_row_idx     <= w_row_idx + 1;
                                w_col_idx_reg <= 0;
                                w_row_idx_reg <= w_row_idx + 1;
                            else
                                w_col_idx     <= w_col_idx + 1;
                                w_col_idx_reg <= w_col_idx + 1;
                                w_row_idx_reg <= w_row_idx;
                            end if;
                            state <= WRITE_RESULTS;
                        else
                            state <= WRITE_FLUSH;
                        end if;
                    end if;

                when WRITE_FLUSH =>
                    if avm_out_waitreq = '0' then
                        avm_out_write <= '0';
                        reg_ready     <= '0';
                        npu_clear     <= '1';
                        state         <= IDLE;
                    end if;
            end case;
        end if;
    end process;

    NPU_CORE_INT16 : entity work.top_level_systolic_array
    port map (
        clk           => clk,
        reset         => n_reset,
        ready         => sa_start_trigger,
        matrix_data   => n_matrix_data,
        matrix_weight => n_matrix_weight,
        active_rows   => reg_m,
        active_cols   => reg_n,
        active_k      => reg_k,
        output        => n_output,
        done          => n_done,
        cycle_count   => n_cycle_count
    );

    NPU_CORE_INT8 : entity work.top_level_systolic_array_int8
    port map (
        clk           => clk,
        reset         => n_reset,
        ready         => sa_start_trigger_int8,
        matrix_data   => n_matrix_data_int8,
        matrix_weight => n_matrix_weight_int8,
        active_rows   => reg_m,
        active_cols   => reg_n,
        active_k      => reg_k,
        output        => n_output_int8,
        done          => n_done_int8,
        cycle_count   => n_cycle_count_int8
    );

end architecture rtl;
