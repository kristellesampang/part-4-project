-- File: de1_soc_top.vhd
-- External communication test for the UART component, connected to physical pins.

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.custom_types.all;

entity de1_soc_top is
    port (
        -- Physical pins (must match your .qsf file)
        CLOCK_50      : in  std_logic;
        KEY           : in  std_logic_vector(1 downto 0);
        LEDR          : out std_logic_vector(9 downto 0);
        FPGA_UART_RX  : in  std_logic;   -- Pin AC18
        FPGA_UART_TX  : out std_logic    -- Pin Y17
    );
end entity de1_soc_top;

architecture Behavioral of de1_soc_top is

    -- Component for the generic UART module
    -- component UART is
    --     generic(
    --         BAUD_RATE      : integer := 9600;
    --         CLOCK_FREQUENCY : integer := 50_000_000
    --     );
    --     port(
    --         i_Clock         : in  std_logic;
    --         i_RX_Serial     : in  std_logic;
    --         o_RX_DV         : out std_logic;
    --         o_RX_Byte       : out std_logic_vector(7 downto 0);
    --         i_TX_DV         : in  std_logic;
    --         i_TX_Byte       : in  std_logic_vector(7 downto 0);
    --         o_TX_Active     : out std_logic;
    --         o_TX_Serial     : out std_logic
    --     );
    -- end component;
    component UART is
        port( CLOCK_50,RST : in std_logic;
		-- SW			: in STD_LOGIC_VECTOR(3 downto 0);
        data_tx : in STD_LOGIC_VECTOR(7 downto 0);
		send		: in STD_LOGIC;
		LED		: out STD_LOGIC_VECTOR(7 downto 0);
		UART_TXD : out std_logic;
		UART_RXD : in std_logic);
    end component;

    -- Component for the debouncer (Assuming it exists and works)
    component debouncer is
        port (
            clk         : in  std_logic;
            reset       : in  std_logic;
            button_in   : in  std_logic;
            tick_out    : out std_logic
        );
    end component;

    -- NPU wrapper component
    component npu_wrapper is
        port (
            clk     : in  bit_1;
            reset   : in  bit_1;
            start   : in  bit_1;
            done    : out bit_1;
            active_m    : out integer;
            active_n    : out integer;
            npu_cycle_count : out std_logic_vector(7 downto 0);
            ready_to_read : out bit_1;
            read_address : in bit_7;
            read_data    : out bit_32
        );
    end component;

    -- Internal signals
    signal data_reg         : std_logic_vector(31 downto 0) := (others => '0');
    signal data_byte       : std_logic_vector(7 downto 0) := (others => '0');
    -- reset, SW is driven by KEY[0] and KEY[1] respectively
    -- send is drive by key[0]
    --  uart_txd amd uart_rxd are connected directly to FPGA_UART_TX and FPGA_UART_RX
    signal s_reset         : std_logic := '1';  -- Active low reset
    signal s_start_pulse   : std_logic := '1';


    -- NPU integration signals
    signal npu_start        : std_logic := '0';
    signal npu_done         : std_logic;
    signal npu_read_address : std_logic_vector(6 downto 0) := (others => '0');
    signal npu_read_data    : std_logic_vector(31 downto 0);
    signal npu_ready_to_read : std_logic := '1';  -- Always ready to read for simplicity
    signal npu_active_m    : integer;
    signal npu_active_n    : integer;
    signal sa_cycle_count  : std_logic_vector(7 downto 0);
    signal send_flag : std_logic := '0';

    -- TX controls
    signal tx_byte          : std_logic_vector(7 downto 0) := (others => '0');
    signal tx_dv            : std_logic := '0';

    -- Dump FSM
    type dump_state_t is (D_IDLE, D_START, D_WAIT_DONE, D_SET_ADDR, D_WAIT_DATA, D_SEND_B0, D_SEND_B1, D_SEND_B2, D_SEND_B3, D_NEXT);
    signal dstate           : dump_state_t := D_IDLE;
    signal addr_ctr         : integer range 0 to N*N := 0;
    signal word_reg         : std_logic_vector(31 downto 0) := (others => '0');
    signal byte_idx         : integer range 0 to 3 := 0;
    
begin

    -- Use KEY[0] as the active-low reset button
    s_reset <= not KEY(0);
	-- s_start_pulse <= not KEY(1);
    


    -- UART starts transmission when the NPU computation starts (so at the same time)
    -- It will continue transmission until the npu is done
    -- When the npu is done uart transmission stops
    -- this will prevent the spamming of the UART port on the receiving end
    
    -- Instantiate the UART component
    UART_inst : UART
        port map (
            CLOCK_50 => CLOCK_50,
            RST      => s_reset,          -- Active low reset
            data_tx => data_byte,
            send     => send_flag,       -- Use KEY[0] as send button (active low)
            LED      => LEDR(7 downto 0), -- Display received byte on lower 8 LEDs
            UART_TXD => FPGA_UART_TX,     -- Connect to physical TX pin
            UART_RXD => FPGA_UART_RX      -- Connect to physical RX pin
        );

    -- Instantiate a debouncer for the "send" button (KEY[1])
    debounce_inst : debouncer
        port map (
            clk         => CLOCK_50,
            reset       => s_reset,
            button_in   => not KEY(1),
            tick_out    => s_start_pulse
        );

    -- Instantiate the NPU wrapper
    npu_i : npu_wrapper
        port map (
            clk          => CLOCK_50,
            reset        => s_reset,
            start        => s_start_pulse,
            done         => npu_done,
            active_m     => npu_active_m,
            active_n     => npu_active_n,
            npu_cycle_count => sa_cycle_count,
            ready_to_read => npu_ready_to_read,
            read_address => npu_read_address,
            read_data    => npu_read_data
        );

    process(CLOCK_50)
    begin
        if rising_edge(CLOCK_50) then
            if s_reset = '1' then
                data_reg <= (others => '0');
                data_reg <= "11111111111111111111111111111111"; -- "11111111 11111111 11111111 11111111";
            else
                if npu_done = '1' then
                   data_byte <= sa_cycle_count; -- Transmit the cycle count when starting
                     send_flag <= '1';
                else
                     send_flag <= '0';
                   end if; 
            end if;
        end if;
		  
    end process;

    

    --- Process to start the NPU by clicking KEY1, wait for the NPU to be done, read the 2D array, then transmit this through uart
    -- process(CLOCK_50)
    -- begin
    --     if rising_edge(CLOCK_50) then
    --         if s_reset = '1' then
    --             dstate       <= D_IDLE;
    --             addr_ctr     <= 0;
    --             word_reg     <= (others => '0');
    --             byte_idx     <= 0;
    --             tx_dv        <= '0';
    --             tx_byte      <= (others => '0');
    --             npu_read_address <= (others => '0');
    --         else
    --             word_reg <= std_logic_vector(to_unsigned(12345, 32));
    --             case dstate is
    --                 when D_IDLE =>
    --                     if s_start_pulse = '1' and npu_ready_to_read = '1' then
    --                         dstate <= D_START;
    --                     end if;

    --                 when D_START =>
    --                     npu_read_address <= (others => '0');
    --                     addr_ctr <= 0;
    --                     dstate <= D_WAIT_DONE;

    --                 when D_WAIT_DONE =>
    --                     if npu_done = '1' then
    --                         dstate <= D_SET_ADDR;
    --                     end if;

    --                 when D_SET_ADDR =>
    --                     npu_read_address <= std_logic_vector(to_unsigned(addr_ctr, 7));
    --                     dstate <= D_WAIT_DATA;

    --                 when D_WAIT_DATA =>
    --                     -- word_reg <= npu_read_data;
    --                     -- Placeholder until npu_read_data is connected
    --                     byte_idx <= 0;
    --                     dstate <= D_SEND_B0;

    --                 when D_SEND_B0 =>
    --                     tx_byte <= word_reg(7 downto 0);
    --                     tx_dv   <= '1';
    --                     dstate  <= D_SEND_B1;

    --                 when D_SEND_B1 =>
    --                     tx_dv   <= '0';
    --                     if s_TX_Active = '0' then
    --                         tx_byte <= word_reg(15 downto 8);
    --                         tx_dv   <= '1';
    --                         dstate  <= D_SEND_B2;
    --                     end if;

    --                 when D_SEND_B2 =>
    --                     tx_dv   <= '0';
    --                     if s_TX_Active = '0' then
    --                         tx_byte <= word_reg(23 downto 16);
    --                         tx_dv   <= '1';
    --                         dstate  <= D_SEND_B3;
    --                     end if;

    --                 when D_SEND_B3 =>
    --                     tx_dv   <= '0';
    --                     if s_TX_Active = '0' then
    --                         tx_byte <= word_reg(31 downto 24);
    --                         tx_dv   <= '1';
    --                         dstate  <= D_NEXT;
    --                     end if;

    --                 when D_NEXT =>
    --                     tx_dv   <= '0';
    --                     if s_TX_Active = '0' then
    --                         if addr_ctr < (npu_active_m * npu_active_n - 1) then 
    --                             addr_ctr <= addr_ctr + 1;
    --                             dstate   <= D_SET_ADDR;
    --                         else
    --                             dstate   <= D_IDLE;  -- Finished sending all data
    --                         end if;
    --                     end if;
    --             end case;
    --         end if;
    --     end if;
    -- end process;
    
    -- When a byte is received, display it on the LEDs
--    process(CLOCK_50)
--    begin
--        if rising_edge(CLOCK_50) then
--            if s_RX_DV = '1' then
--                LEDR(7 downto 0) <= s_RX_Byte;
--            end if;
--        end if;
--    end process;
--
--    -- Show status on the other LEDs
--    LEDR(9) <= FPGA_UART_TX;  -- Show when transmitting
--    LEDR(8) <= s_RX_DV;      -- Show when data is received

    -- NOTE: The connections to FPGA_UART_RX and FPGA_UART_TX are now handled
    -- directly in the UART_inst port map above.

end Behavioral;