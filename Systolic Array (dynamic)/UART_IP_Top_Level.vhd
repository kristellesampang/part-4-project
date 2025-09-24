library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity UART_IP_Top_Level is
    port (
        -- DE1-SoC Board Clock (50 MHz)
        CLOCK_50 : in std_logic;

        -- DE1-SoC Board Reset (Connect to a push button, active-low)
        -- NOTE: Most push buttons on the DE1-SoC are active-low.
        KEY : in std_logic_vector(0 downto 0);

        -- DE1-SoC Board UART Pins
        UART_RXD : in std_logic;
        UART_TXD : out std_logic;

        -- DE1-SoC Board LEDs for output
        LEDR : out std_logic_vector(9 downto 0)
    );
end entity UART_IP_Top_Level;

architecture rtl of UART_IP_Top_Level is

    -- Component declaration for IP 
    component UART_Rx is
        port (
            clk_clk            : in  std_logic := '0';
            reset_reset_n      : in  std_logic := '0';
            rs232_0_address    : in  std_logic := '0';
            rs232_0_chipselect : in  std_logic := '0';
            rs232_0_read       : in  std_logic := '0';
            rs232_0_readdata   : out std_logic_vector(31 downto 0);
            rs232_0_UART_RXD   : in  std_logic := '0';
            rs232_0_UART_TXD   : out std_logic;
            -- Unused ports for this simple test can be tied off.
            rs232_0_byteenable : in  std_logic_vector(3 downto 0) := (others => '0');
            rs232_0_write      : in  std_logic := '0';
            rs232_0_writedata  : in  std_logic_vector(31 downto 0) := (others => '0');
            rs232_0_irq        : out std_logic
        );
    end component UART_Rx;

    -- Define the states for our Avalon bus controller FSM
    type t_state is (
        s_idle,                 -- Wait state
        s_check_status,         -- Drive Avalon signals to read the Status Register
        s_wait_for_status,      -- Wait one cycle for the read to complete
        s_read_data,            -- Drive Avalon signals to read the Data Register
        s_wait_for_data         -- Wait one cycle for the read to complete
    );
    signal r_state : t_state := s_idle;

    -- Signals to drive the Avalon bus interface of the UART IP
    signal w_avalon_address    : std_logic;
    signal w_avalon_read       : std_logic;
    signal w_avalon_readdata   : std_logic_vector(31 downto 0);
    signal r_led_output        : std_logic_vector(7 downto 0) := (others => '0');

    -- Constants for UART register addresses
    constant C_DATA_REG_ADDR   : std_logic := '0';
    constant C_STATUS_REG_ADDR : std_logic := '1';

begin

    -- Instantiate the UART IP core that you generated.
    uart_ip_inst : component UART_Rx
        port map (
            clk_clk            => CLOCK_50,
            reset_reset_n      => KEY(0), -- Connect to the active-low push button
            rs232_0_address    => w_avalon_address,
            rs232_0_chipselect => '1', -- Chipselect is always active for this simple design
            rs232_0_read       => w_avalon_read,
            rs232_0_readdata   => w_avalon_readdata,
            rs232_0_UART_RXD   => UART_RXD,
            rs232_0_UART_TXD   => UART_TXD,
            -- Tie off unused ports
            rs232_0_irq        => open
        );

    -- State machine to control the Avalon bus and read from the UART
    process(CLOCK_50, KEY(0))
    begin
        if KEY(0) = '0' then -- Active-low reset
            r_state      <= s_idle;
            w_avalon_read    <= '0';
            w_avalon_address <= '0';
            r_led_output <= (others => '0');
        elsif rising_edge(CLOCK_50) then
            case r_state is
                -- IDLE: Default state, prepare to check for new data.
                when s_idle =>
                    r_state <= s_check_status;

                -- CHECK STATUS: Start a read transaction on the status register.
                when s_check_status =>
                    w_avalon_address <= C_STATUS_REG_ADDR; -- Address = 1 for status
                    w_avalon_read    <= '1';
                    r_state          <= s_wait_for_status;

                -- WAIT FOR STATUS: Wait one clock cycle for the IP to provide the status data.
                when s_wait_for_status =>
                    w_avalon_read <= '0'; -- De-assert read
                    -- The status register's bit 15 is the "RRDY(Read Ready) flag.
                    -- If it's '1', it means a new byte has arrived.
                    if w_avalon_readdata(15) = '1' then
                        r_state <= s_read_data; -- A byte is ready, go read it.
                    else
                        r_state <= s_check_status; -- No byte yet, check again.
                    end if;

                -- READ DATA: A byte is ready. Start a read transaction on the data register.
                when s_read_data =>
                    w_avalon_address <= C_DATA_REG_ADDR; -- Address = 0 for data
                    w_avalon_read    <= '1';
                    r_state          <= s_wait_for_data;

                -- WAIT FOR DATA: Wait one clock cycle for the IP to provide the data byte.
                when s_wait_for_data =>
                    w_avalon_read <= '0'; -- De-assert read
                    -- The received byte is in the lower 8 bits of the readdata bus.
                    -- Latch this value to display on the LEDs.
                    r_led_output <= w_avalon_readdata(7 downto 0);
                    r_state      <= s_idle; -- Go back to idle to wait for the next byte.

            end case;
        end if;
    end process;

    -- Connect the latched data to the physical LED pins.
    LEDR(7 downto 0) <= r_led_output;
    LEDR(9 downto 8) <= (others => '0'); 

end architecture rtl;
