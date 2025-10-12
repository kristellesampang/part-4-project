-- File: de1_soc_top.vhd
-- External communication test for the UART component, connected to physical pins.

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

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
    component UART is
        generic(
            BAUD_RATE      : integer := 9600;
            CLOCK_FREQUENCY : integer := 50_000_000
        );
        port(
            i_Clock         : in  std_logic;
            i_RX_Serial     : in  std_logic;
            o_RX_DV         : out std_logic;
            o_RX_Byte       : out std_logic_vector(7 downto 0);
            i_TX_DV         : in  std_logic;
            i_TX_Byte       : in  std_logic_vector(7 downto 0);
            o_TX_Active     : out std_logic;
            o_TX_Serial     : out std_logic
        );
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

    -- Internal signals
    signal s_reset       : std_logic;
    signal s_start_pulse : std_logic;
    signal s_TX_DV       : std_logic := '0';
    signal s_RX_DV       : std_logic;
    signal s_RX_Byte     : std_logic_vector(7 downto 0);
    signal s_TX_Active   : std_logic;
    
begin

    -- Use KEY[0] as the active-low reset button
    s_reset <= not KEY(0);

    -- Instantiate the UART component
    UART_inst : UART
        generic map (
             BAUD_RATE      => 9600,            -- Match this to your Pico settings
             CLOCK_FREQUENCY => 50_000_000       -- Match this to CLOCK_50
        )
        port map (
            i_Clock      => CLOCK_50,
            -- Connect RX to the physical pin (FPGA_UART_RX is the input)
            i_RX_Serial  => FPGA_UART_RX, 
            o_RX_DV      => s_RX_DV,
            o_RX_Byte    => s_RX_Byte,
            i_TX_DV      => s_TX_DV,
            i_TX_Byte    => x"AA",              -- Send a test pattern (10101010)
            o_TX_Active  => s_TX_Active,
            -- Connect TX to the physical pin (FPGA_UART_TX is the output)
            o_TX_Serial  => FPGA_UART_TX
        );

    -- Instantiate a debouncer for the "send" button (KEY[1])
    debounce_inst : debouncer
        port map (
            clk         => CLOCK_50,
            reset       => s_reset,
            button_in   => not KEY(1),
            tick_out    => s_start_pulse
        );

    -- Generate a single-cycle pulse to start transmission
    process(CLOCK_50)
    begin
        if rising_edge(CLOCK_50) then
            s_TX_DV <= s_start_pulse;
        end if;
    end process;
    
    -- When a byte is received, display it on the LEDs
    process(CLOCK_50)
    begin
        if rising_edge(CLOCK_50) then
            if s_RX_DV = '1' then
                LEDR(7 downto 0) <= s_RX_Byte;
            end if;
        end if;
    end process;

    -- Show status on the other LEDs
    LEDR(9) <= s_TX_Active;  -- Show when transmitting
    LEDR(8) <= s_RX_DV;      -- Show when data is received

    -- NOTE: The connections to FPGA_UART_RX and FPGA_UART_TX are now handled
    -- directly in the UART_inst port map above.

end Behavioral;