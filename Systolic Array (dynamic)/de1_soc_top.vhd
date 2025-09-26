library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity de1_soc_top is
    port (

        -- CLOCK
        CLOCK_50      : in  std_logic;
        
        -- KEYS (Push Buttons) - Using 2 for this example
        KEY           : in  std_logic_vector(1 downto 0);
        
        -- LEDS - Using 8 for this example
        LEDR          : out std_logic_vector(7 downto 0);
        
        -- SDRAM Interface (all pins must be assigned)
        DRAM_ADDR     : out std_logic_vector(12 downto 0);
        DRAM_BA       : out std_logic_vector(1 downto 0);
        DRAM_CAS_N    : out std_logic;
        DRAM_CKE      : out std_logic;
        DRAM_CLK      : out std_logic;
        DRAM_CS_N     : out std_logic;
        DRAM_DQ       : inout std_logic_vector(15 downto 0);
        DRAM_DQM      : out std_logic_vector(1 downto 0);
        DRAM_RAS_N    : out std_logic;
        DRAM_WE_N     : out std_logic
    );
end entity de1_soc_top;

architecture rtl of de1_soc_top is

    -- Component declaration for the system you built in Platform Designer.
    -- This MUST match the entity name in the file Platform Designer generated.
    component Nios_System_2A is
        port (
            button_pio_external_connection_export : in    std_logic_vector(1 downto 0);
            clocks_ref_clk_clk                    : in    std_logic;
            clocks_ref_reset_reset                : in    std_logic;
            clocks_sdram_clk_clk                  : out   std_logic;
            high_res_timer_irq_irq                : out   std_logic;
            jtag_uart_irq_irq                     : out   std_logic;
            led_pio_external_connection_export    : out   std_logic_vector(7 downto 0);
            sdram_wire_addr                       : out   std_logic_vector(12 downto 0);
            sdram_wire_ba                         : out   std_logic_vector(1 downto 0);
            sdram_wire_cas_n                      : out   std_logic;
            sdram_wire_cke                        : out   std_logic;
            sdram_wire_cs_n                       : out   std_logic;
            sdram_wire_dq                         : inout std_logic_vector(15 downto 0);
            sdram_wire_dqm                        : out   std_logic_vector(1 downto 0);
            sdram_wire_ras_n                      : out   std_logic;
            sdram_wire_we_n                       : out   std_logic
        );
    end component Nios_System_2A;

    -- Signal for the active-low reset. DE1-SoC buttons are active-low.
    signal w_reset : std_logic;

begin

    -- The push buttons (KEY) on the DE1-SoC are active-low. This means they
    -- output a '0' when pressed. The reset input of the NIOS system is
    -- active-high, so we invert the signal from KEY(0).
    w_reset <= not KEY(0);

    -- Instantiate your complete Platform Designer system ("the motherboard").
    the_nios_system : component Nios_System_2A
        port map (
            -- Clocks and Reset
            clocks_ref_clk_clk                    => CLOCK_50,
            clocks_ref_reset_reset                => w_reset,
            
            -- User I/O
            button_pio_external_connection_export => KEY,
            led_pio_external_connection_export    => LEDR,
            
            -- SDRAM Connections
            clocks_sdram_clk_clk                  => DRAM_CLK,
            sdram_wire_addr                       => DRAM_ADDR,
            sdram_wire_ba                         => DRAM_BA,
            sdram_wire_cas_n                      => DRAM_CAS_N,
            sdram_wire_cke                        => DRAM_CKE,
            sdram_wire_cs_n                       => DRAM_CS_N,
            sdram_wire_dq                         => DRAM_DQ,
            sdram_wire_dqm                        => DRAM_DQM,
            sdram_wire_ras_n                      => DRAM_RAS_N,
            sdram_wire_we_n                       => DRAM_WE_N,
            
            -- Interrupts (we are not using them, so leave them open)
            high_res_timer_irq_irq                => open,
            jtag_uart_irq_irq                     => open
        );

end architecture rtl;

