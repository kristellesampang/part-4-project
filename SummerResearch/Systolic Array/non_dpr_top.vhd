library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity non_dpr_top is
    port (
        -- Basic Clock/Reset from Arria 10 Pins
        CLK_50         : in  std_logic;
        BTN_RESET      : in  std_logic; -- Assuming active low from board

        -- Status LEDs for debugging
        LED_STATUS      : out std_logic_vector(7 downto 0)
    );
end non_dpr_top;

architecture rtl of non_dpr_top is
    signal actual_reset : std_logic;

    signal blink_reg : unsigned (24 downto 0);

    -- 1. The Platform Designer (Qsys) Component Declaration
    -- This matches the Verilog module ports you provided
    component NonDPR is
        port (
            clk_clk       : in std_logic := 'X'; 
            reset_reset   : in std_logic := 'X'  
        );
    end component NonDPR;

begin

    actual_reset <= BTN_RESET;

    System : component NonDPR
        port map (
            clk_clk       => CLK_50,
            reset_reset   => actual_reset
        );

    process(CLK_50) begin
        if rising_edge(CLK_50) then
            blink_reg <= blink_reg + 1;
        end if;
    end process;
    LED_STATUS(0) <= std_logic(blink_reg(22));
    LED_STATUS(1) <= std_logic(blink_reg(23));
    LED_STATUS(2) <= std_logic(blink_reg(24));
    LED_STATUS(3) <= not BTN_RESET; -- Active low reset indication

end architecture rtl;