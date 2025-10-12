-- File: UART.vhd
-- Sourced from Noam Elbaum's VHDL-UART repository

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity UART is
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
end entity UART;


architecture RTL of UART is

    -- Baud rate generator logic
    constant c_CLOCK_PERIOD_NS : integer := 1000 / (CLOCK_FREQUENCY / 1_000_000);
    constant c_BIT_PERIOD      : integer := CLOCK_FREQUENCY / BAUD_RATE;

    -- TX state machine
    type t_TX_State is (TX_IDLE, TX_START, TX_DATA, TX_STOP);
    signal r_TX_State : t_TX_State := TX_IDLE;
    
    -- RX state machine
    type t_RX_State is (RX_IDLE, RX_START, RX_DATA, RX_STOP);
    signal r_RX_State : t_RX_State := RX_IDLE;
    
    signal r_TX_Clock_Count : integer range 0 to c_BIT_PERIOD - 1 := 0;
    signal r_TX_Bit_Index   : integer range 0 to 7 := 0;
    signal r_TX_Data        : std_logic_vector(7 downto 0) := (others => '0');

    signal r_RX_Clock_Count : integer range 0 to c_BIT_PERIOD - 1 := 0;
    signal r_RX_Bit_Index   : integer range 0 to 7 := 0;
    signal r_RX_Data        : std_logic_vector(7 downto 0) := (others => '0');


begin
    
    ------------------------------------------------------------------------
    -- RX Logic
    ------------------------------------------------------------------------
    p_RX_Logic : process(i_Clock)
    begin
        if rising_edge(i_Clock) then
            case r_RX_State is
                when RX_IDLE =>
                    o_RX_DV <= '0';
                    
                    -- Wait for start bit
                    if i_RX_Serial = '0' then
                        r_RX_State       <= RX_START;
                        r_RX_Clock_Count <= 0;
                    end if;
                
                when RX_START =>
                    if r_RX_Clock_Count = (c_BIT_PERIOD / 2) then -- Sample in the middle of the start bit
                        if i_RX_Serial = '0' then
                            r_RX_State       <= RX_DATA;
                            r_RX_Clock_Count <= 0;
                            r_RX_Bit_Index   <= 0;
                        else
                            -- Glitch or noise, return to idle
                            r_RX_State <= RX_IDLE;
                        end if;
                    else
                        r_RX_Clock_Count <= r_RX_Clock_Count + 1;
                    end if;

                when RX_DATA =>
                    if r_RX_Clock_Count < c_BIT_PERIOD - 1 then
                        r_RX_Clock_Count <= r_RX_Clock_Count + 1;
                    else
                        r_RX_Data(r_RX_Bit_Index) <= i_RX_Serial; -- Sample data
                        r_RX_Clock_Count          <= 0;
                        
                        if r_RX_Bit_Index < 7 then
                            r_RX_Bit_Index <= r_RX_Bit_Index + 1;
                        else
                            r_RX_State <= RX_STOP;
                        end if;
                    end if;

                when RX_STOP =>
                    if r_RX_Clock_Count < c_BIT_PERIOD - 1 then
                        r_RX_Clock_Count <= r_RX_Clock_Count + 1;
                    else
                        o_RX_Byte        <= r_RX_Data;
                        o_RX_DV          <= '1'; -- Signal Data Valid
                        r_RX_Clock_Count <= 0;
                        r_RX_State       <= RX_IDLE;
                    end if;

                when others =>
                    r_RX_State <= RX_IDLE;
            end case;
        end if;
    end process p_RX_Logic;


    ------------------------------------------------------------------------
    -- TX Logic
    ------------------------------------------------------------------------
    p_TX_Logic : process(i_Clock)
    begin
        if rising_edge(i_Clock) then
            case r_TX_State is
                when TX_IDLE =>
                    o_TX_Serial <= '1'; -- Drive line high when idle (Stop bit/Mark state)
                    o_TX_Active <= '0';

                    if i_TX_DV = '1' then
                        r_TX_State       <= TX_START;
                        r_TX_Data        <= i_TX_Byte;
                        r_TX_Clock_Count <= 0;
                    end if;

                when TX_START =>
                    o_TX_Serial <= '0'; -- Start bit (Space state)
                    o_TX_Active <= '1';
                    
                    if r_TX_Clock_Count < c_BIT_PERIOD - 1 then
                        r_TX_Clock_Count <= r_TX_Clock_Count + 1;
                    else
                        r_TX_State       <= TX_DATA;
                        r_TX_Clock_Count <= 0;
                        r_TX_Bit_Index   <= 0;
                    end if;

                when TX_DATA =>
                    o_TX_Serial <= r_TX_Data(r_TX_Bit_Index);

                    if r_TX_Clock_Count < c_BIT_PERIOD - 1 then
                        r_TX_Clock_Count <= r_TX_Clock_Count + 1;
                    else
                        r_TX_Clock_Count <= 0;
                        
                        if r_TX_Bit_Index < 7 then
                           r_TX_Bit_Index <= r_TX_Bit_Index + 1;
                        else
                           r_TX_State <= TX_STOP;
                        end if;
                    end if;

                when TX_STOP =>
                    o_TX_Serial <= '1'; -- Stop bit (Mark state)
                    
                    if r_TX_Clock_Count < c_BIT_PERIOD - 1 then
                        r_TX_Clock_Count <= r_TX_Clock_Count + 1;
                    else
                        r_TX_State       <= TX_IDLE;
                        r_TX_Clock_Count <= 0;
                    end if;

                when others =>
                    r_TX_State <= TX_IDLE;
            end case;
        end if;
    end process p_TX_Logic;

end RTL;