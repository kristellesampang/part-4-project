-- File: debouncer.vhd
-- Description: Filters out mechanical switch noise and outputs a single-cycle pulse.

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity debouncer is
    generic (
        -- Debounce time: 1,000,000 cycles * 20ns/cycle (at 50MHz) = 20ms
        DEBOUNCE_CYCLES : integer := 1000000 
    );
    port (
        clk       : in  std_logic;
        reset     : in  std_logic;
        button_in : in  std_logic; -- Raw input from a physical button
        tick_out  : out std_logic  -- A single, clean clock tick output
    );
end entity debouncer;

architecture rtl of debouncer is
    type state_t is (IDLE, WAIT_STABLE, PRESSED, WAIT_RELEASE);
    signal current_state : state_t := IDLE;
    signal counter : integer range 0 to DEBOUNCE_CYCLES := 0;
begin

    debounce_proc : process(clk)
    begin
        if rising_edge(clk) then
            if reset = '1' then
                current_state <= IDLE;
                counter       <= 0;
                tick_out      <= '0';
            else
                tick_out <= '0'; -- Output pulse is normally '0'

                case current_state is
                    when IDLE =>
                        if button_in = '1' then
                            current_state <= WAIT_STABLE;
                            counter       <= 0;
                        end if;

                    when WAIT_STABLE =>
                        if button_in = '1' then
                            if counter < DEBOUNCE_CYCLES then
                                counter <= counter + 1;
                            else
                                current_state <= PRESSED;
                            end if;
                        else
                            current_state <= IDLE;
                        end if;

                    when PRESSED =>
                        tick_out      <= '1'; -- Output a single tick
                        current_state <= WAIT_RELEASE;

                    when WAIT_RELEASE =>
                        if button_in = '0' then
                            current_state <= IDLE;
                        end if;
                end case;
            end if;
        end if;
    end process debounce_proc;
end architecture rtl;