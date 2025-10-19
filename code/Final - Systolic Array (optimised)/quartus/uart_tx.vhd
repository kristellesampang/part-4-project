-- File: uart_tx.vhd
-- Description: A simple 9600 baud UART transmitter.

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity uart_tx is
    generic (
        -- For a 50MHz clock: 50,000,000 / 9600 = 5208.33. We'll use 5208.
        CLK_PER_BIT : integer := 5208 
    );
    port (
        clk      : in  std_logic;
        reset    : in  std_logic;
        start    : in  std_logic;
        data_in  : in  std_logic_vector(7 downto 0);
        busy     : out std_logic;
        tx_line  : out std_logic
    );
end entity uart_tx;

architecture rtl of uart_tx is
    type state_t is (IDLE, START_BIT, DATA_BITS, STOP_BIT);
    signal current_state : state_t := IDLE;
    signal clock_counter : integer range 0 to CLK_PER_BIT - 1 := 0;
    signal bit_index     : integer range 0 to 7 := 0;
    signal data_buffer   : std_logic_vector(7 downto 0);
begin

    process(clk)
    begin
        if rising_edge(clk) then
            if reset = '1' then
                current_state <= IDLE;
                clock_counter <= 0;
                bit_index     <= 0;
                tx_line       <= '1'; -- Idle line is high
                busy          <= '0';
            else
                case current_state is
                    when IDLE =>
                        tx_line <= '1';
                        busy    <= '0';
                        if start = '1' then
                            data_buffer   <= data_in;
                            clock_counter <= 0;
                            bit_index     <= 0;
                            current_state <= START_BIT;
                            busy          <= '1';
                        end if;

                    when START_BIT =>
                        tx_line <= '0'; -- Start bit is low
                        if clock_counter < CLK_PER_BIT - 1 then
                            clock_counter <= clock_counter + 1;
                        else
                            clock_counter <= 0;
                            current_state <= DATA_BITS;
                        end if;

                    when DATA_BITS =>
                        tx_line <= data_buffer(bit_index);
                        if clock_counter < CLK_PER_BIT - 1 then
                            clock_counter <= clock_counter + 1;
                        else
                            clock_counter <= 0;
                            if bit_index < 7 then
                                bit_index <= bit_index + 1;
                            else
                                current_state <= STOP_BIT;
                            end if;
                        end if;

                    when STOP_BIT =>
                        tx_line <= '1'; -- Stop bit is high
                        if clock_counter < CLK_PER_BIT - 1 then
                            clock_counter <= clock_counter + 1;
                        else
                            current_state <= IDLE;
                        end if;
                end case;
            end if;
        end if;
    end process;
end architecture rtl;