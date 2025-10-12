library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.custom_types.all;

entity tb_npu_wrapper is
end entity;

architecture sim of tb_npu_wrapper is

    -- Constants
    constant CLK_PER : time := 20 ns; -- 50 MHz

    -- Signals
    signal clk           : bit_1 := '0';
    signal reset         : bit_1 := '0';
    signal start         : bit_1 := '0';
    signal done          : bit_1;
    signal read_address  : bit_7 := (others => '0');
    signal read_data     : bit_32;
    signal rows          : integer range 0 to N-1;
    signal cols          : integer range 0 to N-1;
    signal read_rom_counter : integer range 0 to N*N + 5;

begin

    -- clock
    clk <= not clk after CLK_PER/2;

    -- DUT
    DUT: entity work.npu_wrapper
        port map (
            clk          => clk,
            reset        => reset,
            start        => start,
            done         => done,
            rows         => rows,
            cols         => cols,
            read_rom_counter => read_rom_counter,
            read_address => read_address,
            read_data    => read_data
        );

    -- Main FSM (similar style to tb_memory)
    Stim_FSM: process
        type state_t is (IDLE, APPLY_RESET, TRIGGER_START, WAIT_FOR_DONE, READ_RESULTS, FINISH);
        variable s : state_t := IDLE;
        variable a : integer := 0;
        variable w : std_logic_vector(31 downto 0);
    begin
        case s is
            when IDLE =>
                wait until rising_edge(clk);
                s := APPLY_RESET;

            when APPLY_RESET =>
                reset <= '1';
                wait until rising_edge(clk);
                wait until rising_edge(clk);
                reset <= '0';
                s := TRIGGER_START;

            when TRIGGER_START =>
                start <= '1';
                wait until rising_edge(clk);
                start <= '0';
                s := WAIT_FOR_DONE;

            when WAIT_FOR_DONE =>
                wait until rising_edge(clk);
                if done = '1' then
                    a := 0;
                    s := READ_RESULTS;
                end if;

            when READ_RESULTS =>
                read_address <= std_logic_vector(to_unsigned(a, 7));
                wait until rising_edge(clk); -- 1-cycle latency
                w := read_data;
                -- report "out[" & integer'image(a) & "] = 0x" & to_hstring(w);
                if a = N*N - 1 then
                    s := FINISH;
                else
                    a := a + 1;
                end if;

            when FINISH =>
                report "tb_npu_wrapper finished" severity note;
                wait;
        end case;
    end process;

end architecture;
