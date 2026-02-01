library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.custom_types.all;

entity npu_system_wrapper is
    port (
        clk           : in  bit_1;
        reset_n       : in  bit_1;

        -- Avalon-MM Slave Interface
        avs_address   : in  std_logic_vector(3 downto 0);
        avs_write     : in  bit_1;
        avs_writedata : in  std_logic_vector(31 downto 0);
        avs_read      : in  bit_1;
        avs_readdata  : out std_logic_vector(31 downto 0)
    );
end npu_system_wrapper;

architecture rtl of npu_system_wrapper is
    signal reg_ready       : bit_1 := '0';
    signal reg_m, reg_n, reg_k : integer := 0;
    signal n_cycle_count   : integer := 0;
    signal n_output        : systolic_array_matrix_output;

    signal n_reset      : bit_1;

begin
    n_reset <= not reset_n;

    process(clk, reset_n)
    begin
        if reset_n = '0' then 
            reg_ready <= '0';
        elsif rising_edge(clk) then
            if avs_write = '1' then 
                case avs_address is
                    when "0000" => -- Control Register
                        reg_ready <= avs_writedata(0);
                    when "0001" => -- M Register
                        reg_m <= to_integer(unsigned(avs_writedata(15 downto 0)));
                    when "0010" => -- N Register
                        reg_n <= to_integer(unsigned(avs_writedata(15 downto 0)));
                    when "0011" => -- K Register
                        reg_k <= to_integer(unsigned(avs_writedata(15 downto 0)));
                    when others => null;
                end case;
            end if;

            if avs_read = '1' then
                case avs_address is
                    when x"4" => avs_readdata <= std_logic_vector(to_signed(n_cycle_count, 32));

                    when others => avs_readdata <= (others => '0');
                end case;
            end if;
            
            if reg_ready = '1' then
                reg_ready <= '0'; -- Clear ready when done
            end if;
        end if;
    end process;

    
    NPU_CORE : entity work.top_level_systolic_array
    port map (
        clk           => clk,
        reset         => n_reset,
        ready         => reg_ready,
        matrix_data   => (others => (others => (others => '0'))), -- Placeholder
        matrix_weight => (others => (others => (others => '0'))), -- Placeholder
        active_rows   => reg_m,
        active_cols   => reg_n,
        active_k      => reg_k,
        output        => n_output, -- Not connected
        cycle_count   => n_cycle_count  -- Not connected
    );
end rtl;