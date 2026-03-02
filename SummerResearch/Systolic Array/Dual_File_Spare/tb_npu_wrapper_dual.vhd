library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.custom_types.all;
use work.custom_types_int8.all;
library std;
use std.textio.all;

entity tb_npu_wrapper_dual is
end tb_npu_wrapper_dual;

architecture sim of tb_npu_wrapper_dual is
    signal clk             : std_logic := '0';
    signal reset_n         : std_logic := '0';

    signal avs_address     : std_logic_vector(4 downto 0) := (others => '0');
    signal avs_write       : std_logic := '0';
    signal avs_writedata   : std_logic_vector(31 downto 0) := (others => '0');
    signal avs_read        : std_logic := '0';
    signal avs_readdata    : std_logic_vector(31 downto 0);
    signal avs_waitrequest : std_logic;

    signal avm_act_address    : std_logic_vector(31 downto 0);
    signal avm_act_read       : std_logic;
    signal avm_act_readdata   : std_logic_vector(31 downto 0) := (others => '0');
    signal avm_act_waitreq    : std_logic := '0';

    signal avm_weight_address    : std_logic_vector(31 downto 0);
    signal avm_weight_read       : std_logic;
    signal avm_weight_readdata   : std_logic_vector(31 downto 0) := (others => '0');
    signal avm_weight_waitreq    : std_logic := '0';

    signal avm_out_address    : std_logic_vector(31 downto 0);
    signal avm_out_write      : std_logic;
    signal avm_out_writedata  : std_logic_vector(31 downto 0);
    signal avm_out_waitreq    : std_logic := '0';

    constant CLK_PER : time := 10 ns;

    type mem_array is array(0 to 1023) of std_logic_vector(31 downto 0);
    signal act_mem    : mem_array := (others => (others => '0'));
    signal weight_mem : mem_array := (others => (others => '0'));

    -- ---- INT16 TEST CASE: 4x4 x 4x4 ----
    -- Data: 4x4, Weight: 4x4, K=4
    -- All values small for easy manual verification
    constant M16 : integer := 4;
    constant N16 : integer := 4;
    constant K16 : integer := 4;

    -- Data matrix (row-major, tight packed)
    -- [[1,2,3,4],[5,6,7,8],[1,1,1,1],[2,2,2,2]]
    type int16_test_array is array(0 to 15) of integer;
    constant DATA16 : int16_test_array := (1,2,3,4, 5,6,7,8, 1,1,1,1, 2,2,2,2);
    -- Weight matrix (row-major, tight packed)
    -- [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]] (identity)
    constant WEIGHT16 : int16_test_array := (1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1);

    -- ---- INT8 TEST CASE: 4x4 x 4x4 ----
    constant M8 : integer := 4;
    constant N8_VAL : integer := 4;
    constant K8 : integer := 4;

    -- Data: [[2,4,6,8],[1,3,5,7],[1,2,3,4],[2,1,2,1]]
    type int8_test_array is array(0 to 15) of integer;
    constant DATA8 : int8_test_array := (2,4,6,8, 1,3,5,7, 1,2,3,4, 2,1,2,1);
    -- Weight: identity
    constant WEIGHT8 : int8_test_array := (1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1);

    -- Output capture
    type result_array is array(0 to 15) of integer;
    signal results_int16 : result_array := (others => 0);
    signal results_int8  : result_array := (others => 0);
    signal out_idx       : integer := 0;
    signal capture_done_int16 : std_logic := '0';
    signal capture_done_int8  : std_logic := '0';
    signal current_mode  : std_logic := '0'; -- 0=int16, 1=int8

begin

    DUT: entity work.npu_system_wrapper
    port map (
        clk => clk, reset_n => reset_n,
        avs_address => avs_address, avs_write => avs_write,
        avs_writedata => avs_writedata, avs_read => avs_read,
        avs_readdata => avs_readdata, avs_waitrequest => avs_waitrequest,
        avm_act_address => avm_act_address, avm_act_read => avm_act_read,
        avm_act_readdata => avm_act_readdata, avm_act_waitreq => avm_act_waitreq,
        avm_weight_address => avm_weight_address, avm_weight_read => avm_weight_read,
        avm_weight_readdata => avm_weight_readdata, avm_weight_waitreq => avm_weight_waitreq,
        avm_out_address => avm_out_address, avm_out_write => avm_out_write,
        avm_out_writedata => avm_out_writedata, avm_out_waitreq => avm_out_waitreq
    );

    clk <= not clk after CLK_PER / 2;

    -- RAM response - strips base address with mod
    process(clk)
    begin
        if rising_edge(clk) then
            if avm_act_read = '1' then
                avm_act_readdata <= act_mem((to_integer(unsigned(avm_act_address)) mod 4096) / 4);
            end if;
            if avm_weight_read = '1' then
                avm_weight_readdata <= weight_mem((to_integer(unsigned(avm_weight_address)) mod 4096) / 4);
            end if;
        end if;
    end process;

    -- Output capture
    process(clk)
    begin
        if rising_edge(clk) then
            if avm_out_write = '1' then
                if current_mode = '0' then
                    results_int16(out_idx) <= to_integer(signed(avm_out_writedata));
                    if out_idx = M16 * N16 - 1 then
                        capture_done_int16 <= '1';
                        out_idx <= 0;
                    else
                        out_idx <= out_idx + 1;
                    end if;
                else
                    results_int8(out_idx) <= to_integer(signed(avm_out_writedata));
                    if out_idx = M8 * N8_VAL - 1 then
                        capture_done_int8 <= '1';
                        out_idx <= 0;
                    else
                        out_idx <= out_idx + 1;
                    end if;
                end if;
            end if;
        end if;
    end process;

    -- Main stimulus
    process
    begin
        -- Reset
        reset_n <= '0';
        wait for 50 ns;
        reset_n <= '1';
        wait for 50 ns;

        -- ============================================
        -- TEST 1: INT16 MODE (reg_config = 0)
        -- ============================================
        report "Loading INT16 memories...";

        -- Load data into act_mem (tight packed as int16 in lower 16 bits)
        for i in 0 to M16*K16-1 loop
            act_mem(i) <= std_logic_vector(to_signed(DATA16(i), 32));
        end loop;
        for i in 0 to K16*N16-1 loop
            weight_mem(i) <= std_logic_vector(to_signed(WEIGHT16(i), 32));
        end loop;
        wait for 20 ns;

        -- Set config = 0 (Int16)
        avs_address <= "10000"; avs_writedata <= x"00000000"; avs_write <= '1';
        wait until rising_edge(clk); avs_write <= '0'; wait for 20 ns;

        -- Set M, N, K
        avs_address <= "00100"; avs_writedata <= std_logic_vector(to_unsigned(M16, 32)); avs_write <= '1';
        wait until rising_edge(clk);
        avs_address <= "01000"; avs_writedata <= std_logic_vector(to_unsigned(N16, 32)); avs_write <= '1';
        wait until rising_edge(clk);
        avs_address <= "01100"; avs_writedata <= std_logic_vector(to_unsigned(K16, 32)); avs_write <= '1';
        wait until rising_edge(clk);
        avs_write <= '0'; wait for 20 ns;

        current_mode <= '0';

        -- Trigger
        avs_address <= "00000"; avs_writedata <= x"00000001"; avs_write <= '1';
        wait until rising_edge(clk); avs_write <= '0';

        -- Wait for completion
        wait until capture_done_int16 = '1';
        wait for 100 ns;

        -- Print results
        report "=== INT16 RESULTS (expect identity*data = data) ===";
        for i in 0 to M16-1 loop
            for j in 0 to N16-1 loop
                report "  [" & integer'image(i) & "," & integer'image(j) & "] = " &
                       integer'image(results_int16(i*N16+j));
            end loop;
        end loop;

        -- ============================================
        -- TEST 2: INT8 MODE (reg_config = 1)
        -- ============================================
        report "Loading INT8 memories...";
        wait for 200 ns;

        -- Reload memories with int8 data (still in lower 8 bits of each word)
        for i in 0 to M8*K8-1 loop
            act_mem(i) <= std_logic_vector(to_signed(DATA8(i), 32));
        end loop;
        for i in 0 to K8*N8_VAL-1 loop
            weight_mem(i) <= std_logic_vector(to_signed(WEIGHT8(i), 32));
        end loop;
        wait for 20 ns;

        -- Set config = 1 (Int8)
        avs_address <= "10000"; avs_writedata <= x"00000001"; avs_write <= '1';
        wait until rising_edge(clk); avs_write <= '0'; wait for 20 ns;

        -- Set M, N, K
        avs_address <= "00100"; avs_writedata <= std_logic_vector(to_unsigned(M8, 32)); avs_write <= '1';
        wait until rising_edge(clk);
        avs_address <= "01000"; avs_writedata <= std_logic_vector(to_unsigned(N8_VAL, 32)); avs_write <= '1';
        wait until rising_edge(clk);
        avs_address <= "01100"; avs_writedata <= std_logic_vector(to_unsigned(K8, 32)); avs_write <= '1';
        wait until rising_edge(clk);
        avs_write <= '0'; wait for 20 ns;

        current_mode <= '1';

        -- Trigger
        avs_address <= "00000"; avs_writedata <= x"00000001"; avs_write <= '1';
        wait until rising_edge(clk); avs_write <= '0';

        -- Wait for completion
        wait until capture_done_int8 = '1';
        wait for 100 ns;

        -- Print results
        report "=== INT8 RESULTS (expect identity*data = data) ===";
        for i in 0 to M8-1 loop
            for j in 0 to N8_VAL-1 loop
                report "  [" & integer'image(i) & "," & integer'image(j) & "] = " &
                       integer'image(results_int8(i*N8_VAL+j));
            end loop;
        end loop;

        report "=== SIMULATION COMPLETE ===";
        wait;
    end process;

end architecture;