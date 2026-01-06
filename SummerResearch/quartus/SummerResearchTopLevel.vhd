library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.custom_types.all;

entity SummerResearchTopLevel is
    port (
        -- Basic Clock/Reset
        CLK_50         : in  std_logic;
        BTN_RESET      : in  std_logic;
        BTN_START      : in  std_logic;

        -- DDR4 External Memory Interface (Moved from PR_Test_Top to here)
        ddr4_emif_oct_oct_rzqin        : in    std_logic                     := '0';
        ddr4_emif_mem_mem_ck           : out   std_logic_vector(0 downto 0);
        ddr4_emif_mem_mem_ck_n         : out   std_logic_vector(0 downto 0);
        ddr4_emif_mem_mem_a            : out   std_logic_vector(16 downto 0);
        ddr4_emif_mem_mem_act_n        : out   std_logic_vector(0 downto 0);
        ddr4_emif_mem_mem_ba           : out   std_logic_vector(1 downto 0);
        ddr4_emif_mem_mem_bg           : out   std_logic_vector(0 downto 0);
        ddr4_emif_mem_mem_cke          : out   std_logic_vector(0 downto 0);
        ddr4_emif_mem_mem_cs_n         : out   std_logic_vector(0 downto 0);
        ddr4_emif_mem_mem_odt          : out   std_logic_vector(0 downto 0);
        ddr4_emif_mem_mem_reset_n      : out   std_logic_vector(0 downto 0);
        ddr4_emif_mem_mem_par          : out   std_logic_vector(0 downto 0);
        ddr4_emif_mem_mem_alert_n      : in    std_logic_vector(0 downto 0)  := (others => '0');
        ddr4_emif_mem_mem_dqs          : inout std_logic_vector(7 downto 0)  := (others => '0');
        ddr4_emif_mem_mem_dqs_n        : inout std_logic_vector(7 downto 0)  := (others => '0');
        ddr4_emif_mem_mem_dq           : inout std_logic_vector(63 downto 0) := (others => '0');
        ddr4_emif_mem_mem_dbi_n        : inout std_logic_vector(7 downto 0)  := (others => '0');

        -- Status and LEDs
        LED_STATUS      : out std_logic_vector(7 downto 0);
        LED_DONE        : out std_logic
    );
end SummerResearchTopLevel;

architecture rtl of SummerResearchTopLevel is

    -- 1. The Platform Designer (Qsys) Component Declaration
    component PR_Test_Top is
        port (
            clk_clk                        : in    std_logic                     := '0';
            reset_reset                    : in    std_logic                     := '0';
            pr_freeze_freeze               : out   std_logic;
            sdram_pll_locked_pll_locked    : out   std_logic;
            ddr4_emif_oct_oct_rzqin        : in    std_logic                     := '0';
            ddr4_emif_mem_mem_ck           : out   std_logic_vector(0 downto 0);
            ddr4_emif_mem_mem_ck_n         : out   std_logic_vector(0 downto 0);
            ddr4_emif_mem_mem_a            : out   std_logic_vector(16 downto 0);
            ddr4_emif_mem_mem_act_n        : out   std_logic_vector(0 downto 0);
            ddr4_emif_mem_mem_ba           : out   std_logic_vector(1 downto 0);
            ddr4_emif_mem_mem_bg           : out   std_logic_vector(0 downto 0);
            ddr4_emif_mem_mem_cke          : out   std_logic_vector(0 downto 0);
            ddr4_emif_mem_mem_cs_n         : out   std_logic_vector(0 downto 0);
            ddr4_emif_mem_mem_odt          : out   std_logic_vector(0 downto 0);
            ddr4_emif_mem_mem_reset_n      : out   std_logic_vector(0 downto 0);
            ddr4_emif_mem_mem_par          : out   std_logic_vector(0 downto 0);
            ddr4_emif_mem_mem_alert_n      : in    std_logic_vector(0 downto 0);
            ddr4_emif_mem_mem_dqs          : inout std_logic_vector(7 downto 0);
            ddr4_emif_mem_mem_dqs_n        : inout std_logic_vector(7 downto 0);
            ddr4_emif_mem_mem_dq           : inout std_logic_vector(63 downto 0);
            ddr4_emif_mem_mem_dbi_n        : inout std_logic_vector(7 downto 0);
            sdram_status_local_cal_success : out   std_logic;
            sdram_status_local_cal_fail    : out   std_logic
        );
    end component;

    -- 2. Your NPU Component Declaration
    component top_level_systolic_array is
    port (
        clk           : in  bit_1;
        reset         : in  bit_1;
        ready         : in  bit_1;
        matrix_data   : in  systolic_array_matrix_input;
        matrix_weight : in  systolic_array_matrix_input;
        active_rows   : in  integer;
        active_cols   : in  integer;
        active_k      : in  integer;
        output        : out systolic_array_matrix_output;
        cycle_count   : out integer
    );
    end component;

    -- Internal Signals
    signal pr_freeze_sig : std_logic;
    signal internal_clk   : bit_1;
    signal internal_reset : bit_1;
    signal sa_output_sig  : systolic_array_matrix_output;
    signal sa_cycle_count : integer;

    -- Helper functions for bit_1 conversion
    function to_bit(s : std_logic) return bit_1 is
        variable result_val : bit_1;
    begin
        if s = '1' then result_val := '1'; else result_val := '0'; end if;
        return result_val;
    end function;

    function u16(x : integer) return bit_16 is
    begin return std_logic_vector(to_signed(x, 16)); end function;

    constant TEST_M : integer := 32;
    constant TEST_K : integer := 32;
    constant TEST_N : integer := 32;
    constant MAX_LATENCY : integer := TEST_M + TEST_N + TEST_K - 2;

begin
    internal_clk <= to_bit(CLK_50);
    -- NPU reset is high if global reset is active OR if PR is occurring
    internal_reset <= to_bit(not BTN_RESET) or to_bit(pr_freeze_sig);

    -- 3. Instantiate the Platform Designer System
    qsys_inst : component PR_Test_Top
        port map (
            clk_clk                        => CLK_50,
            reset_reset                    => not BTN_RESET,
            pr_freeze_freeze               => pr_freeze_sig, -- Connects to NPU reset/ready
            
            -- Direct DDR4 pin mapping
            ddr4_emif_oct_oct_rzqin        => ddr4_emif_oct_oct_rzqin,
            ddr4_emif_mem_mem_ck           => ddr4_emif_mem_mem_ck,
            ddr4_emif_mem_mem_ck_n         => ddr4_emif_mem_mem_ck_n,
            ddr4_emif_mem_mem_a            => ddr4_emif_mem_mem_a,
            ddr4_emif_mem_mem_act_n        => ddr4_emif_mem_mem_act_n,
            ddr4_emif_mem_mem_ba           => ddr4_emif_mem_mem_ba,
            ddr4_emif_mem_mem_bg           => ddr4_emif_mem_mem_bg,
            ddr4_emif_mem_mem_cke          => ddr4_emif_mem_mem_cke,
            ddr4_emif_mem_mem_cs_n         => ddr4_emif_mem_mem_cs_n,
            ddr4_emif_mem_mem_odt          => ddr4_emif_mem_mem_odt,
            ddr4_emif_mem_mem_reset_n      => ddr4_emif_mem_mem_reset_n,
            ddr4_emif_mem_mem_par          => ddr4_emif_mem_mem_par,
            ddr4_emif_mem_mem_alert_n      => ddr4_emif_mem_mem_alert_n,
            ddr4_emif_mem_mem_dqs          => ddr4_emif_mem_mem_dqs,
            ddr4_emif_mem_mem_dqs_n        => ddr4_emif_mem_mem_dqs_n,
            ddr4_emif_mem_mem_dq           => ddr4_emif_mem_mem_dq,
            ddr4_emif_mem_mem_dbi_n        => ddr4_emif_mem_mem_dbi_n,
            
            sdram_pll_locked_pll_locked    => open,
            sdram_status_local_cal_success => open,
            sdram_status_local_cal_fail    => open
        );

    -- 4. Instantiate the NPU (The Part we will make a Partition)
    npu_core : component top_level_systolic_array
        port map (
            clk           => internal_clk,
            reset         => internal_reset, -- Held high during PR
            ready         => to_bit(BTN_START) and not to_bit(pr_freeze_sig),
            matrix_data   => (others => (others => u16(1))),
            matrix_weight => (others => (others => u16(1))),
            active_rows   => TEST_M,
            active_cols   => TEST_N,
            active_k      => TEST_K,
            output        => sa_output_sig,
            cycle_count   => sa_cycle_count
        );

    LED_STATUS <= std_logic_vector(to_unsigned(sa_cycle_count,8));
    LED_DONE <= '1' when sa_cycle_count >= MAX_LATENCY else '0';
    
end architecture rtl;