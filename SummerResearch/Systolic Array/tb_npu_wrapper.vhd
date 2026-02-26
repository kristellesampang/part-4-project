library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.custom_types.all;
library std;
use std.textio.all;
use ieee.std_logic_textio.all;

entity tb_npu_wrapper is
end tb_npu_wrapper;

architecture sim of tb_npu_wrapper is
    signal clk           : std_logic := '0';
    signal reset_n       : std_logic := '0';

    -- Avalon-MM Interfaces
    signal avs_address   : std_logic_vector(4 downto 0) := (others => '0');
    signal avs_write     : std_logic := '0';
    signal avs_writedata : std_logic_vector(31 downto 0) := (others => '0');
    signal avs_read      : std_logic := '0';
    signal avs_readdata  : std_logic_vector(31 downto 0);
    signal avs_waitrequest : std_logic;

    signal avm_act_address    : std_logic_vector(31 downto 0);
    signal avm_act_read       : std_logic;
    signal avm_act_readdata   : std_logic_vector(31 downto 0);
    signal avm_act_waitreq    : std_logic := '0';

    signal avm_weight_address    : std_logic_vector(31 downto 0);
    signal avm_weight_read       : std_logic;
    signal avm_weight_readdata   : std_logic_vector(31 downto 0);
    signal avm_weight_waitreq    : std_logic := '0';

    signal avm_out_address    : std_logic_vector(31 downto 0);
    signal avm_out_write      : std_logic;
    signal avm_out_writedata  : std_logic_vector(31 downto 0);
    signal avm_out_waitreq    : std_logic := '0';

    constant clk_period : time := 10 ns;

    -- Memory Signals for Avalon Masters to fetch from
    type mem_array is array(0 to 1023) of std_logic_vector(31 downto 0);
    signal act_mem : mem_array := (others => (others => '0'));
    signal weight_mem : mem_array := (others => (others => '0'));

    -- 1. Helper function for Python-to-VHDL integer conversion
    function s16(x : integer) return std_logic_vector is
    begin
        -- Cast to 32-bit for your Avalon bus, even though data is 16-bit
        return std_logic_vector(to_signed(x, 16));
    end function;

    -- 2. PASTE THE COMPACT DATA_STIM AND WEIGHT_STIM HERE
    -- (The output you got from the last Python script)
    constant DATA_STIM : systolic_array_matrix_input := (
                (s16(0), s16(0), s16(0), s16(0), s16(0), s16(1), s16(0), s16(0), s16(2), s16(0), s16(0), s16(0), s16(1), s16(0), s16(0), s16(3), s16(1), s16(0), s16(3), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0)),
        (s16(0), s16(0), s16(0), s16(0), s16(1), s16(0), s16(0), s16(2), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0), s16(1), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0)),
        (s16(0), s16(0), s16(0), s16(1), s16(0), s16(0), s16(2), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0)),
        (s16(0), s16(0), s16(2), s16(0), s16(0), s16(3), s16(0), s16(0), s16(4), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0)),
        (s16(0), s16(2), s16(0), s16(0), s16(3), s16(0), s16(0), s16(4), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0)),
        (s16(0), s16(1), s16(1), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0)),
        (s16(1), s16(1), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0), s16(1), s16(0), s16(1), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0)),
        (s16(1), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0), s16(1), s16(0), s16(0), s16(1), s16(0), s16(0), s16(1), s16(0), s16(1), s16(3), s16(1), s16(1), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0)),
        (s16(0), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0), s16(1), s16(0), s16(0), s16(1), s16(0), s16(0), s16(1), s16(1), s16(1), s16(3), s16(0), s16(1), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0)),
        (s16(0), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0), s16(1), s16(0), s16(0), s16(1), s16(0), s16(0), s16(1), s16(1), s16(0), s16(3), s16(0), s16(2), s16(0), s16(1), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0)),
        (s16(0), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0), s16(1), s16(0), s16(0), s16(0), s16(2), s16(2), s16(1), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0)),
        (s16(0), s16(0), s16(2), s16(0), s16(0), s16(0), s16(0), s16(0), s16(1), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0), s16(2), s16(2), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0)),
        (s16(0), s16(2), s16(2), s16(0), s16(0), s16(0), s16(0), s16(1), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0), s16(2), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0)),
        (s16(2), s16(2), s16(2), s16(0), s16(0), s16(0), s16(1), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0), s16(2), s16(0), s16(2), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0)),
        (s16(2), s16(2), s16(0), s16(0), s16(0), s16(1), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0), s16(2), s16(0), s16(2), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0)),
        (s16(2), s16(0), s16(0), s16(0), s16(1), s16(1), s16(0), s16(0), s16(0), s16(0), s16(0), s16(1), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0), s16(2), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0)),
        (s16(0), s16(0), s16(0), s16(1), s16(1), s16(0), s16(0), s16(0), s16(0), s16(0), s16(1), s16(1), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0)),
        (s16(0), s16(0), s16(0), s16(1), s16(0), s16(2), s16(0), s16(0), s16(0), s16(1), s16(1), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0)),
        (s16(0), s16(0), s16(0), s16(0), s16(2), s16(2), s16(0), s16(0), s16(0), s16(1), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0)),
        (s16(0), s16(0), s16(0), s16(2), s16(2), s16(2), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0)),
        (s16(0), s16(0), s16(0), s16(2), s16(2), s16(1), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0)),
        (s16(0), s16(0), s16(0), s16(2), s16(1), s16(2), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0), s16(0)), 
        others => (others => s16(0))
    );


    constant WEIGHT_STIM : systolic_array_matrix_input := (
        (s16(26), s16(21), s16(21), s16(23), s16(23), s16(15), s16(11), s16(11), s16(12), s16(5), s16(11), s16(17), s16(9), s16(12), s16(17), s16(16), s16(16), s16(12), s16(6), s16(6), s16(-3), s16(1), s16(17), s16(9), s16(12), s16(12), s16(12), s16(11), s16(11), s16(6), s16(10), s16(2)),
        (s16(-1), s16(2), s16(32), s16(35), s16(25), s16(29), s16(41), s16(33), s16(41), s16(17), s16(13), s16(-8), s16(-3), s16(12), s16(13), s16(26), s16(22), s16(31), s16(21), s16(36), s16(11), s16(16), s16(2), s16(-1), s16(18), s16(21), s16(18), s16(18), s16(13), s16(-18), s16(-17), s16(-41)),
        (s16(-13), s16(3), s16(2), s16(22), s16(37), s16(40), s16(22), s16(0), s16(-23), s16(-27), s16(-52), s16(-22), s16(-3), s16(-8), s16(17), s16(41), s16(54), s16(45), s16(18), s16(-18), s16(-36), s16(-73), s16(-32), s16(-10), s16(-12), s16(25), s16(67), s16(99), s16(112), s16(92), s16(52), s16(8)),
        (s16(-2), s16(-1), s16(6), s16(11), s16(11), s16(13), s16(11), s16(12), s16(11), s16(5), s16(-2), s16(1), s16(-10), s16(-15), s16(-15), s16(-12), s16(-3), s16(3), s16(4), s16(4), s16(7), s16(4), s16(2), s16(-9), s16(-19), s16(-38), s16(-57), s16(-63), s16(-64), s16(-55), s16(-44), s16(-27)),
        (s16(41), s16(13), s16(72), s16(31), s16(14), s16(-5), s16(-16), s16(-28), s16(-47), s16(-70), s16(-37), s16(23), s16(9), s16(66), s16(31), s16(10), s16(-4), s16(-16), s16(-27), s16(-38), s16(-56), s16(-22), s16(50), s16(33), s16(97), s16(61), s16(23), s16(-14), s16(-36), s16(-50), s16(-60), s16(-79)),
        (s16(48), s16(-26), s16(56), s16(-9), s16(16), s16(-2), s16(-17), s16(-16), s16(2), s16(-51), s16(-9), s16(-6), s16(-41), s16(4), s16(-14), s16(3), s16(4), s16(4), s16(-5), s16(19), s16(-5), s16(10), s16(59), s16(-2), s16(48), s16(8), s16(8), s16(5), s16(-3), s16(-27), s16(-1), s16(-44)),
        (s16(-11), s16(-19), s16(3), s16(9), s16(29), s16(40), s16(41), s16(27), s16(2), s16(-9), s16(-21), s16(-10), s16(-12), s16(5), s16(5), s16(12), s16(6), s16(-1), s16(-13), s16(-18), s16(-14), s16(-6), s16(-1), s16(-11), s16(20), s16(19), s16(19), s16(15), s16(4), s16(-19), s16(-29), s16(-24)),
        (s16(-20), s16(-26), s16(-18), s16(-34), s16(-30), s16(-15), s16(-3), s16(-1), s16(-7), s16(5), s16(-3), s16(-1), s16(-10), s16(-11), s16(-34), s16(-27), s16(-7), s16(-10), s16(-18), s16(-18), s16(7), s16(6), s16(-8), s16(-11), s16(3), s16(-23), s16(-46), s16(-35), s16(-33), s16(-17), s16(0), s16(14)),
        (s16(30), s16(-81), s16(86), s16(9), s16(-50), s16(31), s16(18), s16(-5), s16(-14), s16(66), s16(-39), s16(23), s16(-72), s16(69), s16(-65), s16(33), s16(-10), s16(-61), s16(61), s16(-67), s16(53), s16(-43), s16(15), s16(-46), s16(127), s16(-106), s16(77), s16(0), s16(-44), s16(93), s16(-44), s16(43)),
        (s16(-2), s16(0), s16(-7), s16(6), s16(12), s16(-20), s16(-5), s16(16), s16(1), s16(-2), s16(-5), s16(2), s16(2), s16(-14), s16(-2), s16(19), s16(-14), s16(-15), s16(21), s16(11), s16(-11), s16(1), s16(4), s16(3), s16(-10), s16(-12), s16(36), s16(0), s16(-44), s16(8), s16(28), s16(-10)),
        (s16(17), s16(38), s16(-5), s16(-2), s16(-6), s16(4), s16(7), s16(24), s16(45), s16(40), s16(73), s16(34), s16(49), s16(8), s16(12), s16(10), s16(10), s16(11), s16(14), s16(26), s16(36), s16(87), s16(1), s16(12), s16(-65), s16(-69), s16(-87), s16(-88), s16(-102), s16(-118), s16(-101), s16(-75)),
        (s16(6), s16(-11), s16(-37), s16(-35), s16(-36), s16(-32), s16(-27), s16(-13), s16(6), s16(1), s16(1), s16(-13), s16(-27), s16(-53), s16(-53), s16(-47), s16(-29), s16(-23), s16(2), s16(18), s16(8), s16(8), s16(-45), s16(-54), s16(-87), s16(-87), s16(-70), s16(-49), s16(-39), s16(1), s16(14), s16(11)),
        (s16(-13), s16(-13), s16(-6), s16(-9), s16(-13), s16(-12), s16(-7), s16(-7), s16(-7), s16(-10), s16(-20), s16(-8), s16(-8), s16(-1), s16(-5), s16(-8), s16(-4), s16(-3), s16(-5), s16(-4), s16(-2), s16(-16), s16(-4), s16(-2), s16(14), s16(18), s16(16), s16(14), s16(12), s16(6), s16(3), s16(3)),
        (s16(-13), s16(-17), s16(-18), s16(-19), s16(-23), s16(-22), s16(-28), s16(-20), s16(-22), s16(-16), s16(-12), s16(21), s16(14), s16(20), s16(16), s16(18), s16(16), s16(18), s16(19), s16(19), s16(19), s16(20), s16(18), s16(19), s16(24), s16(22), s16(28), s16(31), s16(35), s16(38), s16(37), s16(26)),
        (s16(-22), s16(15), s16(-30), s16(20), s16(41), s16(38), s16(22), s16(7), s16(-14), s16(-46), s16(14), s16(-10), s16(21), s16(-40), s16(-13), s16(10), s16(12), s16(-7), s16(-10), s16(-5), s16(-40), s16(12), s16(-1), s16(14), s16(-34), s16(-5), s16(37), s16(47), s16(18), s16(3), s16(4), s16(-39)),
        (s16(-1), s16(3), s16(-13), s16(16), s16(27), s16(-15), s16(-26), s16(-1), s16(15), s16(-5), s16(1), s16(1), s16(2), s16(-18), s16(10), s16(30), s16(-17), s16(-21), s16(12), s16(16), s16(-8), s16(-1), s16(0), s16(4), s16(-24), s16(20), s16(40), s16(-37), s16(-31), s16(19), s16(15), s16(-10)),
        (s16(-2), s16(-3), s16(-1), s16(0), s16(3), s16(3), s16(-4), s16(-5), s16(-2), s16(-3), s16(-1), s16(8), s16(7), s16(-2), s16(-8), s16(-10), s16(-4), s16(2), s16(10), s16(10), s16(6), s16(2), s16(-4), s16(4), s16(10), s16(14), s16(10), s16(4), s16(-1), s16(-9), s16(-8), s16(-8)),
        (s16(41), s16(29), s16(33), s16(23), s16(30), s16(42), s16(25), s16(2), s16(-16), s16(-21), s16(-1), s16(26), s16(14), s16(13), s16(1), s16(16), s16(22), s16(4), s16(-14), s16(-21), s16(-23), s16(-28), s16(46), s16(32), s16(32), s16(16), s16(39), s16(47), s16(27), s16(5), s16(-17), s16(-28)),
        (s16(-2), s16(3), s16(-14), s16(27), s16(-39), s16(8), s16(13), s16(-7), s16(-2), s16(1), s16(3), s16(-1), s16(12), s16(-25), s16(44), s16(-42), s16(8), s16(25), s16(-17), s16(5), s16(5), s16(-2), s16(-3), s16(11), s16(-28), s16(63), s16(-65), s16(1), s16(49), s16(-42), s16(4), s16(14)),
        (s16(1), s16(17), s16(14), s16(-41), s16(-37), s16(12), s16(38), s16(15), s16(-14), s16(-3), s16(1), s16(-3), s16(13), s16(30), s16(-33), s16(-37), s16(16), s16(33), s16(6), s16(-17), s16(-4), s16(1), s16(-5), s16(19), s16(43), s16(-43), s16(-64), s16(17), s16(47), s16(8), s16(-23), s16(-8)),
        (s16(-68), s16(-14), s16(34), s16(85), s16(83), s16(65), s16(40), s16(12), s16(-19), s16(-42), s16(-23), s16(-79), s16(-27), s16(4), s16(59), s16(65), s16(48), s16(16), s16(3), s16(-17), s16(-26), s16(-7), s16(-89), s16(-36), s16(15), s16(90), s16(102), s16(54), s16(8), s16(-9), s16(-19), s16(-26)),
        (s16(20), s16(-11), s16(-22), s16(-64), s16(3), s16(59), s16(63), s16(1), s16(-24), s16(-14), s16(-3), s16(25), s16(4), s16(-21), s16(-70), s16(9), s16(63), s16(52), s16(-18), s16(-22), s16(-2), s16(5), s16(24), s16(-2), s16(-35), s16(-85), s16(31), s16(96), s16(43), s16(-46), s16(-33), s16(-4)),
        (s16(-9), s16(-2), s16(-10), s16(-10), s16(-8), s16(-4), s16(-3), s16(-2), s16(1), s16(-5), s16(0), s16(-1), s16(7), s16(0), s16(-1), s16(-1), s16(2), s16(5), s16(0), s16(4), s16(3), s16(6), s16(-9), s16(-2), s16(-6), s16(-9), s16(-10), s16(-7), s16(-5), s16(-6), s16(-2), s16(-2)),  
        others => (others => s16(0))
    );

    -- 3. PASTE THE M, N, K CONSTANTS HERE
    constant M_VAL : integer := 22;
    constant N_VAL : integer := 32;
    constant K_VAL : integer := 23;
    file output_file : text open write_mode is "npu_output2.txt";
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

    clk <= not clk after clk_period / 2;

    process
    begin
        -- STEP 1: LOAD 2D MATRIX INTO 1D RAM SIGNALS
        -- This converts the matrix constants into the memory the NPU actually reads
        for i in 0 to M_VAL-1 loop
            for j in 0 to K_VAL-1 loop
                act_mem(i*K_VAL + j) <= std_logic_vector(resize(signed(DATA_STIM(i,j)), 32));
            end loop;
        end loop;
        for i in 0 to K_VAL-1 loop
            for j in 0 to N_VAL-1 loop
                weight_mem(i*N_VAL + j) <= std_logic_vector(resize(signed(WEIGHT_STIM(i,j)), 32));
            end loop;
        end loop;

        -- STEP 2: RESET
        reset_n <= '0';
        wait for 50 ns;
        reset_n <= '1';
        wait for 50 ns;

        -- STEP 3: CONFIGURE REGISTERS
        avs_address <= "00100"; avs_writedata <= std_logic_vector(to_unsigned(M_VAL, 32)); avs_write <= '1';
        wait until rising_edge(clk);
        avs_address <= "01000"; avs_writedata <= std_logic_vector(to_unsigned(N_VAL, 32)); avs_write <= '1';
        wait until rising_edge(clk);
        avs_address <= "01100"; avs_writedata <= std_logic_vector(to_unsigned(K_VAL, 32)); avs_write <= '1';
        wait until rising_edge(clk);
        avs_write <= '0'; 
        wait for 20 ns;

        -- STEP 4: START
        avs_address <= "00000"; avs_writedata <= x"00000001"; avs_write <= '1';
        wait until rising_edge(clk);
        avs_write <= '0';

        wait for 20000 ns; -- Long enough for 22x32 execution
        report "Simulation Complete.";
        wait;
    end process;

    -- STEP 5: RAM RESPONSE (Corrected Addressing)
    -- Your Avalon Masters will look for data at 0x2000 and 0x1000.
    -- We ignore the base address bits and just use the offset for act_mem/weight_mem.
    process(clk)
    begin
        if rising_edge(clk) then
            if avm_act_read = '1' then
                -- Mask out high bits (base address) to get local offset
                -- Divide by 4 because Avalon is byte-addressed
                avm_act_readdata <= act_mem((to_integer(unsigned(avm_act_address)) mod 4096) / 4);
            end if;
            if avm_weight_read = '1' then
                avm_weight_readdata <= weight_mem((to_integer(unsigned(avm_weight_address)) mod 4096) / 4);
            end if;
        end if;
    end process;

    -- STEP 6: CAPTURE OUTPUT WRITES
    process(clk)
        variable L        : line;
        variable addr_int : integer;
        variable row      : integer;
        variable col      : integer;
        variable data_int : integer;
    begin
        if rising_edge(clk) then
            if avm_out_write = '1' then

                addr_int := to_integer(unsigned(avm_out_address)) / 4;
                row := addr_int / 32;  -- use your N_VAL if dynamic
                col := addr_int mod 32;

                data_int := to_integer(signed(avm_out_writedata));

                -- Write element to file
                write(L, data_int);
                write(L, string'(" "));
                
                -- End line after each row
                if col = 31 then
                    writeline(output_file, L);
                end if;

            end if;
        end if;
    end process;
end architecture;