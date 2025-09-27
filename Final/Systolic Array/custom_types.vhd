-- Custom types 
-- Project #43 (2025)

LIBRARY ieee;
USE ieee.std_logic_1164.ALL;
PACKAGE custom_types IS

    -- bit sizes (from ReCOP, Zoran Salcic)
	SUBTYPE bit_32 IS STD_LOGIC_VECTOR(31 DOWNTO 0);
	SUBTYPE bit_23 IS STD_LOGIC_VECTOR(22 DOWNTO 0);
	SUBTYPE bit_22 IS STD_LOGIC_VECTOR(21 DOWNTO 0);
	SUBTYPE bit_20 IS STD_LOGIC_VECTOR(19 DOWNTO 0);
	SUBTYPE bit_19 IS STD_LOGIC_VECTOR(18 DOWNTO 0); 
	SUBTYPE bit_18 IS STD_LOGIC_VECTOR(17 DOWNTO 0);
	SUBTYPE bit_17 IS STD_LOGIC_VECTOR(16 DOWNTO 0);
	SUBTYPE bit_16 IS STD_LOGIC_VECTOR(15 DOWNTO 0);
	SUBTYPE bit_13 IS STD_LOGIC_VECTOR(12 DOWNTO 0);
	SUBTYPE bit_12 IS STD_LOGIC_VECTOR(11 DOWNTO 0);
	SUBTYPE bit_11 IS STD_LOGIC_VECTOR(10 DOWNTO 0);
	SUBTYPE bit_10 IS STD_LOGIC_VECTOR(9 DOWNTO 0);
	SUBTYPE bit_9 IS STD_LOGIC_VECTOR(8 DOWNTO 0);
	SUBTYPE bit_8 IS STD_LOGIC_VECTOR(7 DOWNTO 0);
	SUBTYPE bit_7 IS STD_LOGIC_VECTOR(6 DOWNTO 0);
	SUBTYPE bit_6 IS STD_LOGIC_VECTOR(5 DOWNTO 0);
	SUBTYPE bit_5 IS STD_LOGIC_VECTOR(4 DOWNTO 0);
	SUBTYPE bit_4 IS STD_LOGIC_VECTOR(3 DOWNTO 0);
	SUBTYPE bit_3 IS STD_LOGIC_VECTOR(2 DOWNTO 0);
	SUBTYPE bit_2 IS STD_LOGIC_VECTOR(1 DOWNTO 0);
	SUBTYPE bit_1 IS STD_LOGIC;

	-- Dynamic Systolic Arrays 
	constant N : integer := 8; -- array dimension (only need to change this)

	-- input and output matrix
	type systolic_array_matrix_input is array (0 to N-1, 0 to N-1) of bit_8; -- must match the systolic array size
	type systolic_array_matrix_output is array (0 to N-1, 0 to N-1) of bit_32; -- must match the systolic array size
	-- shift registers 
	type input_shift_matrix is array (0 to N-1) of bit_8; -- 1xN size 
	-- PE enabled mask
	type enabled_PE_matrix is array (0 to N-1, 0 to N-1) of bit_1; -- 1 bit enable for all PEs
	-- inter-PE signals (modify based on design)
	type data_bus_matrix is array(0 to N, 0 to N) of bit_8; -- includes the bus going out of the right   
	type weight_bus_matrix is array(0 to N, 0 to N) of bit_8; -- includes the bus going out of the bottom
	type result_matrix is array(0 to N-1, 0 to N-1) of bit_32; -- holds the accumlated 8-bit value of each PE as a matrix


END custom_types;