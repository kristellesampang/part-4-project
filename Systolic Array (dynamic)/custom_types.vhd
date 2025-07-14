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

    -- matrix sizes 
    -- !! below are just testing values 
    type matrix_3x3 is array (0 to 2, 0 to 2) of bit_8; -- might need to change the size of the bit
    type matrix_3x3_output is array (0 to 2, 0 to 2) of bit_32; 
    -- to hold a row or column of elements 
    type matrix_1x3 is array (0 to 2) of bit_8; 
    type PE_en_3x3 is array (0 to 2, 0 to 2) of bit_1; -- or bit_1 if alias


	-- Dynamic 
	-- shift registers 
	type input_shift_matrix is array (0 to 2) of bit_8; 
	-- inter-PE signals (modify based on design)
	type data_bus_matrix is array(0 to 3, 0 to 3) of bit_8;     -- includes right boundary
	type weight_bus_matrix is array(0 to 3, 0 to 3) of bit_8;   -- includes bottom boundary
	type result_matrix is array(0 to 2, 0 to 2) of bit_32;


END custom_types;