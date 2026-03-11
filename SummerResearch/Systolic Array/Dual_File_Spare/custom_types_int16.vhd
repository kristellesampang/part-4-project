-- Custom types INT16 -- Project #43 (2025)
LIBRARY ieee;
USE ieee.std_logic_1164.ALL;

PACKAGE custom_types_int16 IS

    SUBTYPE bit_64 IS STD_LOGIC_VECTOR(63 DOWNTO 0);
    SUBTYPE bit_32 IS STD_LOGIC_VECTOR(31 DOWNTO 0);
    SUBTYPE bit_16 IS STD_LOGIC_VECTOR(15 DOWNTO 0);
    SUBTYPE bit_8  IS STD_LOGIC_VECTOR(7 DOWNTO 0);
    SUBTYPE bit_1  IS STD_LOGIC;

    constant N16 : integer := 16;

    type systolic_array_matrix_input_int16  is array (0 to N16-1, 0 to N16-1) of bit_16;
    type systolic_array_matrix_output_int16 is array (0 to N16-1, 0 to N16-1) of bit_32;
    type input_shift_matrix_int16           is array (0 to N16-1) of bit_16;
    type enabled_PE_matrix_int16            is array (0 to N16-1, 0 to N16-1) of bit_1;
    type data_bus_matrix_int16              is array (0 to N16, 0 to N16) of bit_16;
    type weight_bus_matrix_int16            is array (0 to N16, 0 to N16) of bit_16;
    type result_matrix_int16                is array (0 to N16-1, 0 to N16-1) of bit_32;

END custom_types_int16;