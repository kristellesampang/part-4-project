-- (C) 2001-2024 Intel Corporation. All rights reserved.
-- Your use of Intel Corporation's design tools, logic functions and other 
-- software and tools, and its AMPP partner logic functions, and any output 
-- files from any of the foregoing (including device programming or simulation 
-- files), and any associated documentation or information are expressly subject 
-- to the terms and conditions of the Intel Program License Subscription 
-- Agreement, Intel FPGA IP License Agreement, or other applicable 
-- license agreement, including, without limitation, that your use is for the 
-- sole purpose of programming logic devices manufactured by Intel and sold by 
-- Intel or its authorized distributors.  Please refer to the applicable 
-- agreement for further details.




LIBRARY ieee;
USE ieee.std_logic_1164.all;

LIBRARY altera_lnsim;
USE altera_lnsim.altera_lnsim_components.all;

ENTITY DataBuffer1_ram_2port_2041_tqhxgji IS
    PORT
    (
        aclr       : IN STD_LOGIC  := '0';
        address_a       : IN STD_LOGIC_VECTOR (9 DOWNTO 0);
        address_b       : IN STD_LOGIC_VECTOR (9 DOWNTO 0);
        clock       : IN STD_LOGIC  := '1';
        data_a       : IN STD_LOGIC_VECTOR (15 DOWNTO 0);
        data_b       : IN STD_LOGIC_VECTOR (15 DOWNTO 0);
        enable       : IN STD_LOGIC  := '1';
        freeze       : IN STD_LOGIC  := '0';
        wren_a       : IN STD_LOGIC  := '0';
        wren_b       : IN STD_LOGIC  := '0';
        q_a       : OUT STD_LOGIC_VECTOR (15 DOWNTO 0);
        q_b       : OUT STD_LOGIC_VECTOR (15 DOWNTO 0)
    );
END DataBuffer1_ram_2port_2041_tqhxgji;


ARCHITECTURE SYN OF DataBuffer1_ram_2port_2041_tqhxgji IS

    SIGNAL sub_wire0    : STD_LOGIC_VECTOR (15 DOWNTO 0);
    SIGNAL sub_wire1    : STD_LOGIC_VECTOR (15 DOWNTO 0);

BEGIN
    q_a     <= sub_wire0 (15 DOWNTO 0);
    q_b     <= sub_wire1 (15 DOWNTO 0);

    altera_syncram_component : altera_syncram
    GENERIC MAP (
            address_reg_b  => "CLOCK0",
            clock_enable_input_a  => "NORMAL",
            clock_enable_input_b  => "NORMAL",
            clock_enable_output_a  => "NORMAL",
            clock_enable_output_b  => "NORMAL",
            indata_reg_b  => "CLOCK0",
            init_file  => "/home/pratham/Documents/Github/part-4-project/SummerResearch/Memory/Buffers/MIFs/DataBuffer1.mif",
            enable_force_to_zero  => "FALSE",
            intended_device_family  => "Arria 10",
            lpm_type  => "altera_syncram",
            maximum_depth  => 1024,
            numwords_a  => 1024,
            numwords_b  => 1024,
            operation_mode  => "BIDIR_DUAL_PORT",
            outdata_aclr_a  => "CLEAR0",
            outdata_sclr_a  => "NONE",
            outdata_aclr_b  => "CLEAR0",
            outdata_sclr_b  => "NONE",
            outdata_reg_a  => "CLOCK0",
            outdata_reg_b  => "CLOCK0",
            power_up_uninitialized  => "FALSE",
            ram_block_type  => "M20K",
            read_during_write_mode_mixed_ports  => "DONT_CARE",
            read_during_write_mode_port_a  => "NEW_DATA_NO_NBE_READ",
            read_during_write_mode_port_b  => "NEW_DATA_NO_NBE_READ",
            widthad_a  => 10,
            widthad_b  => 10,
            width_a  => 16,
            width_b  => 16,
            width_byteena_a  => 1,
            width_byteena_b  => 1
    )
    PORT MAP (
        aclr0 => aclr,
        address_a => address_a,
        address_b => address_b,
        clock0 => clock,
        clocken0 => enable,
        data_a => data_a,
        data_b => data_b,
        wren_a => wren_a,
        wren_b => wren_b,
        q_a => sub_wire0,
        q_b => sub_wire1
    );



END SYN;

