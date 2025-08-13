-- Project #43 (2025) - Top-level integration of Control Unit and Systolic Array
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.custom_types.all;

entity top_level_systolic_array is
    port (
        clk           : in  bit_1;
        reset         : in  bit_1;

        -- Inputs to feed matrices
        --matrix_data   : in  systolic_array_matrix_input;
        --matrix_weight : in  systolic_array_matrix_input;

        -- Outputs from the systolic array
        output        : out systolic_array_matrix_output;
        cycle_count   : out integer;

        -- Debug signals
        dbg_db_rdaddr     : out std_logic_vector(10 downto 0);
        dbg_db_data_out   : out std_logic_vector(7 downto 0);
        dbg_wb_rdaddr     : out std_logic_vector(10 downto 0);
        dbg_wb_data_out   : out std_logic_vector(7 downto 0);
        dbg_matrix_data   : out systolic_array_matrix_input;
        dbg_matrix_weight : out systolic_array_matrix_input
    );
end top_level_systolic_array;

architecture structure of top_level_systolic_array is

    -- Internal signals to connect control unit and systolic array
    signal data_shift_sig    : input_shift_matrix;
    signal weight_shift_sig  : input_shift_matrix;
    signal enabled_PE_mask   : enabled_PE_matrix;

    -- Data BRAM interface
    signal db_rdaddr    : std_logic_vector(10 downto 0);
    signal db_data_out  : std_logic_vector(7 downto 0);

    -- Weight BRAM interface
    signal wb_rdaddr    : std_logic_vector(10 downto 0);
    signal wb_data_out  : std_logic_vector(7 downto 0);

    -- matrices fed to control unit
    signal matrix_data_sig  : systolic_array_matrix_input := (others => (others => (others => '0')));
    signal matrix_weight_sig  : systolic_array_matrix_input := (others => (others => (others => '0')));

    --Indices for counting through row and column
    signal row, col : integer range 0 to N-1 := 0;
    signal loadingCompleteFlag : boolean := false;

    signal start_sig : bit_1 := '0';
begin

    DataBuffer : entity work.DataBuffer
        port map (
            clock		=> clk,
            data		=> (others => '0'),
            rdaddress   => db_rdaddr,
            wraddress   => (others => '0'),
            wren		=> '0',
            q		    => db_data_out
	);

    WeightBuffer : entity work.WeightBuffer
        port map (
            clock		=> clk,
            data		=> (others => '0'),
            rdaddress   => wb_rdaddr,
            wraddress   => (others => '0'),
            wren		=> '0',
            q		    => wb_data_out
	);

    -- Reading process
    process(clk)
    begin
        if rising_edge(clk) then
            if reset = '1' then
                db_rdaddr <= (others => '0');
                wb_rdaddr <= (others => '0');
                row       <= 0;
                col       <= 0;
                loadingCompleteFlag <= false;
                start_sig <= '0';

            elsif not loadingCompleteFlag then
                --Capture the BRAM outputs into matrix (its kinda relying on it being in the same cycle which it is not)
                matrix_data_sig(row, col) <= db_data_out;
                matrix_weight_sig(row, col) <= wb_data_out;

                -- End of Row
                if col = N-1 then
                    col <= 0;
                    if row = N-1 then
                        loadingCompleteFlag <= true; 
                        start_sig <= '1';
                    else
                        row <= row + 1;
                    end if;
                else
                    col <= col + 1;
                end if;

                -- Next BRAM address
                db_rdaddr <= std_logic_vector(to_unsigned(row * N + col + 1, db_rdaddr'length));
                wb_rdaddr <= std_logic_vector(to_unsigned(row * N + col + 1, wb_rdaddr'length));
            end if;
        end if;
    end process;

    -- Instantiate the Control Unit
    control_unit: entity work.control_unit
        port map (
            clk              => clk,
            reset            => reset,
            start            => start_sig,
            matrix_data      => matrix_data_sig,
            matrix_weight    => matrix_weight_sig,
            data_shift       => data_shift_sig,
            weight_shift     => weight_shift_sig,
            cycle_count      => cycle_count,
            PE_enabled_mask  => enabled_PE_mask
        );

    -- Instantiate the Systolic Array
    systolic_array: entity work.systolic_array
        port map (
            clk         => clk,
            reset       => reset,
            data_shift  => data_shift_sig,
            weight_shift=> weight_shift_sig,
            enabled_PE  => enabled_PE_mask,
            output      => output
        );

end architecture;