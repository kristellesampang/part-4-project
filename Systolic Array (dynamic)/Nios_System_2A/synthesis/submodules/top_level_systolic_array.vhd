library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.custom_types.all;

entity top_level_systolic_array is
    port (
        -- === System Signals ===
        clk   : in  std_logic;
        reset : in  std_logic;

        -- === Avalon Memory-Mapped SLAVE Interface (for NIOS control) ===
        avs_control_address   : in  std_logic_vector(2 downto 0);
        avs_control_write     : in  std_logic;
        avs_control_writedata : in  std_logic_vector(31 downto 0);
        avs_control_read      : in  std_logic;
        avs_control_readdata  : out std_logic_vector(31 downto 0);

        -- === Avalon Memory-Mapped MASTER Interface (to read from DATA BRAM) ===
        avm_data_address      : out std_logic_vector(31 downto 0);
        avm_data_read         : out std_logic;
        avm_data_readdata     : in  std_logic_vector(31 downto 0); -- Reads 32-bit packed data
        avm_data_waitrequest  : in  std_logic;

        -- Avalon Memory-Mapped MASTER Interface (to read from WEIGHT BRAM) 
        avm_weight_address      : out std_logic_vector(31 downto 0);
        avm_weight_read         : out std_logic;
        avm_weight_readdata     : in  std_logic_vector(31 downto 0); -- Reads 32-bit packed data
        avm_weight_waitrequest  : in  std_logic
    );
end entity top_level_systolic_array;

architecture structure of top_level_systolic_array is

    -- Internal signals for control registers, driven by the NIOS via the Avalon Slave
    signal r_active_rows : integer range 0 to N := 0;
    signal r_active_cols : integer range 0 to N := 0;
    signal r_start       : std_logic := '0';
    signal r_done        : std_logic := '1';

    -- Internal signals for the matrices, now filled by a state machine
    signal w_matrix_data   : systolic_array_matrix_input;
    signal w_matrix_weight : systolic_array_matrix_input;
    
    -- Internal signals connecting CU and Systolic Array
    signal w_data_shift   : input_shift_matrix;
    signal w_weight_shift : input_shift_matrix;
    signal w_pe_mask      : enabled_PE_matrix;
    signal w_output       : systolic_array_matrix_output;
    signal w_cycle_count  : integer;

    -- NOTE: A full implementation would have a state machine here to drive the
    -- Avalon Master ports (avm_data_*, avm_weight_*) to read from the BRAMs
    -- and fill the w_matrix_data/weight signals before starting the CU.
    -- For now, we will tie them off to allow the component to be created.

begin

    process(clk, reset)
    begin
        if reset = '1' then
            r_active_rows   <= 0;
            r_active_cols   <= 0;
            r_start         <= '0';
            r_done          <= '1';
            avs_control_readdata <= x"00000000";
        elsif rising_edge(clk) then
            r_start <= '0';

            if avs_control_write = '1' then
                case avs_control_address is
                    when "000" => r_active_rows <= to_integer(unsigned(avs_control_writedata(3 downto 0)));
                    when "001" => r_active_cols <= to_integer(unsigned(avs_control_writedata(3 downto 0)));
                    when "010" =>
                        r_start <= '1';
                        r_done  <= '0';
                    when others => null;
                end case;
            end if;

            if avs_control_read = '1' then
                if avs_control_address = "011" then
                    avs_control_readdata <= (31 downto 1 => '0') & r_done;
                else
                    avs_control_readdata <= x"00000000";
                end if;
            else
                avs_control_readdata <= (others => 'Z');
            end if;
        end if;
    end process;
    
    -- Tie off Master ports for now. A state machine will drive these later.
    avm_data_read <= '0';
    avm_data_address <= (others => '0');
    avm_weight_read <= '0';
    avm_weight_address <= (others => '0');
    
    -- Tie off internal matrices for now.
    gen_matrix_tie_offs : for i in 0 to N-1 generate
        gen_row_tie_offs : for j in 0 to N-1 generate
            w_matrix_data(i,j)   <= (others => '0');
            w_matrix_weight(i,j) <= (others => '0');
        end generate gen_row_tie_offs;
    end generate gen_matrix_tie_offs;
    
    -- Instantiate the internal Control Unit
    cu_inst : entity work.control_unit
        port map (
            clk             => clk,
            reset           => r_start, -- Use the start pulse to reset/start the CU
            matrix_data     => w_matrix_data,
            matrix_weight   => w_matrix_weight,
            active_rows     => r_active_rows,
            active_cols     => r_active_cols,
            data_shift      => w_data_shift,
            weight_shift    => w_weight_shift,
            cycle_count     => w_cycle_count,
            PE_enabled_mask => w_pe_mask
        );

    -- Instantiate the internal Systolic Array
    sa_inst : entity work.systolic_array
        port map (
            clk          => clk,
            reset        => r_start, -- Use the start pulse to reset/start the SA
            data_shift   => w_data_shift,
            weight_shift => w_weight_shift,
            enabled_PE   => w_pe_mask,
            output       => w_output
        );

end architecture structure;

