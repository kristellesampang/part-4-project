library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.custom_types.all;

entity NPU_Wrapper is
    port (
        -- Physical Board pin connections 
        CLOCK_50 : in  std_logic;
    );
end entity NPU_Wrapper;

architecture NPU of NPU_Wrapper is

    -- Component declaration for complete NPU system
    component top_level_systolic_array is
        port (
            clk           : in  std_logic;
            reset         : in  std_logic;
            matrix_data   : in  systolic_array_matrix_input;
            matrix_weight : in  systolic_array_matrix_input;
            active_rows   : in  integer;
            active_cols   : in  integer;
            output        : out systolic_array_matrix_output;
            cycle_count   : out integer
        );
    end component top_level_systolic_array;


begin
    -- Instantiate the NPU 
    npu_core_inst : component top_level_systolic_array
        port map (
            clk           => CLOCK_50,
            reset         => ??,
            matrix_data   => ??,
            matrix_weight => ??,
            active_rows   => 8, 
            active_cols   => 8,
            output        => ??,
            cycle_count   => ??
        );

end architecture NPU;
