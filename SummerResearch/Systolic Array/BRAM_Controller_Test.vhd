library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.custom_types.all; -- Contains definition for bit_16 type

entity BRAM_Controller_Test is
    port (
        clk_in  : in  std_logic;
        reset_in : in  std_logic;
        start_transfer_cmd_in : in  std_logic;

        dma_write_data_16 : in bit_16;
        dma_write_data_64 : in std_logic_vector(63 downto 0);

        npu_data_read : out bit_16;
        npu_weight_read : out bit_16;

        current_buffer_is_ping : out std_logic;

        current_state_out : out state_t
    );
end entity BRAM_Controller_Test;

architecture rtl of BRAM_Controller_Test is

    constant N_ADDR_BITS : integer := 10;
    
    -- Control and State Signals (All declarations must come before functions/begin)
    signal ping_pong_sel : std_logic := '0';
    signal npu_addr : std_logic_vector(N_ADDR_BITS-1 downto 0);
    signal dma_addr : std_logic_vector(N_ADDR_BITS-1 downto 0);
    signal npu_addr_int : integer range 0 to 1023 := 0;
    signal dma_addr_int : integer range 0 to 1023 := 0;

    -- Data Signals from BRAM outputs (NOW std_logic_vector to fix synthesis port error)
    signal ping_a_q : std_logic_vector(15 downto 0);
    signal pong_a_q : std_logic_vector(15 downto 0);
    signal ping_b_q : std_logic_vector(15 downto 0);
    signal pong_b_q : std_logic_vector(15 downto 0);
    signal output_c_q : std_logic_vector(63 downto 0);

    -- DMA Write Data (Mimic for testbench input)
    --signal dma_write_data_16 : bit_16 := (others => '0');
    --signal dma_write_data_64 : std_logic_vector(63 downto 0) := (others => '0');

    -- Control Signals
    signal ping_a_wren, pong_a_wren : std_logic := '0';
    signal ping_b_wren, pong_b_wren : std_logic := '0';
    signal output_c_wren : std_logic := '0';
    
    -- Shared utility signals
    signal unused_std_logic : std_logic := '0';
    signal unused_std_logic_vector_16 : std_logic_vector(15 downto 0) := (others => '0');
    signal unused_std_logic_vector_64 : std_logic_vector(63 downto 0) := (others => '0');


    --type state_t is (S_IDLE, S_PING_COMPUTE, S_SWITCH, S_PONG_COMPUTE);
    signal current_state : state_t := S_IDLE;


    --  COMPONENT DECLARATIONS (Required for Quartus Synthesis) 
    -- 16-bit Input Buffer Component
    component DataBuffer1 is
        port (
            data_a    : in  std_logic_vector(15 downto 0);
            q_a       : out std_logic_vector(15 downto 0);
            data_b    : in  std_logic_vector(15 downto 0);
            q_b       : out std_logic_vector(15 downto 0);
            address_a : in  std_logic_vector(9 downto 0);
            address_b : in  std_logic_vector(9 downto 0);
            wren_a    : in  std_logic;
            wren_b    : in  std_logic;
            clock     : in  std_logic;
            freeze    : in  std_logic;
            enable    : in  std_logic;
            aclr      : in  std_logic
        );
    end component DataBuffer1;

    component DataBuffer2 is
        port (
            data_a    : in  std_logic_vector(15 downto 0);
            q_a       : out std_logic_vector(15 downto 0);
            data_b    : in  std_logic_vector(15 downto 0);
            q_b       : out std_logic_vector(15 downto 0);
            address_a : in  std_logic_vector(9 downto 0);
            address_b : in  std_logic_vector(9 downto 0);
            wren_a    : in  std_logic;
            wren_b    : in  std_logic;
            clock     : in  std_logic;
            freeze    : in  std_logic;
            enable    : in  std_logic;
            aclr      : in  std_logic
        );
    end component DataBuffer2;

    component WeightBuffer1 is
        port (
            data_a    : in  std_logic_vector(15 downto 0);
            q_a       : out std_logic_vector(15 downto 0);
            data_b    : in  std_logic_vector(15 downto 0);
            q_b       : out std_logic_vector(15 downto 0);
            address_a : in  std_logic_vector(9 downto 0);
            address_b : in  std_logic_vector(9 downto 0);
            wren_a    : in  std_logic;
            wren_b    : in  std_logic;
            clock     : in  std_logic;
            freeze    : in  std_logic;
            enable    : in  std_logic;
            aclr      : in  std_logic
        );
    end component WeightBuffer1;

    component WeightBuffer2 is
        port (
            data_a    : in  std_logic_vector(15 downto 0);
            q_a       : out std_logic_vector(15 downto 0);
            data_b    : in  std_logic_vector(15 downto 0);
            q_b       : out std_logic_vector(15 downto 0);
            address_a : in  std_logic_vector(9 downto 0);
            address_b : in  std_logic_vector(9 downto 0);
            wren_a    : in  std_logic;
            wren_b    : in  std_logic;
            clock     : in  std_logic;
            freeze    : in  std_logic;
            enable    : in  std_logic;
            aclr      : in  std_logic
        );
    end component WeightBuffer2;
    
    -- 64-bit Output Buffer Component
    component OutputBuffer is
        port (
            data_a    : in  std_logic_vector(63 downto 0);
            q_a       : out std_logic_vector(63 downto 0);
            data_b    : in  std_logic_vector(63 downto 0);
            q_b       : out std_logic_vector(63 downto 0);
            address_a : in  std_logic_vector(9 downto 0);
            address_b : in  std_logic_vector(9 downto 0);
            wren_a    : in  std_logic;
            wren_b    : in  std_logic;
            clock     : in  std_logic;
            freeze    : in  std_logic;
            enable    : in  std_logic;
            aclr      : in  std_logic
        );
    end component OutputBuffer;

    -- CUSTOM TYPE CONVERSION FUNCTIONS
    
    -- Utility function 1: Converts std_logic to bit_1 (FIXED SYNTAX)
    function to_bit (S : std_logic) return bit_1 is
        variable V : bit_1; -- Local variable for sequential assignment
    begin
        if S = '1' then
            V := '1';
        else
            V := '0';
        end if;
        return V;
    end function to_bit;

    -- Utility function 2: Converts std_logic_vector(15 downto 0) to custom bit_16
    function to_bit_16 (S : std_logic_vector(15 downto 0)) return bit_16 is
        variable V : bit_16;
    begin
        for I in 0 to 15 loop
            V(I) := to_bit(S(I));
        end loop;
        return V;
    end function to_bit_16;


begin -- Start of Concurrent Statements
    -- Debug Output
    current_buffer_is_ping <= not ping_pong_sel;

    current_state_out <= current_state;
    
    -- Address conversion (used concurrently)
    npu_addr <= std_logic_vector(to_unsigned(npu_addr_int, npu_addr'length));
    dma_addr <= std_logic_vector(to_unsigned(dma_addr_int, dma_addr'length));
    
    -- NPU data read multiplexer (Concurrent) - USES LOCAL CONVERSION FUNCTION
    npu_data_read <= to_bit_16(ping_a_q) when ping_pong_sel = '0' else to_bit_16(pong_a_q);
    npu_weight_read <= to_bit_16(ping_b_q) when ping_pong_sel = '0' else to_bit_16(pong_b_q);

    -- DMA Write Enable Control (Concurrent)
    ping_a_wren <= '1' when current_state = S_PONG_COMPUTE else '0'; 
    pong_a_wren <= '1' when current_state = S_PING_COMPUTE else '0'; 
    ping_b_wren <= '1' when current_state = S_PONG_COMPUTE else '0';
    pong_b_wren <= '1' when current_state = S_PING_COMPUTE else '0';
    output_c_wren <= '1' when current_state /= S_IDLE else '0'; 
    
    -- *** COMPONENT INSTANTIATION (5 instances) ***
    
    -- 1. DATA A PING (16-bit)
    BRAM_A_PING_INST : component DataBuffer1 
        port map (
            data_a    => unused_std_logic_vector_16,
            q_a       => ping_a_q,
            data_b    => std_logic_vector(dma_write_data_16),
            q_b       => unused_std_logic_vector_16,
            address_a => npu_addr,
            address_b => dma_addr,
            wren_a    => '0',
            wren_b    => pong_a_wren,
            clock     => clk_in,
            freeze    => unused_std_logic,
            enable    => '1',
            aclr      => reset_in
        );
    
    -- 2. DATA A PONG (16-bit)
    BRAM_A_PONG_INST : component DataBuffer2 
        port map (
            data_a    => unused_std_logic_vector_16,
            q_a       => pong_a_q,
            data_b    => std_logic_vector(dma_write_data_16),
            q_b       => unused_std_logic_vector_16,
            address_a => npu_addr,
            address_b => dma_addr,
            wren_a    => '0',
            wren_b    => ping_a_wren,
            clock     => clk_in,
            freeze    => unused_std_logic,
            enable    => '1',
            aclr      => reset_in
        );
        
    -- 3. WEIGHT B PING (16-bit)
    BRAM_B_PING_INST : component WeightBuffer1 
        port map (
            data_a    => unused_std_logic_vector_16,
            q_a       => ping_b_q,
            data_b    => std_logic_vector(dma_write_data_16),
            q_b       => unused_std_logic_vector_16,
            address_a => npu_addr,
            address_b => dma_addr,
            wren_a    => '0',
            wren_b    => pong_b_wren,
            clock     => clk_in,
            freeze    => unused_std_logic,
            enable    => '1',
            aclr      => reset_in
        );
        
    -- 4. WEIGHT B PONG (16-bit)
    BRAM_B_PONG_INST : component WeightBuffer2
        port map (
            data_a    => unused_std_logic_vector_16,
            q_a       => pong_b_q,
            data_b    => std_logic_vector(dma_write_data_16),
            q_b       => unused_std_logic_vector_16,
            address_a => npu_addr,
            address_b => dma_addr,
            wren_a    => '0',
            wren_b    => ping_b_wren,
            clock     => clk_in,
            freeze    => unused_std_logic,
            enable    => '1',
            aclr      => reset_in
        );

    -- 5. OUTPUT C (64-bit)
    BRAM_C_OUT_INST : component OutputBuffer 
        port map (
            data_a    => unused_std_logic_vector_64,
            q_a       => output_c_q,
            data_b    => dma_write_data_64,
            q_b       => unused_std_logic_vector_64,
            address_a => dma_addr,
            address_b => npu_addr,
            wren_a    => '0',
            wren_b    => output_c_wren,
            clock     => clk_in,
            freeze    => unused_std_logic,
            enable    => '1',
            aclr      => reset_in
        );
    -- *** FSM and Address Generation (SEQUENTIAL LOGIC) ***

    process(clk_in, reset_in)
    begin
        if reset_in = '1' then
            current_state <= S_IDLE;
            ping_pong_sel <= '0';
            npu_addr_int  <= 0;
            dma_addr_int  <= 0;
        elsif rising_edge(clk_in) then
            
            -- DMA Address Generation 
            dma_addr_int <= (dma_addr_int + 1) mod 1024;
            
            -- NPU Address Generation 
            if current_state = S_PING_COMPUTE or current_state = S_PONG_COMPUTE then
                npu_addr_int <= (npu_addr_int + 1) mod 1024;
            end if;

            case current_state is
                when S_IDLE =>
                    if start_transfer_cmd_in = '1' then 
                        current_state <= S_PING_COMPUTE;
                    end if;

                when S_PING_COMPUTE =>
                    if npu_addr_int = 1023 then 
                        current_state <= S_SWITCH; 
                    end if;

                when S_SWITCH =>
                    if start_transfer_cmd_in = '1' then 
                        ping_pong_sel <= not ping_pong_sel; 
                        
                        if ping_pong_sel = '0' then 
                            current_state <= S_PONG_COMPUTE; 
                        else 
                            current_state <= S_PING_COMPUTE; 
                        end if;
                        
                        npu_addr_int <= 0;
                    end if;

                when S_PONG_COMPUTE =>
                    if npu_addr_int = 1023 then 
                        current_state <= S_SWITCH;
                    end if;
            end case;
        end if;
    end process;

end architecture rtl;