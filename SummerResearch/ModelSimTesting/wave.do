onerror {resume}
quietly WaveActivateNextPane {} 0
add wave -noupdate /tb_npu_memory_system/clk_tb
add wave -noupdate /tb_npu_memory_system/reset_tb
add wave -noupdate /tb_npu_memory_system/start_transfer_cmd_tb
add wave -noupdate /tb_npu_memory_system/npu_data_read_tb
add wave -noupdate /tb_npu_memory_system/npu_weight_read_tb
add wave -noupdate /tb_npu_memory_system/npu_read_sig
add wave -noupdate /tb_npu_memory_system/sa_output_sig
add wave -noupdate /tb_npu_memory_system/sa_cycle_count
add wave -noupdate /tb_npu_memory_system/CLK_PERIOD
add wave -noupdate /tb_npu_memory_system/DUT_Controller/ping_pong_sel
add wave -noupdate /tb_npu_memory_system/DUT_Controller/npu_addr
add wave -noupdate /tb_npu_memory_system/DUT_Controller/dma_addr
add wave -noupdate /tb_npu_memory_system/DUT_Controller/npu_addr_int
add wave -noupdate /tb_npu_memory_system/DUT_Controller/dma_addr_int
add wave -noupdate /tb_npu_memory_system/DUT_Controller/ping_a_q
add wave -noupdate /tb_npu_memory_system/DUT_Controller/pong_a_q
add wave -noupdate /tb_npu_memory_system/DUT_Controller/ping_b_q
add wave -noupdate /tb_npu_memory_system/DUT_Controller/pong_b_q
add wave -noupdate /tb_npu_memory_system/DUT_Controller/output_c_q
add wave -noupdate /tb_npu_memory_system/DUT_Controller/dma_write_data_16
add wave -noupdate /tb_npu_memory_system/DUT_Controller/dma_write_data_64
add wave -noupdate /tb_npu_memory_system/DUT_Controller/ping_a_wren
add wave -noupdate /tb_npu_memory_system/DUT_Controller/pong_a_wren
add wave -noupdate /tb_npu_memory_system/DUT_Controller/ping_b_wren
add wave -noupdate /tb_npu_memory_system/DUT_Controller/pong_b_wren
add wave -noupdate /tb_npu_memory_system/DUT_Controller/output_c_wren
add wave -noupdate /tb_npu_memory_system/DUT_Controller/unused_std_logic
add wave -noupdate /tb_npu_memory_system/DUT_Controller/unused_std_logic_vector_16
add wave -noupdate /tb_npu_memory_system/DUT_Controller/unused_std_logic_vector_64
add wave -noupdate /tb_npu_memory_system/DUT_Controller/current_state
add wave -noupdate /tb_npu_memory_system/NPU_Core/clk
add wave -noupdate /tb_npu_memory_system/NPU_Core/reset
add wave -noupdate /tb_npu_memory_system/NPU_Core/ready
add wave -noupdate /tb_npu_memory_system/NPU_Core/matrix_data
add wave -noupdate /tb_npu_memory_system/NPU_Core/matrix_weight
add wave -noupdate /tb_npu_memory_system/NPU_Core/active_rows
add wave -noupdate /tb_npu_memory_system/NPU_Core/active_cols
add wave -noupdate /tb_npu_memory_system/NPU_Core/active_k
add wave -noupdate /tb_npu_memory_system/NPU_Core/output
add wave -noupdate /tb_npu_memory_system/NPU_Core/cycle_count
add wave -noupdate /tb_npu_memory_system/NPU_Core/data_shift_sig
add wave -noupdate /tb_npu_memory_system/NPU_Core/weight_shift_sig
add wave -noupdate /tb_npu_memory_system/NPU_Core/enabled_PE_mask
TreeUpdate [SetDefaultTree]
WaveRestoreCursors {{Cursor 1} {2583564 ps} 0}
quietly wave cursor active 1
configure wave -namecolwidth 149
configure wave -valuecolwidth 100
configure wave -justifyvalue left
configure wave -signalnamewidth 0
configure wave -snapdistance 10
configure wave -datasetprefix 0
configure wave -rowmargin 4
configure wave -childrowmargin 2
configure wave -gridoffset 0
configure wave -gridperiod 1
configure wave -griddelta 40
configure wave -timeline 0
configure wave -timelineunits ps
update
WaveRestoreZoom {0 ps} {16639382 ps}
