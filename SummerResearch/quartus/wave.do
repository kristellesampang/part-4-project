onerror {resume}
quietly WaveActivateNextPane {} 0
add wave -noupdate /tb_npu_wrapper/clk
add wave -noupdate /tb_npu_wrapper/reset_n
add wave -noupdate /tb_npu_wrapper/avs_address
add wave -noupdate /tb_npu_wrapper/avs_write
add wave -noupdate /tb_npu_wrapper/avs_writedata
add wave -noupdate /tb_npu_wrapper/avs_waitrequest
add wave -noupdate /tb_npu_wrapper/avm_act_address
add wave -noupdate /tb_npu_wrapper/avm_act_read
add wave -noupdate /tb_npu_wrapper/avm_act_readdata
add wave -noupdate /tb_npu_wrapper/avm_act_waitreq
add wave -noupdate /tb_npu_wrapper/avm_weight_read
add wave -noupdate /tb_npu_wrapper/avm_weight_readdata
add wave -noupdate /tb_npu_wrapper/avm_weight_waitreq
add wave -noupdate /tb_npu_wrapper/avm_out_waitreq
add wave -noupdate /tb_npu_wrapper/clk_period
add wave -noupdate /tb_npu_wrapper/DUT/reg_ready
add wave -noupdate /tb_npu_wrapper/DUT/reg_m
add wave -noupdate /tb_npu_wrapper/DUT/reg_n
add wave -noupdate /tb_npu_wrapper/DUT/reg_k
add wave -noupdate /tb_npu_wrapper/DUT/n_cycle_count
add wave -noupdate -radix decimal /tb_npu_wrapper/DUT/n_output
add wave -noupdate /tb_npu_wrapper/DUT/n_done
add wave -noupdate -radix decimal /tb_npu_wrapper/DUT/n_matrix_data
add wave -noupdate -radix decimal /tb_npu_wrapper/DUT/n_matrix_weight
add wave -noupdate /tb_npu_wrapper/DUT/fetch_counter
add wave -noupdate /tb_npu_wrapper/DUT/write_counter
add wave -noupdate /tb_npu_wrapper/DUT/row_idx
add wave -noupdate /tb_npu_wrapper/DUT/col_idx
add wave -noupdate /tb_npu_wrapper/DUT/w_col_idx
add wave -noupdate /tb_npu_wrapper/DUT/sa_start_trigger
add wave -noupdate /tb_npu_wrapper/DUT/state
add wave -noupdate /tb_npu_wrapper/DUT/NPU_CORE/systolic_array/output
add wave -noupdate /tb_npu_wrapper/DUT/NPU_CORE/systolic_array/data_bus
add wave -noupdate /tb_npu_wrapper/DUT/NPU_CORE/systolic_array/weight_bus
add wave -noupdate /tb_npu_wrapper/DUT/NPU_CORE/systolic_array/results
TreeUpdate [SetDefaultTree]
WaveRestoreCursors {{Cursor 1} {465 ns} 0}
quietly wave cursor active 1
configure wave -namecolwidth 233
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
configure wave -timelineunits ns
update
WaveRestoreZoom {425 ns} {604 ns}
