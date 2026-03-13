	NonDPR_npu_system_19 u0 (
		.clk                 (_connected_to_clk_),                 //   input,   width = 1,          clock.clk
		.reset_n             (_connected_to_reset_n_),             //   input,   width = 1,          reset.reset_n
		.avs_address         (_connected_to_avs_address_),         //   input,   width = 5, avalon_slave_0.address
		.avs_write           (_connected_to_avs_write_),           //   input,   width = 1,               .write
		.avs_writedata       (_connected_to_avs_writedata_),       //   input,  width = 32,               .writedata
		.avs_read            (_connected_to_avs_read_),            //   input,   width = 1,               .read
		.avs_readdata        (_connected_to_avs_readdata_),        //  output,  width = 32,               .readdata
		.avs_waitrequest     (_connected_to_avs_waitrequest_),     //  output,   width = 1,               .waitrequest
		.avm_act_address     (_connected_to_avm_act_address_),     //  output,  width = 32,            act.address
		.avm_act_read        (_connected_to_avm_act_read_),        //  output,   width = 1,               .read
		.avm_act_readdata    (_connected_to_avm_act_readdata_),    //   input,  width = 32,               .readdata
		.avm_act_waitreq     (_connected_to_avm_act_waitreq_),     //   input,   width = 1,               .waitrequest
		.avm_weight_address  (_connected_to_avm_weight_address_),  //  output,  width = 32,         weight.address
		.avm_weight_read     (_connected_to_avm_weight_read_),     //  output,   width = 1,               .read
		.avm_weight_readdata (_connected_to_avm_weight_readdata_), //   input,  width = 32,               .readdata
		.avm_weight_waitreq  (_connected_to_avm_weight_waitreq_),  //   input,   width = 1,               .waitrequest
		.avm_out_address     (_connected_to_avm_out_address_),     //  output,  width = 32,            out.address
		.avm_out_write       (_connected_to_avm_out_write_),       //  output,   width = 1,               .write
		.avm_out_writedata   (_connected_to_avm_out_writedata_),   //  output,  width = 32,               .writedata
		.avm_out_waitreq     (_connected_to_avm_out_waitreq_)      //   input,   width = 1,               .waitrequest
	);

