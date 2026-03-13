module NonDPR_npu_system_14 (
		input  wire        clk,                 //          clock.clk
		input  wire        reset_n,             //          reset.reset_n
		input  wire [4:0]  avs_address,         // avalon_slave_0.address
		input  wire        avs_write,           //               .write
		input  wire [31:0] avs_writedata,       //               .writedata
		input  wire        avs_read,            //               .read
		output wire [31:0] avs_readdata,        //               .readdata
		output wire        avs_waitrequest,     //               .waitrequest
		output wire [31:0] avm_act_address,     //            act.address
		output wire        avm_act_read,        //               .read
		input  wire [31:0] avm_act_readdata,    //               .readdata
		input  wire        avm_act_waitreq,     //               .waitrequest
		output wire [31:0] avm_weight_address,  //         weight.address
		output wire        avm_weight_read,     //               .read
		input  wire [31:0] avm_weight_readdata, //               .readdata
		input  wire        avm_weight_waitreq,  //               .waitrequest
		output wire [31:0] avm_out_address,     //            out.address
		output wire        avm_out_write,       //               .write
		output wire [31:0] avm_out_writedata,   //               .writedata
		input  wire        avm_out_waitreq      //               .waitrequest
	);
endmodule

