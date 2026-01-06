module mSGDMA_PR (
		input  wire         clock_clk,                    //            clock.clk,          Clock Input
		input  wire         reset_n_reset_n,              //          reset_n.reset_n,      Reset Input
		input  wire [31:0]  csr_writedata,                //              csr.writedata
		input  wire         csr_write,                    //                 .write
		input  wire [3:0]   csr_byteenable,               //                 .byteenable
		output wire [31:0]  csr_readdata,                 //                 .readdata
		input  wire         csr_read,                     //                 .read
		input  wire [2:0]   csr_address,                  //                 .address
		input  wire         descriptor_slave_write,       // descriptor_slave.write
		output wire         descriptor_slave_waitrequest, //                 .waitrequest
		input  wire [127:0] descriptor_slave_writedata,   //                 .writedata
		input  wire [15:0]  descriptor_slave_byteenable,  //                 .byteenable
		output wire         csr_irq_irq,                  //          csr_irq.irq
		output wire [31:0]  mm_read_address,              //          mm_read.address
		output wire         mm_read_read,                 //                 .read
		output wire [3:0]   mm_read_byteenable,           //                 .byteenable
		input  wire [31:0]  mm_read_readdata,             //                 .readdata
		input  wire         mm_read_waitrequest,          //                 .waitrequest
		input  wire         mm_read_readdatavalid,        //                 .readdatavalid
		output wire [1:0]   mm_read_burstcount,           //                 .burstcount
		output wire [31:0]  mm_write_address,             //         mm_write.address
		output wire         mm_write_write,               //                 .write
		output wire [3:0]   mm_write_byteenable,          //                 .byteenable
		output wire [31:0]  mm_write_writedata,           //                 .writedata
		input  wire         mm_write_waitrequest,         //                 .waitrequest
		output wire [1:0]   mm_write_burstcount           //                 .burstcount
	);
endmodule

