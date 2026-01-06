module PR_Test_Top_alt_pr_1 (
		input  wire        clk,                    //        clk.clk
		input  wire        nreset,                 //     nreset.reset_n
		output wire        freeze,                 //     freeze.freeze
		input  wire [3:0]  avmm_slave_address,     // avmm_slave.address
		input  wire        avmm_slave_read,        //           .read
		input  wire [31:0] avmm_slave_writedata,   //           .writedata
		input  wire        avmm_slave_write,       //           .write
		output wire [31:0] avmm_slave_readdata,    //           .readdata
		output wire        avmm_slave_waitrequest, //           .waitrequest
		output wire        irq                     //  interrupt.irq
	);
endmodule

