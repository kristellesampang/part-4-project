// (C) 2001-2024 Intel Corporation. All rights reserved.
// Your use of Intel Corporation's design tools, logic functions and other 
// software and tools, and its AMPP partner logic functions, and any output 
// files from any of the foregoing (including device programming or simulation 
// files), and any associated documentation or information are expressly subject 
// to the terms and conditions of the Intel Program License Subscription 
// Agreement, Intel FPGA IP License Agreement, or other applicable 
// license agreement, including, without limitation, that your use is for the 
// sole purpose of programming logic devices manufactured by Intel and sold by 
// Intel or its authorized distributors.  Please refer to the applicable 
// agreement for further details.


`timescale 1ns/1ns
module alt_pr_bitstream_compatibility_checker_ext_host(
	clk,
	nreset,
	freeze,
	crc_error,
	pr_error,
	pr_ready,
	pr_done,
	data,
	data_valid,
	data_ready,
	o_bitstream_incompatible
);
	parameter CDRATIO = 1;
	parameter CB_DATA_WIDTH = 16;
	parameter EXT_HOST_PRPOF_ID = 32'h12345678; //305419896
	
	localparam [1:0]	IDLE = 0,
						WAIT_FOR_READY = 1,
						CHECK_START = 2,
						CHECK_COMPLETE = 3;
	
	input clk; 
	input nreset;
	input freeze;
	input crc_error;
	input pr_error;
	input pr_ready;
	input pr_done;
	input data_valid;
	input data_ready;
	input [CB_DATA_WIDTH-1:0] data;
	
	output o_bitstream_incompatible;

	reg [1:0] data_count;
	reg [1:0] check_state;
	reg bitstream_incompatible_reg;
	
	assign o_bitstream_incompatible = bitstream_incompatible_reg;
	
	always @(posedge clk)
	begin
		if (~nreset) begin
			check_state <= IDLE;
		end
		else begin
			case (check_state)
				IDLE: 
				begin
					data_count <= 2'd0;
					bitstream_incompatible_reg <= 1'b0;
					
					if (freeze) begin
						check_state <= WAIT_FOR_READY;
					end	
				end
				
				WAIT_FOR_READY: 
				begin
					if (~freeze) begin
						check_state <= IDLE;
					end
					else if (pr_ready) begin
						// wait 3 clock cycles before sending 
						// actual PR data at 4th clock
						if (data_count == 2'd3) begin
							data_count <= 2'd0;
							check_state <= CHECK_START;
						end
						else begin
							data_count <= data_count + 2'd1;
						end
					end
				end
				
				CHECK_START: 
				begin
					if (~freeze || ~pr_ready || crc_error || pr_error || pr_done) begin
						check_state <= CHECK_COMPLETE;
					end
					else if (data_valid && data_ready) begin
						if (data_count != 2'd3) begin
							data_count <= data_count + 2'd1;
						end
						
						// For plain bitstream, the id stored in 3rd and 4th data words
						// 0	:	93b0;
						// 1	:	0001;
						// 2	:	xxxx; (id1)
						// 3	:	xxxx; (id2)
						if (CDRATIO == 1) begin
							if (data_count == 2'd2) begin
								if (data[15:0] != EXT_HOST_PRPOF_ID[15:0]) begin
									bitstream_incompatible_reg <= 1'b1;
									check_state <= CHECK_COMPLETE;
								end
							end
							else if (data_count == 2'd3) begin
								if (data[15:0] != EXT_HOST_PRPOF_ID[31:16]) begin
									bitstream_incompatible_reg <= 1'b1;
								end
								check_state <= CHECK_COMPLETE;
							end
						end
						
						// For compressed bitstream, the id stored in 3rd and 4th data words
						// To differentiate encrypted bitstream, check the 1st and 2nd to be 0's
						// 0	:	0000;
						// 1	:	0000;
						// 2	:	xxxx; (id1)
						// 3	:	xxxx; (id2)
						else if (CDRATIO == 4) begin
							if ((data_count == 2'd0) || (data_count == 2'd1)) begin
								if (data[15:0] != 16'd0) begin
									check_state <= CHECK_COMPLETE;
								end
							end
							else if (data_count == 2'd2) begin
								if (data[15:0] != EXT_HOST_PRPOF_ID[15:0]) begin
									bitstream_incompatible_reg <= 1'b1;
									check_state <= CHECK_COMPLETE;
								end
							end
							else if (data_count == 2'd3) begin
								if (data[15:0] != EXT_HOST_PRPOF_ID[31:16]) begin
									bitstream_incompatible_reg <= 1'b1;
								end
								check_state <= CHECK_COMPLETE;
							end
						end
						else if (CDRATIO == 2) begin
							// encrypted bitstream, skip the checking
							check_state <= CHECK_COMPLETE;
						end
					end
				end
				
				CHECK_COMPLETE: 
				begin
					if (~freeze) begin
						check_state <= IDLE;
					end
				end
				
				default: 
				begin
					check_state <= IDLE;
				end
			endcase
		end
	end
	
endmodule

`ifdef QUESTA_INTEL_OEM
`pragma questa_oem_00 "mXd8TZA7wcZD1M13kc63+RPfQcbqACFa9i0aqHtSQ62nFR2tUIuo2NzwpI/4cEyddIaPImJhX2CebOmYVV5f401JpALFknks4OMLrt70wgtKfnPjlusCxrHWEBHooOs6pkJXBTjC6oUhD2WVo05l5UBMB3Bob+qvd+BE/CfQQqbe8vYxUQkLM+Zgo0aA7Fpi0A6ZVpcWA/DDjPpZK8kPgz9Vfztbmx0x1RvFi9qEstKUBKEmJu6oS6UW3/cmRYI6IHw3UBVN4SkFdr2Wc7SqMBFVZ8U+Np1HQiw6XaBrVMgOWlfZXQ2cLhXtzbALd2SVxog00VFSKcjN8YhFGSdIMz34c5gxN4pqtdwXtmvW1I88QNdG6vM2alzj342BoFx2XVSndcovh+oTw8O2Cqg5B2CsYn5MR/O71cnoFJNEkAomaLdWw90TKJXoqqUERDluMicie2Ph42Uz4fjxcBuruHWNlb0feowEDG1+fcFH6+YHjiADUBLeUFP67AMvwxjVxqvehS+Z9KLU7NET6BfkmSx74piNyEew+VlC5hF+fNqdfYNKGvCYAf8yc0aMY1tka+///Cgj+iexKof4BDnLAaEyJJuJH9qHVkWK7Asy7vzsHCSpHvwszWOscxGUIytf9AJkO9PSQYRIJnbfDftM/hpdQeFIl5DOg3v+6kNg+zXajDi1g7JptjUkYZL9Qhox8g70+rqxyR1aSYi2uGJtKP7E8aPeHOuejXlSl+vak+AgTA6A2lDYfTe3bt+l3x8s5ZY0GEKcf8pKLwFh0qoI5/tNpk5TSJkKAOBUeceDKDgJsvYLRVwP2IP6pBm5KAV5qNsZqhK8izTQnW2kfHG/c0yw7IF7GyktYF/SQCCo7DGJkhqtx3x1omD3PEETtv6/Z9lui8RYxzNzcMQ/NEUHDWi3ZGZLrwEDmQ+AQ6xQaXe1nBTW7sbQzKF5+O8FQxOmlSTobKfBDWRN9AtYyuXQt/5ochp60UQkCuKov1zFT7LZS2Dl7Nwf+CAG+ThiUaxa"
`endif