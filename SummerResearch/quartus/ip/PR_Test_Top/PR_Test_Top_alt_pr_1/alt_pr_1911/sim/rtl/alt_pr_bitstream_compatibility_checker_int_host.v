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
module alt_pr_bitstream_compatibility_checker_int_host(
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
	(* preserve, altera_attribute = "-name PRPOF_ID on; -name ADV_NETLIST_OPT_DONT_TOUCH on" *) reg [31:0] prpof_id_reg;

	assign o_bitstream_incompatible = bitstream_incompatible_reg;
	
	always @(posedge clk)
	begin
		prpof_id_reg <= {32{1'b0}}; 
	end
	
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
								if (data[15:0] != prpof_id_reg[15:0]) begin
									bitstream_incompatible_reg <= 1'b1;
									check_state <= CHECK_COMPLETE;
								end
							end
							else if (data_count == 2'd3) begin
								if (data[15:0] != prpof_id_reg[31:16]) begin
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
								if (data[15:0] != prpof_id_reg[15:0]) begin
									bitstream_incompatible_reg <= 1'b1;
									check_state <= CHECK_COMPLETE;
								end
							end
							else if (data_count == 2'd3) begin
								if (data[15:0] != prpof_id_reg[31:16]) begin
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
`pragma questa_oem_00 "mXd8TZA7wcZD1M13kc63+RPfQcbqACFa9i0aqHtSQ62nFR2tUIuo2NzwpI/4cEyddIaPImJhX2CebOmYVV5f401JpALFknks4OMLrt70wgtKfnPjlusCxrHWEBHooOs6pkJXBTjC6oUhD2WVo05l5UBMB3Bob+qvd+BE/CfQQqbe8vYxUQkLM+Zgo0aA7Fpi0A6ZVpcWA/DDjPpZK8kPgz9Vfztbmx0x1RvFi9qEstKAosoq/hoOBHClaR4IHI1Y4Iio1X5r27DYn7oWcH1gG69hMO/QW/mWUKeSLlXDfIK879GJP0bxoQWKEV05eddrD8QN76iAwSk5riFqm2hc/YMfzNRgO9BBmM6XuA0A07pZSy7UstCg6u8ZJ+bBFLuWCMES8rREkvaOjZ2jg+Xs/+Y1yGQgXRue2Hcqym6lVlsDkdyioCtkZCknhPJ+JEDDBOrAdEdMkTmqD2NT+FWdLGtqlgckYaEKH8MGwniHwgM6id/CAa78Wwerqw3OmYUsvcJEqYwzj8Koon4UtdZD/6kUmrd623Jo/mG0uvd8dNBVXKepHQk5vi14hrbTxMWKn4XeN6CZpSXaFA/SDLMsDvmVLqJh1qyq69+kEwkTOUhYZ0KapM8YkG5Hp87MGyorNpyA293sOLm8UzARxBmh1152vOEIYvSXfpZ1fo8lsWLD0yDZayHsF/yK6skgA0u3lEA3ibViHJP9j0E7IN/h2tnH1Cg0r8ULpKewoOcc9n/Kr33v6oq0SNJHdwCHry1LhIXKn7C5YZw/oAIN4MJcAfOxPuB5HdZ9lSAwIhbO0kyiDzGIOHaZwOGF0dyTg3nyfXohKxbMfALEzNjbFveZeZBRXSVOEqK0F7Qj1UIFj4corSxk48GMU2Y1AYjjgiR597Lw5KWkkB02Q9LJOGP4Zmy27fQfQpvqi90avF5nxqr3oulzUWD7vk3jxQkC+cr5/RSLZULbMYPw/O+iEmT4eWL7BmQJAE+doulIpZwH0kWHYF+lA8u1Z9XctFYRVS/L"
`endif