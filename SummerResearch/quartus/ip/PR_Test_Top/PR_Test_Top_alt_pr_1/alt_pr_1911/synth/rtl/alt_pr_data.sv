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
// This is a modified version of `alt_pfl_data`.
module alt_pr_data
(
	clk,
	data_request,
	data_in,
	data_in_ready,
	data_in_read,
	
	data_out,
	data_out_ready,
	data_out_read
);

	parameter DATA_WIDTH = 8;
	parameter DELAY = 0;
	localparam READ_DELAY = (DELAY == 0) ? 0 : DELAY + 2;
	input data_request;
	input clk;
	input [DATA_WIDTH-1:0] data_in;
	input data_in_ready;
	output data_in_read;
	output [DATA_WIDTH-1:0] data_out;
	output data_out_ready;
	input data_out_read;

	reg data_out_ready;
	reg [DATA_WIDTH-1:0] data_out;
	
	// data clocking
	wire data_clk;
	generate
		if (DELAY > 0) begin
			genvar i;
			wire [DELAY-1:0] delays /* synthesis keep */;
			for (i=0; i < DELAY; i=i+1) begin : DELAY_LOOP
				if (i == 0)
					or (delays[i], clk, clk);
				else
					or (delays[i], delays[i-1], delays[i-1]);
			end
			assign data_clk = delays[DELAY-1];
		end
		else begin
			assign data_clk = clk;
		end
	endgenerate
	wire data_ready;
	wire data_in_read_wire = (data_in_ready & (~data_out_ready | data_out_read));
	generate
		if (READ_DELAY > 0) begin
			genvar j;
			wire [READ_DELAY-1:0] read_delays /* synthesis keep */;
			for (j=0; j < READ_DELAY; j=j+1) begin : READ_DELAY_LOOP
				if (j == 0)
					or (read_delays[j], data_in_read_wire, data_in_read_wire);
				else
					or (read_delays[j], read_delays[j-1], read_delays[j-1]);
			end
			assign data_ready = read_delays[READ_DELAY-1];
		end
		else begin
			assign data_ready = data_in_read_wire;
		end
	endgenerate

	always @ (posedge data_clk) begin
		if (data_ready)
			data_out <= data_in;
		else
			data_out <= data_out;
	end
	
	always @ (negedge data_request or posedge clk) begin
		if (~data_request)
			data_out_ready <= 1'b0;
		else if (data_out_ready) begin
			if (data_out_read & ~data_in_ready) // the only condition that can turn me down
				data_out_ready <= 1'b0;
			else 
				data_out_ready <= data_out_ready;
		end
		else if (data_in_ready) // the only condition that can turn me up
				data_out_ready <= 1'b1;
		else 
			data_out_ready <= data_out_ready;
	end

	assign data_in_read = data_in_read_wire;
endmodule
`ifdef QUESTA_INTEL_OEM
`pragma questa_oem_00 "mXd8TZA7wcZD1M13kc63+RPfQcbqACFa9i0aqHtSQ62nFR2tUIuo2NzwpI/4cEyddIaPImJhX2CebOmYVV5f401JpALFknks4OMLrt70wgtKfnPjlusCxrHWEBHooOs6pkJXBTjC6oUhD2WVo05l5UBMB3Bob+qvd+BE/CfQQqbe8vYxUQkLM+Zgo0aA7Fpi0A6ZVpcWA/DDjPpZK8kPgz9Vfztbmx0x1RvFi9qEstJk0U1UPbDE7+6Pcr00hKm1BO0JwnYAbiq0kVp5rEGDkmRPAU7bBlWpGNRwNYW43x6HRZ8Cy9IAeX2VcGKn7d0Xlv87C6drlJUnJYbekx6rE1HkCl+JsFQE4oHMBfvt3cf1mp8HthgcFg5IWjne98EHpsg13Cwtc8EC8BZ+FIXPCbQ8qpgE5kmv5YK/YheRkbzAakS411wvT1nf4TdP3FJV08m4b+0FzuItZMrtRb/aJxSLlmKUzumzFEGv9FKIOzfUHMz3h6A7RdiAQyVoJu7efmJ9HQjnofycm3T2xN77XMAU5lDOfBaMcd+7xJmqPqzNAFs6i4tgYxvEc7DGpcp8gDkkKcXypa5YpZdK4CI4UNKGeV/d9UWoVwsVdG8hnZdcsyYSH01N1x+eEpYdYRSJBkIXwu6/5wuQCG0nLO7zuDbXnk+GmT2areFw01NWlpbhqs+OlM3qMEe641CBoArCE5yYWLoMWG2JnrVLC+pcirw1RPu0GAq5UwerZkhkjwpYf07IrPbmxAZztybSjSizhvX/mqG/v8z9HKGFLK71e37sWu++LofNfcbx3GK4o2tGmXW6A0rS3/grie3t6imgRBcdYbWaI0sMyC5kWcjTyiPFtkN7GQ3Y6GdtNCBeV40U+gflYBo+YfTthW0FevXapqYVc40MQnc02DdURg9yQ4mXiLUbi9kOINABC+cp00zKYuE4c49d4y23i7YdX3jcrVUrxSWo+6Wz3ogL275FoQ6tcRbLmbwLGw/IfDOQSMFs996Gn2awwrTHgZmUljYa"
`endif