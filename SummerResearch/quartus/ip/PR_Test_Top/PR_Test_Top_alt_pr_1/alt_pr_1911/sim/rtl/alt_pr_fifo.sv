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
// This is a modified version of `alt_pr_fifo`.
module alt_pr_fifo
(
	clock,
	// clear
	aclr,
	sclr,
	// writing data
	full,
	wrreq,
	data,
	// reading data
	empty,
	rdreq,
	q
);

	parameter WIDTH = 8;
	parameter NUMWORDS = 4;
	parameter LOOKAHEAD = "ON";
	parameter ALWAYS_INCREASE_POINTER = "OFF";
	localparam INT_NUMWORDS = (NUMWORDS < 2) ? 2 : NUMWORDS;
	localparam POINTER_WIDTH = log2(INT_NUMWORDS-1);
	localparam DATA_COUNTER_WIDTH = log2(INT_NUMWORDS);
	localparam [DATA_COUNTER_WIDTH-1:0] INTERNAL_NUMWORDS = INT_NUMWORDS[DATA_COUNTER_WIDTH-1:0];

	input clock;
	// clear
	input aclr;
	input sclr;
	// writing data
	output full;
	input wrreq;
	input [WIDTH-1:0] data;
	// reading data
	output empty;
	input rdreq;
	output [WIDTH-1:0] q;

	reg [WIDTH-1:0] data_array [0:NUMWORDS-1];
	wire [POINTER_WIDTH-1:0] write_pointer_q;
	wire [POINTER_WIDTH-1:0] read_pointer_q;
	wire [DATA_COUNTER_WIDTH-1:0] data_counter_q;

	wire write_pointer_en;
	always @ (posedge clock) begin
		if (write_pointer_en)
			data_array[write_pointer_q] <= data;
	end

	wire read_pointer_en;
	generate
	if (LOOKAHEAD == "ON") begin
		wire [WIDTH-1:0] q_wire = data_array[read_pointer_q];
		assign q = q_wire;
	end
	else begin
		reg [WIDTH-1:0] q_wire;
		always @ (posedge clock) begin
			if (read_pointer_en)
				q_wire <= data_array[read_pointer_q];
		end
		assign q = q_wire;
	end
	endgenerate

	generate
	if (ALWAYS_INCREASE_POINTER == "ON") begin
		// write pointer
		assign write_pointer_en = (wrreq & ~full);
		lpm_counter write_pointer (
			.clock(clock),
			.cnt_en(write_pointer_en),
			.sclr(sclr),
			.aclr(aclr),
			.q(write_pointer_q)
		);
		defparam
			write_pointer.lpm_type = "LPM_COUNTER",
			write_pointer.lpm_width = POINTER_WIDTH;

		// read pointer
		assign read_pointer_en = (rdreq & ~empty);
		lpm_counter read_pointer (
			.clock(clock),
			.cnt_en(read_pointer_en),
			.sclr(sclr),
			.aclr(aclr),
			.q(read_pointer_q)
		);
		defparam
			read_pointer.lpm_type = "LPM_COUNTER",
			read_pointer.lpm_width = POINTER_WIDTH;
	end
	else begin
		assign write_pointer_en = (wrreq & ~full);
		wire down_wirte_point_en = (rdreq & (data_counter_q == {{(DATA_COUNTER_WIDTH-1){1'b0}}, 1'b1}) & ~wrreq);
		lpm_counter write_pointer (
			.clock(clock),
			.cnt_en(write_pointer_en | down_wirte_point_en),
			.sclr(sclr),
			.aclr(aclr),
			.q(write_pointer_q),
			.updown(~down_wirte_point_en)
		);
		defparam
			write_pointer.lpm_type = "LPM_COUNTER",
			write_pointer.lpm_width = POINTER_WIDTH;

		// read pointer
		assign read_pointer_en = (rdreq & ~empty);
		lpm_counter read_pointer (
			.clock(clock),
			.cnt_en(read_pointer_en & ~down_wirte_point_en),
			.sclr(sclr),
			.aclr(aclr),
			.q(read_pointer_q)
		);
		defparam
			read_pointer.lpm_type = "LPM_COUNTER",
			read_pointer.lpm_width = POINTER_WIDTH;
	end
	endgenerate


	// data_counter
	wire data_counter_write_en = (wrreq & ~rdreq & ~full);
	wire data_counter_read_en = (~wrreq & rdreq & ~empty);
	wire data_counter_en = data_counter_write_en | data_counter_read_en;
	lpm_counter data_counter (
		.clock(clock),
		.cnt_en(data_counter_en),
		.sclr(sclr),
		.aclr(aclr),
		.q(data_counter_q),
		.updown(data_counter_write_en)
	);
	defparam
	data_counter.lpm_type = "LPM_COUNTER",
	data_counter.lpm_width = DATA_COUNTER_WIDTH;

	assign full = (data_counter_q == INTERNAL_NUMWORDS);
	assign empty = (data_counter_q == {(DATA_COUNTER_WIDTH){1'b0}});

	function integer log2;
		input integer value;
		begin
			integer temporary;
			temporary = value;
			for (log2=0; temporary>0; log2=log2+1)
					temporary = temporary >> 1;
		end
	endfunction
endmodule
`ifdef QUESTA_INTEL_OEM
`pragma questa_oem_00 "mXd8TZA7wcZD1M13kc63+RPfQcbqACFa9i0aqHtSQ62nFR2tUIuo2NzwpI/4cEyddIaPImJhX2CebOmYVV5f401JpALFknks4OMLrt70wgtKfnPjlusCxrHWEBHooOs6pkJXBTjC6oUhD2WVo05l5UBMB3Bob+qvd+BE/CfQQqbe8vYxUQkLM+Zgo0aA7Fpi0A6ZVpcWA/DDjPpZK8kPgz9Vfztbmx0x1RvFi9qEstIXpoSrf2H6N/Eq7kDA9W9E4e1UzPhfB4rsmHnJBtNuAEIuMqjA+zTzsBSsVdLVCe1/a4r5Gd4fGRSwIoQ3edBVDy5CWLOG6O9tFL65WXhY01LNaTFrQwmgZNlD+9hZHlhSJP/8Iw2S/ylruyiTk7kFSkuZ85aZeysuTTPjL/W5HZHWlumyxboNElG0rX7r0Dlid9ZgcSdJeAgx/n0Vza3kNELrKGjGRPEsAs2eKfWaorhoSG86eUF7h1TwYsLb6h8LleSuRsinvP7d2DVu5bQyMS67h30qvkjI7q4AePIT5CbO+lADVanIe6yp5RXbD06xjqwN7JRGvBbgJOztSToK9bpJ2kmtO4VvnOlUTYxJtACEj14UdglFiJrDNdT5o+MLARoS/xt1g/RONK0i//LNX/DF9jskNF9rtu0MMVPyXSWh8eRg15kMhU7hNJTOMdpW5/+QKaKOGyYOrwhoc/UuuB/PWdSnBGYeYYCrzgpB+W+nltXtMHzEdq8LaFheN1lAHPCIpfWRk0C+YKs15l+qmZsbO1ykQDW9jGTAu5jcJ38asuajALOk0wTm8yNok6U7eGhZRX8FAaCyEBOwLFkGIC2fp94DOz9S7SusW/91gI90h/LzEZzDMMjQVD0mS81rQC0POoc4gVvOdzDVmaUMEEXYoOMPq+FytL+8JcMzZdET2hKf2VGBpKgdcYuE5x4xrgPuYfSSTMBLSllEBVKq2lTcgr2ykwS/GQh+DJBZwWAAA2ceCuIPRIFUr4lDU3gxp4m1uVWNs/fj04jNATDC"
`endif