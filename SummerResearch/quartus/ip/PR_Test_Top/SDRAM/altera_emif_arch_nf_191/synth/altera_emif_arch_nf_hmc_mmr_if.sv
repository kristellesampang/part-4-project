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


///////////////////////////////////////////////////////////////////////////////
// This module is responsible for exposing the MMR Avalon interfaces through which
// soft logic interacts with the Hard Memory Controller inside the tile. 
// The tile WYSIWYG blocks collapse the individual Avalon signals into big
// buses. This module re-wires the big buses into proper Avalon interfaces.
// 
///////////////////////////////////////////////////////////////////////////////
module altera_emif_arch_nf_hmc_mmr_if #(

   // Definition of port widths for "ctrl_mmr" interface (auto-generated)
   parameter PORT_CTRL_MMR_SLAVE_ADDRESS_WIDTH             = 1,
   parameter PORT_CTRL_MMR_SLAVE_RDATA_WIDTH               = 1,
   parameter PORT_CTRL_MMR_SLAVE_WDATA_WIDTH               = 1,
   parameter PORT_CTRL_MMR_SLAVE_BCOUNT_WIDTH              = 1
) (
   // MMR signals between core and HMC
   input  logic [33:0]                                        ctl2core_mmr_0,
   output logic [50:0]                                        core2ctl_mmr_0,
   input  logic [33:0]                                        ctl2core_mmr_1,
   output logic [50:0]                                        core2ctl_mmr_1,
   
   // Ports for "ctrl_mmr" interface (auto-generated)
   output logic                                               mmr_slave_waitrequest_0,
   input  logic                                               mmr_slave_read_0,
   input  logic                                               mmr_slave_write_0,
   input  logic [PORT_CTRL_MMR_SLAVE_ADDRESS_WIDTH-1:0]       mmr_slave_address_0,
   output logic [PORT_CTRL_MMR_SLAVE_RDATA_WIDTH-1:0]         mmr_slave_readdata_0,
   input  logic [PORT_CTRL_MMR_SLAVE_WDATA_WIDTH-1:0]         mmr_slave_writedata_0,
   input  logic [PORT_CTRL_MMR_SLAVE_BCOUNT_WIDTH-1:0]        mmr_slave_burstcount_0,
   input  logic                                               mmr_slave_beginbursttransfer_0,
   output logic                                               mmr_slave_readdatavalid_0,
   
   output logic                                               mmr_slave_waitrequest_1,
   input  logic                                               mmr_slave_read_1,
   input  logic                                               mmr_slave_write_1,
   input  logic [PORT_CTRL_MMR_SLAVE_ADDRESS_WIDTH-1:0]       mmr_slave_address_1,
   output logic [PORT_CTRL_MMR_SLAVE_RDATA_WIDTH-1:0]         mmr_slave_readdata_1,
   input  logic [PORT_CTRL_MMR_SLAVE_WDATA_WIDTH-1:0]         mmr_slave_writedata_1,
   input  logic [PORT_CTRL_MMR_SLAVE_BCOUNT_WIDTH-1:0]        mmr_slave_burstcount_1,
   input  logic                                               mmr_slave_beginbursttransfer_1,
   output logic                                               mmr_slave_readdatavalid_1   
);
   timeunit 1ns;
   timeprecision 1ps;
   
   assign core2ctl_mmr_0[9:0]        = mmr_slave_address_0;
   assign core2ctl_mmr_0[13:10]      = 'b0;
   assign core2ctl_mmr_0[45:14]      = mmr_slave_writedata_0;
   assign core2ctl_mmr_0[46]         = mmr_slave_write_0;
   assign core2ctl_mmr_0[47]         = mmr_slave_read_0;
   assign core2ctl_mmr_0[49:48]      = mmr_slave_burstcount_0;
   assign core2ctl_mmr_0[50]         = mmr_slave_beginbursttransfer_0;
   
   assign mmr_slave_readdata_0       = ctl2core_mmr_0[31:0];
   assign mmr_slave_readdatavalid_0  = ctl2core_mmr_0[32];
   assign mmr_slave_waitrequest_0    = ctl2core_mmr_0[33];

   assign core2ctl_mmr_1[9:0]        = mmr_slave_address_1;
   assign core2ctl_mmr_1[13:10]      = 'b0;
   assign core2ctl_mmr_1[45:14]      = mmr_slave_writedata_1;
   assign core2ctl_mmr_1[46]         = mmr_slave_write_1;
   assign core2ctl_mmr_1[47]         = mmr_slave_read_1;
   assign core2ctl_mmr_1[49:48]      = mmr_slave_burstcount_1;
   assign core2ctl_mmr_1[50]         = mmr_slave_beginbursttransfer_1;
   
   assign mmr_slave_readdata_1       = ctl2core_mmr_1[31:0];
   assign mmr_slave_readdatavalid_1  = ctl2core_mmr_1[32];
   assign mmr_slave_waitrequest_1    = ctl2core_mmr_1[33];   
   
endmodule

`ifdef QUESTA_INTEL_OEM
`pragma questa_oem_00 "BUwlYfHqKCYPcXIP7xrnQ9Se5hc+WCW/arrBU5W+lFIz4wbefW0My5vX6xBCsRZeDFCX8Hl+sMTGZJoji0A9oHt2LftULu0FLRu+mk51alTJ3uomtGywN+PFSqf/fgMyVODG1WmlMI935DQeVtY8c0/LTaQ2hClpkazdo6IufJnkztEP0APWCavclOK3sEzCw6ySG+devOHMd72zKrOr9Hnpx+hVpDg9hFG/o4PmuuGND3bmQwpRMGakNnz3HaZ1MW/JaSTtLJ/LgHqI+dzuB3e5ZjyR7EeNq1ij6890BEUgHzTmA+ViwlVasULXr99OT6QC+aaTU7kAoNHe3NT/90HbzCMYGrxbOg8UOlngvqQTBPdUbVFLxcwnn1BIJLFuC5/0BCpQHtwTaH5HtPGgFngVS78+kKsj2WUvksLeOf5ioi22jc4uT4VqyTq4Q83GUZJiGk7/pnmPXfdarYEZiuUDG1YxQ8BzcJ7c9sLtA6C5eiVxXLurlj1ZApFGu9AEluLQi5oJk6FdDURf2dKPafvOJ1YQNlqDXNKvuabWJJgDBlvlDysWDBJYtoOuiLTJ8lECxPPRlpfPYQu63plqYVe8qjW2YT2hpWjlgrpj4Ng2aoICWaGl65BIuiAy+3YmKSwHzs76EcVyYiXskEjUFrHlw8bDRv+Wx4aNn28nPg2J8u9YGbXq8kxcd3+nOPi0PdWKLsAaq2wDC9F7Hxn3cOmf2A1KA/+IiGXi33Sfqfkt0Jq5LW0T4O3hb46zUhFDGXsNBAL7j/K5rx5Pb31mICbWYzOi4zB9gHFIwLt0xeFXVLSePu66pD5dkrctfs09zSJWC/QIXXLhK+Lu2bDyQXuntxs8aUoaUHAcATp5s98EDfmYBTynuSaMfXQ9d/ngCplq7hQGa70btzJHrCWC+psPITKkS9kIROUVUN/JAEgxkm/0XMXhzgN6+o8Bo40bupqVi+mjhlTkRspUDZSvP3607jtHdGx/y7qRS9oIeyc8MJMw7qnNxbgjggChpRQz"
`endif