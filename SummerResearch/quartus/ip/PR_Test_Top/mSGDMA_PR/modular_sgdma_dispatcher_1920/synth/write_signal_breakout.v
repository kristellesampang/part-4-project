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


/*

  This block is used to breakout the 256 bit streaming ports to and from the write master.
  The information sent through the streaming ports is a bundle of wires and buses so it's
  fairly inconvenient to constantly refer to them by their position amungst the 256 lines.
  This block also provides a layer of abstraction since the descriptor buffers block has
  no clue what format the descriptors are in except that the 'go' bit is written to.  This
  means that using this block you could move descriptor information around without affecting
  the top level dispatcher logic.
  
  
  1.0  06/29/2009 - First version of this block of wires
  
  1.1  11/15/2012 - Added in an additional 32 bits of address for extended descriptors
  
*/


// synthesis translate_off
`timescale 1ns / 1ps
// synthesis translate_on

// turn off superfluous verilog processor warnings 
// altera message_level Level1 
// altera message_off 10034 10035 10036 10037 10230 10240 10030 


module write_signal_breakout (
  write_command_data_in,     // descriptor from the write FIFO
  write_command_data_out,  // reformated descriptor to the write master

  // breakout of command information
  write_address,
  write_length,
  write_park,
  write_end_on_eop,
  write_transfer_complete_IRQ_mask,
  write_early_termination_IRQ_mask,
  write_error_IRQ_mask,
  write_wait_for_response,
  write_burst_count,      // when 'ENHANCED_FEATURES' is 0 this will be driven to ground
  write_stride,           // when 'ENHANCED_FEATURES' is 0 this will be driven to ground
  write_sequence_number,  // when 'ENHANCED_FEATURES' is 0 this will be driven to ground

  // additional control information that needs to go out asynchronously with the command data
  write_stop,
  write_sw_reset
);

  parameter DATA_WIDTH = 256;  // 256 bits when enhanced settings are enabled otherwise 128 bits

  input [DATA_WIDTH-1:0] write_command_data_in;
  output wire [255:0] write_command_data_out;

  output wire [63:0] write_address;
  output wire [31:0] write_length;
  output wire write_park;
  output wire write_end_on_eop;
  output wire write_transfer_complete_IRQ_mask;
  output wire write_early_termination_IRQ_mask;
  output wire [7:0] write_error_IRQ_mask;
  output wire write_wait_for_response;
  output wire [7:0] write_burst_count;
  output wire [15:0] write_stride;
  output wire [15:0] write_sequence_number;

  input write_stop;
  input write_sw_reset;

  assign write_address[31:0] = write_command_data_in[63:32];
  assign write_length = write_command_data_in[95:64];


  generate
    if (DATA_WIDTH == 256)
    begin
      assign write_park = write_command_data_in[235];
      assign write_end_on_eop = write_command_data_in[236];
      assign write_transfer_complete_IRQ_mask = write_command_data_in[238];
      assign write_early_termination_IRQ_mask = write_command_data_in[239];
      assign write_error_IRQ_mask = write_command_data_in[247:240];
      assign write_wait_for_response = write_command_data_in[249];
      assign write_burst_count = write_command_data_in[127:120];
      assign write_stride = write_command_data_in[159:144];
      assign write_sequence_number = write_command_data_in[111:96];
      assign write_address[63:32] = write_command_data_in[223:192];
    end
    else
    begin
      assign write_park = write_command_data_in[107];
      assign write_end_on_eop = write_command_data_in[108];
      assign write_transfer_complete_IRQ_mask = write_command_data_in[110];
      assign write_early_termination_IRQ_mask = write_command_data_in[111];
      assign write_error_IRQ_mask = write_command_data_in[119:112];
      assign write_wait_for_response = write_command_data_in[121];
      assign write_burst_count = 8'h00;
      assign write_stride = 16'h0000;
      assign write_sequence_number = 16'h0000; 
      assign write_address[63:32] = 32'h00000000;    
    end
  endgenerate


  // big concat statement to glue all the signals back together to go out to the write master (MSBs to LSBs)
  assign write_command_data_out = {{132{1'b0}},  // zero pad the upper 132 bits
                                  write_address[63:32],
                                  write_stride,
                                  write_burst_count,
                                  write_sw_reset,
                                  write_stop,
                                  write_wait_for_response,
                                  write_end_on_eop,
                                  write_length,
                                  write_address[31:0]};

endmodule
`ifdef QUESTA_INTEL_OEM
`pragma questa_oem_00 "d5vhRwahtu8K9yyM8KR4OOUSgHRUUd/iZK0O+4Vc7vQ0A733p9gY0o587LXEP52iAIIMByDdR1wcsL18HQtfJF8E2VG+wVMmpQ64EE6q84uGd7nmdLQeT1PX73A3+xvRTo2VqBPPQs9TDabvm3GtHppSmtjgfDWMnGnsoEsUgMh7jxHPXtTG04d7CfaIAPo+gEAxaU+3an+UaRyraEAreqvUHpkjs4PfOEYgQrDmYQUVo9szMNEGhhQVVxpKY4fpr9bGV7N1oXgdmdZyBek54kper+7KgadgA60A+YKzZTnYwXTgvoMhat+u68yhghDpisPZ0b+bLEKFlk9ha/z/SfZjlGz87rvSJKm9lMmZdqsjd7SqaSHe81rzj2GElxFljcfVBMhEGExQ/MlcNT5Ag5xowubva2vdmfd7L6hRVUNsg+BHjs23amuxKn/xp4m24l7zWy0J6fNZubGpcwnnaxANEoK9wZB3MwgUKbBnNHXKh5uw/XOrsT+8oWOtOYY8G0acbe8uOCsJ9NGWoPrsoLWIIMdkC1fdtJDT148cMvPjyLE37E4o3rb01VhR/Qs1PVLPxioA6Fh35uT2/PxowHaujEyDfiT9Lj6olFqeXgUdxTOo83oIXSElyp0AKdl9hIMHBtk1SbvM2gq0V7/4ES0yaohJ/8Szva26jrrHnx+wp1R6c5nRnQywyZY6ogXjMpCubn8kCv4WJhtizC66rOZr8tA8/ZTcwvCNEN7Wy0Ss6M7e0zDs8YTycqwGOJDdw9pQjLEZz5KyMBSsQmHw1J9Q2WIaA6jKzpjTHTIaCVGQjHIya06mgEwKD3WJhFDeyFdJ0rFHaezQSh5hbf+UDR6CWJVLJX7G2tVmHNj8lgW8pzJE7yAPNpUW4JVgY3aen47aF5v7VE3cdkfDCsFkurjnh3YD5kAp1Cr4VYrCeK2bQpcWvl358SxOBiqxkA/+gDMe2pF2L43O/wR0Ozau99am1Qr3dAuIlZrrb6jjvnQ9BaICNOb6SOJl5xDbIKSb"
`endif