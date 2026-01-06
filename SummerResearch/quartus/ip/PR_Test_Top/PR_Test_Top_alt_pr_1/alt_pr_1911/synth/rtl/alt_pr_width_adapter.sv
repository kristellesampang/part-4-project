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
module alt_pr_width_adapter#(
    parameter SINK_DATA_WIDTH     = 16,
    parameter SOURCE_DATA_WIDTH   = 16,
    parameter ENABLE_DATA_PACKING = 1 // Only applicable when it is narrow to wide.
) (
    input  logic                          clk,
    input  logic                          nreset,

    input  logic                          sink_valid,
    input  logic[SINK_DATA_WIDTH - 1:0]   sink_data,
    output logic                          sink_ready,

    output logic                          source_valid,
    output logic[SOURCE_DATA_WIDTH - 1:0] source_data,
    input  logic                          source_ready
);
    generate
        if (SINK_DATA_WIDTH < SOURCE_DATA_WIDTH) begin
            alt_pr_up_converter#(
                .SINK_DATA_WIDTH(SINK_DATA_WIDTH),
                .SOURCE_DATA_WIDTH(SOURCE_DATA_WIDTH),
                .ENABLE_DATA_PACKING(ENABLE_DATA_PACKING)
            ) alt_pr_up_converter (
                .clk(clk),
                .nreset(nreset),

                .sink_valid(sink_valid),
                .sink_data(sink_data),
                .sink_ready(sink_ready),

                .source_valid(source_valid),
                .source_data(source_data),
                .source_ready(source_ready)
            );
        end else if (SINK_DATA_WIDTH > SOURCE_DATA_WIDTH) begin
            alt_pr_down_converter#(
                .SINK_DATA_WIDTH(SINK_DATA_WIDTH),
                .SOURCE_DATA_WIDTH(SOURCE_DATA_WIDTH)
            ) alt_pr_down_converter (
                .clk(clk),
                .nreset(nreset),

                .sink_valid(sink_valid),
                .sink_data(sink_data),
                .sink_ready(sink_ready),

                .source_valid(source_valid),
                .source_data(source_data),
                .source_ready(source_ready)
            );
        end else begin
            assign sink_ready = source_ready;

            assign source_valid = sink_valid;
            assign source_data = sink_data;
        end
    endgenerate
endmodule
`ifdef QUESTA_INTEL_OEM
`pragma questa_oem_00 "mXd8TZA7wcZD1M13kc63+RPfQcbqACFa9i0aqHtSQ62nFR2tUIuo2NzwpI/4cEyddIaPImJhX2CebOmYVV5f401JpALFknks4OMLrt70wgtKfnPjlusCxrHWEBHooOs6pkJXBTjC6oUhD2WVo05l5UBMB3Bob+qvd+BE/CfQQqbe8vYxUQkLM+Zgo0aA7Fpi0A6ZVpcWA/DDjPpZK8kPgz9Vfztbmx0x1RvFi9qEstIXzDEclrXBgvQvFxvFNiLCY5aIIaIzPoelXugETIV56WQFtbfcxzdeb2f6a5nnV/B+3h/hsxDhj33PW/yKR2z120HiePuTRY77dZiFbYd3ehepeFcwkAbswWvf6vTfi0JDl14jaj1cOE1LQm/Z6+uFLN+pL4WBo1c4SQ/DO8umxuK5ZKc/QmhsJ0TSqzac6yz+LQeijne8x4Lz8fNmrDga0dqa33UP0lqoxDZEyFgsgb86h1p4T2c7puMp7/1aQzcTobRj6rqupIFI1LLJSps6EAoq8SaNB6aZ2a3Gm7madBObusYB6xCw8kCe6SjCewfd2OzRX/4TZOYTo9/Mfeh9MD38CzkZjVqFrs2cc9Potpgek6PoRa0ym+FNxbrAjA+lYN2Fcq0gDzZ/uwlwTlcBqtMutgAoJRLsVkb1MK4Qm88yd7Q8W0NctfhGHf3QXezy6OrPPQWop4YxaJIC8o5PkNPwUDPN3cIOQl4EwbE1LnivA8P7htoUTVYEP5MGzbKCypOVG5FmARa5lThKQpxR+FuXvYiGGBw8GzBMYUT6ixzTXgA20CCK2np9GyqbuL+uGKmOPi1UWxSdCaGV9ZLoE2wvgdmzGOro2iU3FgLZWLjPJPi97hgM683pbYdI5JCqGDokUGfLSppCVhZ3GlOVhmMxVpEdmF5HPwT+pfWQWQ0t2P9GVKS0jhcBQClbG5VnXM94oJdDBwdlYLwt3C98yERlr3rCpcsdW8ogN1nW5kZlo6F9cNoW1v79TqWyFrp8ANit1CZNaH1of6jynZ4E"
`endif