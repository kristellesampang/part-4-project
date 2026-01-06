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
module alt_pr_mux#(
    parameter DATA_WIDTH = 16
) (
    input  logic                   select,

    input  logic                   sink0_valid,
    input  logic[DATA_WIDTH - 1:0] sink0_data,
    output logic                   sink0_ready,

    input  logic                   sink1_valid,
    input  logic[DATA_WIDTH - 1:0] sink1_data,
    output logic                   sink1_ready,

    output logic                   source_valid,
    output logic[DATA_WIDTH - 1:0] source_data,
    input  logic                   source_ready
);
    assign sink0_ready = select == 0 ? source_ready : 1'b0;

    assign sink1_ready = select == 0 ? 1'b0 : source_ready;

    assign source_valid = select == 0 ? sink0_valid : sink1_valid;
    assign source_data = select == 0 ? sink0_data : sink1_data;
endmodule
`ifdef QUESTA_INTEL_OEM
`pragma questa_oem_00 "mXd8TZA7wcZD1M13kc63+RPfQcbqACFa9i0aqHtSQ62nFR2tUIuo2NzwpI/4cEyddIaPImJhX2CebOmYVV5f401JpALFknks4OMLrt70wgtKfnPjlusCxrHWEBHooOs6pkJXBTjC6oUhD2WVo05l5UBMB3Bob+qvd+BE/CfQQqbe8vYxUQkLM+Zgo0aA7Fpi0A6ZVpcWA/DDjPpZK8kPgz9Vfztbmx0x1RvFi9qEstIiRSkr6Hqr7bv1ksc1vtEfRK2V2gFDFesWYX7N1Pgdm95FM0yaVKsKEE//gLoPawaTfEIrEfL1vdvGx6cMewmGf+2GHlN4RpPs6rXX/HiHu/nv6HsQ+V/aMs13eIJN6CqxtUBp2oTaWMxwO9w2TSwSC7nESeDnqVwOdSaP9Ccv8HGreFRVlPwGNyBWgBIF7Rw9nfVOTC4R4v3PRAjCfbZM32u9ELYZU7Mn+LfdxQ1DEjUCbN/T1VOstvWO9oroGMCoBftYUXxSnM+ulTilSyNOgIiNM0mJ52XjY2AUFI1fF4jJqQDMWqi6CKNHlcXSKxH9OSx1A6DMGW+08cdf1RkPGfYMynYAypah6GznzvZfUy0EtFM55RSuV1cixjcDeY8XYKM69wVGjUWEqr9FZkKPxreubZy6I8b7Hfc6wepQOePSxV9MrTPxDZeer9kUgGRq3zyVJfr0ZtfWTl4pVS2air4ni0H7V4iu6AgZCHm9yG3sQHA/vc7d2e4KQDY2GTUx+4esOOlhbKOZAXsWHd/gf7q3uw1IW0jaIfw6YuuB1A5hAABEGUvZ4Oa9RSyGlzjro7VVlKRtZot2eqUDD9ViwtqRnZEi09nHU0Ku5Rj00h2SzbOLjqeNh+aHsRuTNzaeswvOkp3BDRVBNqoVXIlSEeTLn1LuBeKrxckKWhyXP7NzvFSrg2zidbcgKCstBEkoPeNpHJezQoZYno1mmsXqmt9JAxZp9IWg+ldh3Ypla9TNtXzFQoeRZnNf0aeV3Y7O69q8apCReXTLQOnJlcEF"
`endif