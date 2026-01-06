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
module alt_pr_bitstream_compatibility_checker_v2(
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
    o_pr_pof_id,
    o_bitstream_incompatible
);

    parameter CDRATIO = 1;
    parameter CB_DATA_WIDTH = 16;
    parameter PR_INTERNAL_HOST = 1;
    parameter EXT_HOST_PRPOF_ID = 32'hFFFFFFFF; //4294967295

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

    output [31:0] o_pr_pof_id;
    output o_bitstream_incompatible;

    reg [7:0] data_count;
    reg [1:0] check_state;
    reg bitstream_incompatible_reg;

    wire [31:0] prpof_id_holder;

    generate
        if (PR_INTERNAL_HOST == 1) begin
            (* preserve, altera_attribute = "-name PRPOF_ID on; -name ADV_NETLIST_OPT_DONT_TOUCH on" *) reg [31:0] prpof_id_reg;

            always @(posedge clk) begin
                prpof_id_reg <= {32{1'b1}};
               // synthesis translate_off
               // In simulation use a constant DEADBEEF
               prpof_id_reg <= 32'hDEAD_BEEF;
               // synthesis translate_on
            end
            assign prpof_id_holder = prpof_id_reg; 
        end
        else
            assign prpof_id_holder = EXT_HOST_PRPOF_ID;       
    endgenerate
    
    assign o_pr_pof_id = prpof_id_holder;

    assign o_bitstream_incompatible = bitstream_incompatible_reg;

    always @(posedge clk) begin
        if (~nreset)
            check_state <= IDLE;
        else begin
            case (check_state)
                IDLE: 
                    begin
                        data_count <= 8'd0;
                        bitstream_incompatible_reg <= 1'b0;

                        if (freeze)
                            check_state <= WAIT_FOR_READY;
                    end

                WAIT_FOR_READY: 
                    begin
                        if (~freeze)
                            check_state <= IDLE;
                        else if (pr_ready)
                            check_state <= CHECK_START;
                    end

                CHECK_START: 
                    begin
                        if (~freeze || ~pr_ready || crc_error || pr_error || pr_done)
                            check_state <= CHECK_COMPLETE;
                        else if (data_count == 8'd71) begin     // use for 32
                            if (data_valid && data_ready) begin
                                if (CB_DATA_WIDTH == 32) begin
                                    if (data[31:0] != prpof_id_holder[31:0])
                                        bitstream_incompatible_reg <= 1'b1;
                                    
                                    check_state <= CHECK_COMPLETE;
                                end
                                else
                                    data_count <= data_count + 8'd1;
                            end
                        end
                        else if (data_count == 8'd142) begin    // use for 16
                            if (data_valid && data_ready) begin
                                if (data[15:0] != prpof_id_holder[15:0]) begin
                                    bitstream_incompatible_reg <= 1'b1;
                                    check_state <= CHECK_COMPLETE;
                                end
                                else
                                    data_count <= data_count + 8'd1;
                            end
                        end
                        else if (data_count == 8'd143) begin    // use for 16
                            if (data_valid && data_ready) begin
                                if (data[15:0] != prpof_id_holder[31:16])
                                    bitstream_incompatible_reg <= 1'b1;
                                    
                                check_state <= CHECK_COMPLETE;
                            end
                        end
                        else if (data_valid && data_ready)
                                data_count <= data_count + 8'd1;
                    end

                CHECK_COMPLETE: 
                    begin
                        if (~freeze)
                            check_state <= IDLE;
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
`pragma questa_oem_00 "mXd8TZA7wcZD1M13kc63+RPfQcbqACFa9i0aqHtSQ62nFR2tUIuo2NzwpI/4cEyddIaPImJhX2CebOmYVV5f401JpALFknks4OMLrt70wgtKfnPjlusCxrHWEBHooOs6pkJXBTjC6oUhD2WVo05l5UBMB3Bob+qvd+BE/CfQQqbe8vYxUQkLM+Zgo0aA7Fpi0A6ZVpcWA/DDjPpZK8kPgz9Vfztbmx0x1RvFi9qEstLB57PwcMnNnUzL0te5NZeygudHoR0kfT4amnAjJvvqLG2gzZsztPA66nW1XfrZk7oA+zyGL82/fz1oJvXwdF2H8chBVCgT9/15a2te+tyrAnpPyHYxVkyUKcwAusVC6sqC9Q9gbMT4+CFKjOrfYLihaqHTtj5TLJesy6uDLVbyRM6F4Jk+262S+D7k1aY8Z+qBslOHKBbvJliloyLHzglqwyo4opqhmfoSetwjx+4XVrIqZUg3CL931prlkbtmiATqEGDfYW/DIxJc4haP9239xsDKGarLeVuooMOLDUCmiq0sANsDmMylA/1H8tR6t2JpRH0xdzSL+QJLELWUpSwFlwY7I2byF1V98JIxHjoRWwJ0S9xA1ulXRSmieo/kUZ+Kpw+kbHYT2gSl4QZsj9L2wDsdE61ET5ilVYDiEd+gT1ZL+N+3SL1K8Vh68a5Yj0/qAzZumPMe2UJe9TusAuCC/psfnG8zVgQW2r43A+O8ZdrO1g8D+G3IbDhehUKq8eEU0IQdHHOAmUVZIAiBldGBQW8B1ZyzkSMWvYAHJlm6cnWTj7FzxiKBI6PK9StpSdkKiO0h+0Bu/e5xB8n5vAtJVsJ34gT0aQ6baIo6QAFOxcUFqhBE9Ekgb7T+XHrLUjoO4e6BTnu7A/25JODoj8++rxVjJUmKMztxWRJd6B3yhdMbUSiwPAiioEGCN/sg4kC7yhJzfQlsN0jxlDjM4Yi913b2TX1+0HpS48xumAAa6yO0T90nRsx7dOh7FKYdwifpC/3Jh087jAL9wg6iHP5M"
`endif