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



module altera_emif_arch_nf_cal_counter # (
   parameter IS_HPS = 0
) (
   input logic pll_ref_clk_int,
   input logic global_reset_n_int,
   input logic afi_cal_in_progress
);
   timeunit 1ps;
   timeprecision 1ps;
   
   logic                         done;
   logic [31:0]                  clk_counter;
   
   generate
      if (IS_HPS == 0) begin : non_hps
         logic                         reset_n_sync;
         logic                         cal_in_progress_sync;

         altera_std_synchronizer_nocut
         inst_sync_reset_n (
            .clk     (pll_ref_clk_int),
            .reset_n (1'b1),
            .din     (global_reset_n_int),
            .dout    (reset_n_sync)
         );

         altera_std_synchronizer_nocut
         inst_sync_cal_in_progress (
            .clk     (pll_ref_clk_int),
            .reset_n (1'b1),
            .din     (afi_cal_in_progress),
            .dout    (cal_in_progress_sync)
         );

         enum {
            INIT,
            IDLE,
            COUNT_CAL,
            STOP
         } counter_state;

         assign done = ((counter_state == STOP) ? 1'b1 : 1'b0);

         always_ff @(posedge pll_ref_clk_int) begin
            if(reset_n_sync == 1'b0) begin
               counter_state <= INIT;
            end
            else begin
               case(counter_state)
                  INIT:
                  begin
                     clk_counter <= 32'h0;
                     counter_state <= IDLE;
                  end

                  IDLE:
                  begin
                     if (cal_in_progress_sync == 1'b1)
                     begin
                        counter_state <= COUNT_CAL;
                     end
                  end

                  COUNT_CAL:
                  begin
                     clk_counter[31:0] <= clk_counter[31:0] + 32'h0000_0001;

                     if (cal_in_progress_sync == 1'b0)
                     begin
                        counter_state <= STOP;
                     end
                  end

                  STOP:
                  begin
                     counter_state <= STOP;
                  end

                  default:
                  begin
                     counter_state <= INIT;
                  end
               endcase
            end
         end
      end else begin : hps
         assign done = 1'b1;
         assign clk_counter = '0;
      end
   endgenerate

`ifdef ALTERA_EMIF_ENABLE_ISSP
   altsource_probe #(         
      .sld_auto_instance_index ("YES"),
      .sld_instance_index      (0),
      .instance_id             ("CALC"),
      .probe_width             (33),
      .source_width            (0),
      .source_initial_value    ("0"),
      .enable_metastability    ("NO")
      ) cal_counter_issp (
      .probe  ({done, clk_counter[31:0]})
   );
`endif

endmodule
`ifdef QUESTA_INTEL_OEM
`pragma questa_oem_00 "BUwlYfHqKCYPcXIP7xrnQ9Se5hc+WCW/arrBU5W+lFIz4wbefW0My5vX6xBCsRZeDFCX8Hl+sMTGZJoji0A9oHt2LftULu0FLRu+mk51alTJ3uomtGywN+PFSqf/fgMyVODG1WmlMI935DQeVtY8c0/LTaQ2hClpkazdo6IufJnkztEP0APWCavclOK3sEzCw6ySG+devOHMd72zKrOr9Hnpx+hVpDg9hFG/o4PmuuE4TTI7BwAqis83K9Zmh9ygM1chMyvIadmgw+9xq6/x8qW4Cr7p1/934jYMssppFaBTwo5gei5er4OXIKcXOr0oqiHzZ+nFJTOf7BYsA+EaroXbQpKOB+dwaK7KmkGp+edSIfrnKBQHgQctG3g7UgpfBTbT5KNI3iT9ydqIr31hso5kIOZYXdAWzm9nJl7hpoh+0533yV6g/vxRHKqYJ1HKL19hAr+f4AwzmDD5B+dsFV89G882sUdr8EphoCLUZapVj1XoBxb8MnwRpa2pAxHaUOweSQiDByeQGzDad2RE5p6VmPoWUyMbRU2BkaGoRMJhJ4kLqgaUa5tuBMQtY9sALt5OIxwgMcZxeggh8Wc/m2/iNeansIZVyt1zNGLtOHYIZ1YgBcwxWCoEduO2Af0DL7iC7NRyeS2c8O365qk8Wx+Kfw9eVcHZCN8+XV1RQjRGWJAGNOv4X49vmv02cT2NtQVdyXg5YZ4mQ39wOhArqVo0AMUUXts5fLRYkhrPeo5wXHeMtJQp27iJAqp1BwAn3wLaJk+6bagMwB2l2X/fYDxw7mzvXLMVHkGqnkZ34o2hLNarwaSySaws0LZCkYVtSA1ITR8SJmUNrKgV33583EelL/Cy57kh8av3CSruXLpyUAwdAokNtZ2vNk0VTL+MBKLI3KWnBTVTFHyBGPLHIcwsh/OFKWKWiI0UanCBTYqacVWRjP9yzGjpEN38gXny5RIqVYu3jy8sTdanel6LWl5q2AJFEbL69kmGvFjugakq6DzuHfgG1GEVHwu8P+XD"
`endif