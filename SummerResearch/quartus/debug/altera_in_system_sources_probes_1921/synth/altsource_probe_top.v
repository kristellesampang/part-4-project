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


module altsource_probe_top
#(
    parameter lpm_type = "altsource_probe",     // required by the coding standard
    parameter lpm_hint = "UNUSED",              // required by the coding standard 

    parameter sld_auto_instance_index = "YES",  // Yes, if the instance index should be automatically assigned.
    parameter sld_instance_index = 0,           // unique identifier for the altsource_probe instance.
    parameter sld_node_info_parameter = 4746752 + sld_instance_index,   // The NODE ID to uniquely identify this node on the hub.  Type ID: 9 Version: 0 Inst: 0 MFG ID 110 -- ***NOTE*** this parameter cannot be called SLD_NODE_INFO or Quartus Standard will think it's an ISSP impl.
    parameter sld_ir_width = 4,                 
                                        
    parameter instance_id = "UNUSED",           // optional name for the instance.
    parameter probe_width = 1,                  // probe port width
    parameter source_width= 1,                  // source port width
    parameter source_initial_value = "0",       // initial source port value
    parameter enable_metastability = "NO"       // yes to add two register
)
(
    input [probe_width - 1 : 0] probe,       // probe inputs
    output [source_width - 1 : 0] source,    // source outputs
    input source_clk,                        // clock of the registers used to metastabilize the source output
    input tri1 source_ena                    // enable of the registers used to metastabilize the source output
);

    altsource_probe #(
        .lpm_type(lpm_type),
        .lpm_hint(lpm_hint),   
        .sld_auto_instance_index(sld_auto_instance_index),
        .sld_instance_index(sld_instance_index),
        .SLD_NODE_INFO(sld_node_info_parameter),
        .sld_ir_width(sld_ir_width),                          
        .instance_id(instance_id),
        .probe_width(probe_width),
        .source_width(source_width),
        .source_initial_value(source_initial_value),
        .enable_metastability(enable_metastability)
    )issp_impl
    (
        .probe(probe),
        .source(source),
        .source_clk(source_clk),
        .source_ena(source_ena)
    );
    
endmodule
`ifdef QUESTA_INTEL_OEM
`pragma questa_oem_00 "u7/YsPkyP82xbBDtLEznXhS5SZvzKn3nJ64FaEXxxPEjs1Fo1j8wgE/I9TM13dQVg8464Be69dsMv0DAMe5G1VIlCUUNtcrqhwWDc5W6Ndtpzr3rHC5Sad14rDbOEAu95K3wyZTcrJjxSeTiIpoTJ316ii+u+UP3Q9Q9IupM+pgo/Oxs6FGXsUnKNn8iqsbWVqNdMquBd1GvFeNneF2XE9KyYAZdobqoOqpVGjR/B72jmFti53wzjJ59gpZCA2vTh4ZeOQLL7l2FxNrtmVrSRPurFAzq6uvAZmKKv8+3yZg4OrSPje3ZBEqjTDKAJbXb5i+n3HHKK7JTwyVGAoTx4khXhlQAKwSG+r3nRh+g8dg5/4quU8Tu96jbRBNpvgH2G7xSn+C7eTW0jRX4dscQ2gnYyAdPZQeJUbWu6P+cpfoCDG8jPdkqD5RXdV89jiVIvMAHsaZcgiZH+xW64JVedXoevhOOTLGq7molPGt7ioxYiNhXIFbE5kGjVdFEGA8TsDtnBLc6f96xXSYSNA53esweuvx4l1kQF722xh0I+j+oWSj2DLiAxfwFg9vEXr+AfmNz6277daPHHWJHouSdYfYvwwsuljyT/BzYNIg0zphwob3fNDDigyFhRpMdpHp8/1OSbyqH7rMX+Pkf/LySxe6gdRis7OL/wcBw0FXlqdA1Cy2eziNP44+jgTI+j/L0Q8/ZkAiCyfd3UL6Pi32HZbhPXms0M7ZsR7cMfOb47QMnmSVoryba3JsCMSTRZLRTdyOg4QDWOsjF/s2dofN4cIa0wl40flf9r77QSqJ2a9l/sJllPMcnihcpEFM38VELSCDuQGRrN3m6LBSLQ25QVaNW2iA3iy0M0qy2OomQTtflJXjwmv50h44+gmQ2CD0BOtCpNqG1Rh1yNraE67QV+2rNJCMs8MgQkiQ+5YIHYmUZJxGYodqNaDFC7cdqyof5w83JNluXqChZk8rEJTr3x3hqmPoyThWPxwZfdWHR4qVJ3aLfrqJJhEhYdzGrihmZ"
`endif