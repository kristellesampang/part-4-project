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


module altera_emif_arch_nf_buf_unused (
   output logic o
);
   timeunit 1ns;
   timeprecision 1ps;

   assign o = 1'b0;
endmodule

`ifdef QUESTA_INTEL_OEM
`pragma questa_oem_00 "BUwlYfHqKCYPcXIP7xrnQ9Se5hc+WCW/arrBU5W+lFIz4wbefW0My5vX6xBCsRZeDFCX8Hl+sMTGZJoji0A9oHt2LftULu0FLRu+mk51alTJ3uomtGywN+PFSqf/fgMyVODG1WmlMI935DQeVtY8c0/LTaQ2hClpkazdo6IufJnkztEP0APWCavclOK3sEzCw6ySG+devOHMd72zKrOr9Hnpx+hVpDg9hFG/o4PmuuEUoEitcMYlx3h2rzMvlqG4rXi8O0ZJV9LXVBY6qymZbVWcYLln5JBIxXYT1VKcS7XCAmJKL7RuzmoMe+/8ZfJoRIW0DQR0N08G0gdzdewCwxa4WkCG/W5kPhS0aE2yACH+cTqj4aX3zVCNMPeJMagO19NJy9fFQFdbQXWVneIaIJO24WUojEFd+xsk+BE9dBz8//AyUqUG+0vcoRhhtTxZNI3a5gFZTmlw8IP55rp6R+UYj7JMFbja+ihLgRd9dl0TFcYdzSdAXrjjrnFNwV1zNyTBd9kvZzbkBt69YYSGqNFHeZKRRQT5IrXEjCrJdl0xgf5IXyT4D8lwvmz7HL6yG4kM9867lx13D9Md/dXzg6XgS2z8AdSNr36gHrUr2aYoTd9RT78BWek9ZW6UrdT4sIAG0xcyyLx0da/6ZhzzJW7XifzegC0YN4gsmTG7Nj0pZxkaTHKYspyc+ZqP/MphTRXZ/y7uoY4Z/VGOq7kbwRvbb7n0RfRr67ymgf9eQebw9iALEshmjdJ4aEj9UFClymBNZGzSOvBlxRfN3ug4v4iWFPtV9ee6yGYTnyPhTCnvfGawCHMZIUiZ5GgSihAw1s2wtFG/3mbrokIB3chcPjav0BMf7F1QxA+ouqz8JMC6I2jmlPyFWe154dPD/7NNM5lu7/mCWsQmmvPxTo6KfN5O5tg0GeKKz3ewgXynevUtnhKtfkak5+/ViHsbenEB862VIKq/6wKZRuwp+/nIsX46qALpnNbzVocZPTlWEjpA7/FtJ7pao9dIvrxaSzmL"
`endif