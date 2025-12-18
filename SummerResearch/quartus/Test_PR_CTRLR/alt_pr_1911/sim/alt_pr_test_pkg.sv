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



package alt_pr_test_pkg;

    // Global variable for prblock interface reference
    class twentynm_prblock_if_mgr;
        virtual twentynm_prblock_if if_ref;
    
        static twentynm_prblock_if_mgr obj_single;
    
        local function new ();
        endfunction
        
        function void set_if_reference (virtual twentynm_prblock_if if_ref);
            this.if_ref = if_ref;
        endfunction

        function virtual twentynm_prblock_if get_if_reference (virtual twentynm_prblock_if if_ref);
            return this.if_ref;
        endfunction

        static function twentynm_prblock_if_mgr get ();
            if (obj_single == null) begin
                obj_single = new();
            end
            
            return obj_single;
        endfunction
    endclass


endpackage
`ifdef QUESTA_INTEL_OEM
`pragma questa_oem_00 "mXd8TZA7wcZD1M13kc63+RPfQcbqACFa9i0aqHtSQ62nFR2tUIuo2NzwpI/4cEyddIaPImJhX2CebOmYVV5f401JpALFknks4OMLrt70wgtKfnPjlusCxrHWEBHooOs6pkJXBTjC6oUhD2WVo05l5UBMB3Bob+qvd+BE/CfQQqbe8vYxUQkLM+Zgo0aA7Fpi0A6ZVpcWA/DDjPpZK8kPgz9Vfztbmx0x1RvFi9qEstLcJQjLk3yu2QyJaS7Lt0CZ6FoNc+1JdFQR8jmxjbv58e3dClAu3MQ4s3ulZdi0iSSfcumxv/kJYR7m7v1PEWRIfCPfO9DbJ67H0ZIlD56Z9tX1bm6/IZGZ2XtKrCN2WIAHBebvmrQb0cXbXWZ6b6PjLoAAc7ZmHNrcDa5n0SYfolQuR6XTLQrHKg31CjnMtkUEHqpV4POICBBWuBO39J/xLwkBWBy0FtGC/5j2L+kJ0uQnUR3bt3Z5PuWyFAs5axKRZKEUHpbQ+KNJMFbydO0lej0DUmQBhf4WiIfi9N6jXmjp5M3bBlRuWD8xg/ivhuA6czldKkOY7RrObXcaaHmsvoip2TwarItMSUqr3buwarQwzVHN806L35pwmrJicneuf6Ke3GzyPPD5mNJzcfL9V998bdymJeuzh8/120gbKelwTIbIHgPwigC0+NxsD/f2UlLT2/ZsnwCymnDRvZmuTC415K/daIqzySXMT1yQjpmOt0HGAitWyCeyBXqt1m05NBLATB+6R4k/tMGYmzvvCBXSP8zW8ZsNDx8VXVAY3OJ01d/WCJj7esZEPXpQMieEeLDXHexWs+PsozH1x364ZnSxPq/E+gZqwCQMrf4cOO6M8vpjUsGQ1GqK2oDIaBHYwB9rRa6aHPsPGkZg4n5aFGGPFIRFqkXY9wVhTUTUuXycv/60T5eqrkJ4qoXQGmjz/xbyUh37ybOLdOBzmYc89aoC7974MrV4/MqDCr6jJeW2WL+gEHWHWrfAW4JWw5PF1dH6gB6UfuqGJZkRon8t"
`endif