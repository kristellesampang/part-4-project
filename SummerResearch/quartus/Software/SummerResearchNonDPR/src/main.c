/*
 ============================================================================
 Name        : main.c
 Author      : Pratham Chhabra
 Version     :
 Copyright   : Your copyright notice
 Description : Hello RISC-V World in C
 ============================================================================
 */

//#include <stdio.h>
#include "system.h"
#include "altera_avalon_jtag_uart_regs.h"

#define NPU_CTRL_BASE    0x30000
#define NPU_REG_READY    (*(volatile unsigned int*)(NPU_CTRL_BASE + 0x00))
#define NPU_REG_M        (*(volatile unsigned int*)(NPU_CTRL_BASE + 0x04))
#define NPU_REG_N        (*(volatile unsigned int*)(NPU_CTRL_BASE + 0x08))
#define NPU_REG_K        (*(volatile unsigned int*)(NPU_CTRL_BASE + 0x0C))
#define DATA_MEM         ((volatile int*)DATA_MEM_BASE)
#define WEIGHT_MEM       ((volatile int*)WEIGHT_MEM_BASE)
#define OUT_MEM          ((volatile int*)OUT_MEM_BASE)

unsigned char read_byte(void) {
    unsigned int data;
    while(1) {
        data = IORD_ALTERA_AVALON_JTAG_UART_DATA(JTAG_UART_0_BASE);
        if (data & ALTERA_AVALON_JTAG_UART_DATA_RVALID_MSK)
            return (unsigned char)(data & ALTERA_AVALON_JTAG_UART_DATA_DATA_MSK);
    }
}

void write_byte(unsigned char c) {
    IOWR_ALTERA_AVALON_JTAG_UART_DATA(JTAG_UART_0_BASE, c);
}

int main(void) {
    while(1) {
        // Read header
        unsigned char m = read_byte();
        unsigned char n = read_byte();
        unsigned char k = read_byte();

        // Read data matrix (m*k * 4 bytes each)
        for(int i = 0; i < m * k; i++) {
            unsigned int val = 0;
            val |= (unsigned int)read_byte();
            val |= (unsigned int)read_byte() << 8;
            val |= (unsigned int)read_byte() << 16;
            val |= (unsigned int)read_byte() << 24;
            DATA_MEM[i] = (int)val;
        }

        // Read weight matrix (k*n * 4 bytes each)
        for(int i = 0; i < k * n; i++) {
            unsigned int val = 0;
            val |= (unsigned int)read_byte();
            val |= (unsigned int)read_byte() << 8;
            val |= (unsigned int)read_byte() << 16;
            val |= (unsigned int)read_byte() << 24;
            WEIGHT_MEM[i] = (int)val;
        }
       
        // Set dimensions and trigger
        NPU_REG_M = m;
        NPU_REG_N = n;
        NPU_REG_K = k;
        NPU_REG_READY = 1;

        // Wait for done
        while(NPU_REG_READY != 0);

        // Send results back
        for(int i = 0; i < m * n; i++) {
            int val = OUT_MEM[i];
            write_byte((val) & 0xFF);
            write_byte((val >> 8) & 0xFF);
            write_byte((val >> 16) & 0xFF);
            write_byte((val >> 24) & 0xFF);
        }
    }
    return 0;
}
