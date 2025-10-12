# File: main.py
# Description: A script to read bytes from UART and print their binary and decimal values.

import sys
import machine
import select

# Use UART 0, which is on pins GP0 (TX) and GP1 (RX)
# Baud rate must match the VHDL: 9600
uart = machine.UART(0, baudrate=9600)

# A poll object to check for incoming data without blocking
poll_obj = select.poll()
poll_obj.register(uart, select.POLLIN)

print("Pico binary/decimal UART listener is running.")
print("Press KEY[1] on the FPGA to send data.")

# Counter for printed lines
print_counter = 0

# Loop forever
while True:
    # Check if there is any data waiting on the UART RX pin
    if poll_obj.poll(1):
        # Read all available bytes
        data_bytes = uart.read()
        
        if data_bytes:
            # For each byte received...
            for byte in data_bytes:
                # Increment counter
                print_counter += 1
                
                
                # Convert to an 8-bit binary string
                binary_string = f'{byte:08b}'
                # The byte variable already holds the decimal (integer) value
                decimal_value = byte
                
                # Print both values with counter
                print(f"[{print_counter}] Received: Decimal = {decimal_value:3}, Binary = {binary_string}")

