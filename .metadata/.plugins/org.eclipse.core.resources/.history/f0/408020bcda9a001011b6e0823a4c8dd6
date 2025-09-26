#include <stdio.h>
#include <alt_stdio.h> // For the JTAG UART functions

int main()
{
  // This line flushes the output buffer to ensure messages appear immediately.
  // It's good practice for debugging.
  alt_putstr("INFO: NIOS II Echo Server Running.\n");

  char received_char;

  // This is an infinite loop that forms the core of our server.
  while (1)
  {
    // alt_getchar() will wait here patiently until a character is
    // received from the PC via the JTAG UART.
    received_char = alt_getchar();

    // As soon as a character is received, alt_putchar() sends it
    // right back to the PC.
    alt_putchar(received_char);
  }

  return 0;
}
