import intel_jtag_uart
import time

ju = intel_jtag_uart.intel_jtag_uart()
ju.write(b"AAAA\n")
print("Sent: AAAA")

start = time.time()
response = b''
while len(response) == 0:
    if time.time() - start > 5:
        print("TIMEOUT")
        break
    response = ju.read()
    time.sleep(0.1)

print(f"Received: {response}")