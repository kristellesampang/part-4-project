import intel_jtag_uart
import numpy as np
import struct
import time

ju = intel_jtag_uart.intel_jtag_uart()

M, N, K = 4, 4, 4
data = np.ones(16, dtype=np.int32).reshape((M, K))
weight = np.ones(16, dtype=np.int32).reshape((K, N))

# Send header
ju.write(struct.pack('BBB', M, N, K))
time.sleep(0.5)

# Send each int32 one at a time with a gap
for val in data.flatten():
    ju.write(struct.pack('<i', int(val)))
    time.sleep(0.1)

for val in weight.flatten():
    ju.write(struct.pack('<i', int(val)))
    time.sleep(0.1)

# Wait for results
expected = M * N * 4
response = b''
start = time.time()
while len(response) < expected:
    if time.time() - start > 15:
        print("TIMEOUT")
        break
    chunk = ju.read()
    if chunk:
        response += chunk
    time.sleep(0.05)

if len(response) == expected:
    results = np.frombuffer(response, dtype=np.int32).reshape(M, N)
    sw = np.matmul(data.astype(np.int64), weight.astype(np.int64)).astype(np.int32)
    print("Hardware:", results)
    print("Software:", sw)
    print("Match:", np.array_equal(results, sw))