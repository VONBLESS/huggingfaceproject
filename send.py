# send_epoch.py
import time, serial

PORT = "COM5"          # your Arduino port
BAUD = 115200

with serial.Serial(PORT, BAUD, timeout=2) as ser:
    time.sleep(2)  # let Arduino reset after opening port
    epoch = int(time.time())  # current UNIX time in seconds (UTC)
    line = f"EPOCH:{epoch}\n"
    ser.write(line.encode("ascii"))
    print("Sent:", line.strip())
