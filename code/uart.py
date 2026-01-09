import serial

ser = serial.Serial('/dev/ttyUSB0', 9600, timeout=1)

print("Listening...")

while True:
    line = ser.readline().decode().strip()
    if line:
        print("Received:", line)
