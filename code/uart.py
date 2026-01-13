import serial

# Changed /dev/ttyUSB0 to /dev/serial0
# /dev/serial0 points to the GPIO 14 (TX) and 15 (RX) pins
ser = serial.Serial('/dev/serial0', 9600, timeout=1)

print("Listening on GPIO Pins 14 (TX) and 15 (RX)...")

try:
    while True:
        if ser.in_waiting > 0:
            line = ser.readline().decode('utf-8', errors='ignore').strip()
            if line:
                print("Received:", line)
except KeyboardInterrupt:
    print("\nClosing Serial Port.")
    ser.close()