import serial
import time

# /dev/serial0 is the default UART for GPIO pins 8 and 10
ser = serial.Serial('/dev/serial0', baudrate=9600, timeout=1)

print("Waiting for Pico...")

try:
    while True:
        if ser.in_waiting > 0:
            # Read the line from Pico
            line = ser.readline().decode('utf-8').rstrip()
            print(f"Pico sent: {line}")
            
            # Send a confirmation back
            ser.write("Pico, I heard you!\n".encode('utf-8'))
            
except KeyboardInterrupt:
    print("Closing connection.")
    ser.close()