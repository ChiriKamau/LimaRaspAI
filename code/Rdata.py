import serial
import time
from datetime import datetime

# Port setup
try:
    ser = serial.Serial('/dev/serial0', baudrate=9600, timeout=1)
    print("UART Listener Started. Waiting for Pico...")
except Exception as e:
    print(f"Error: {e}")
    exit()

while True:
    if ser.in_waiting > 0:
        try:
            # Read line and decode
            line = ser.readline().decode('utf-8').strip()
            
            # Create Timestamp
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # Split comma-separated values
            data = line.split(',')
            
            if len(data) == 5:
                at, ah, st, m1, m2 = data
                print(f"[{timestamp}] Air: {at}C/{ah}% | Soil T: {st}% | M1: {m1}% | M2: {m2}%")
            else:
                print(f"[{timestamp}] Raw: {line}")
                
        except Exception as e:
            print(f"Error processing data: {e}")
            
    time.sleep(0.01)