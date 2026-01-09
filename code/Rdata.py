import serial
import time
from datetime import datetime

# ================== SERIAL CONFIG ==================
# Replace with your actual port from `ls /dev/serial/by-id/`
SERIAL_PORT = '/dev/ttyUSB0'
BAUD_RATE = 9600

ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
time.sleep(2)  # Wait for serial to initialize

print("Listening for Pico data...")

while True:
    if ser.in_waiting > 0:
        try:
            # Read line from Pico
            line = ser.readline().decode('utf-8').strip()
            
            # Timestamp
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # Split CSV fields
            data = line.split(',')
            
            if len(data) == 8:
                airTemp, airHum, soilTemp, m1, m2, gp0, gp1, valve = data
                
                # Convert numeric values
                airTemp = float(airTemp)
                airHum = float(airHum)
                soilTemp = int(soilTemp)
                m1 = int(m1)
                m2 = int(m2)
                gp0 = int(gp0)
                gp1 = int(gp1)
                valve = int(valve)
                
                # Print nicely
                print(f"[{timestamp}] Air: {airTemp}C / {airHum}% | Soil T: {soilTemp}% | "
                      f"M1: {m1}% | M2: {m2}% | Pump1: {gp0} | Pump2: {gp1} | Valve: {valve}")
            else:
                print(f"[{timestamp}] Raw: {line}")
                
        except Exception as e:
            print(f"Error processing data: {e}")
            
    time.sleep(0.01)
