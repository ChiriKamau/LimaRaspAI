import serial
import time
from datetime import datetime
import os
import json

# ================== SERIAL CONFIG ==================
SERIAL_PORT = '/dev/ttyUSB0'  # Update if different
BAUD_RATE = 9600

ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
time.sleep(2)  # Wait for serial to initialize

# ================== DATA FOLDER ==================
DATA_FOLDER = os.path.expanduser('~/analog_data')
if not os.path.exists(DATA_FOLDER):
    os.makedirs(DATA_FOLDER)

print("Listening for Pico data...")

def get_unique_filename(base_folder, base_name):
    """
    Returns a unique file path. If base_name.json exists, appends _1, _2, etc.
    """
    file_path = os.path.join(base_folder, f"{base_name}.json")
    counter = 1
    while os.path.exists(file_path):
        file_path = os.path.join(base_folder, f"{base_name}_{counter}.json")
        counter += 1
    return file_path

# Keep track of today's file
current_date = datetime.now().strftime("%Y-%m-%d")
current_file = get_unique_filename(DATA_FOLDER, current_date)

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
                record = {
                    "timestamp": timestamp,
                    "airTemp": float(airTemp),
                    "airHumidity": float(airHum),
                    "soilTempPct": int(soilTemp),
                    "m1Pct": int(m1),
                    "m2Pct": int(m2),
                    "gp0Active": int(gp0),
                    "gp1Active": int(gp1),
                    "valveActive": int(valve)
                }
                
                # Print nicely
                print(f"[{timestamp}] Air: {record['airTemp']}C / {record['airHumidity']}% | "
                      f"Soil T: {record['soilTempPct']}% | M1: {record['m1Pct']}% | "
                      f"M2: {record['m2Pct']}% | Pump1: {record['gp0Active']} | "
                      f"Pump2: {record['gp1Active']} | Valve: {record['valveActive']}")
                
                # Check if day has changed
                today_date = datetime.now().strftime("%Y-%m-%d")
                if today_date != current_date:
                    current_date = today_date
                    current_file = get_unique_filename(DATA_FOLDER, current_date)
                
                # Save to JSON file (append)
                with open(current_file, 'a') as f:
                    f.write(json.dumps(record) + '\n')
                
            else:
                print(f"[{timestamp}] Raw: {line}")
                
        except Exception as e:
            print(f"Error processing data: {e}")
            
    time.sleep(0.01)
