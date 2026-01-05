import cv2
import numpy as np
import tflite_runtime.interpreter as tflite
import time

# --- CONFIGURATION ---
MODEL_PATH = "/home/lima/LimaRaspAI/Quantized/model_quant.tflite"
IMAGE_PATH = "/home/lima/LimaRaspAI/Images/image1.jpg"  # Ensure you have an image here!
CONFIDENCE_THRESHOLD = 0.5

# --- LOAD MODEL ---
print(f"Loading model: {MODEL_PATH}...")
interpreter = tflite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Get target size (e.g., 640x640)
input_height = input_details[0]['shape'][1]
input_width = input_details[0]['shape'][2]
print(f"Model expects input: {input_width}x{input_height}")

# --- PREPROCESS IMAGE ---
image = cv2.imread(IMAGE_PATH)
if image is None:
    print("Error: Image not found!")
    exit()

original_h, original_w = image.shape[:2]

# Resize to match model input
input_image = cv2.resize(image, (input_width, input_height))
input_image = np.expand_dims(input_image, axis=0)

# Check if model needs Float (0-1) or Int (0-255)
input_type = input_details[0]['dtype']
if input_type == np.float32:
    print("Normalizing input to 0-1 (Float32)")
    input_image = (np.float32(input_image) / 255.0)
else:
    print("Keeping input as Integer (Uint8)")
    # Note: Sometimes int8 models still need float inputs depending on export settings
    # If the output looks junk, we might need to change this line.

# --- INFERENCE ---
interpreter.set_tensor(input_details[0]['index'], input_image)

start_time = time.time()
interpreter.invoke()
end_time = time.time()

# --- RAW OUTPUT ---
output_data = interpreter.get_tensor(output_details[0]['index'])

print("\n" + "="*30)
print(f"Inference Time: {(end_time - start_time) * 1000:.2f} ms")
print(f"Output Shape: {output_data.shape}")
print("="*30)

# If shape is [1, 84, 8400] -> It's standard YOLOv8
# We need to transpose it to [1, 8400, 84] to parse boxes easier
if output_data.shape[1] < output_data.shape[2]:
    print("Output format is [Channels, Anchors]. Transposing...")
    output_data = np.transpose(output_data, (0, 2, 1))

# Simple check to see if we detected anything
# This just prints the first valid detection it finds
boxes = output_data[0]
for box in boxes:
    # box format: [x, y, w, h, class1_score, class2_score...]
    scores = box[4:] 
    max_score = np.max(scores)
    if max_score > CONFIDENCE_THRESHOLD:
        class_id = np.argmax(scores)
        print(f"Found Class {class_id} with confidence {max_score:.2f}")