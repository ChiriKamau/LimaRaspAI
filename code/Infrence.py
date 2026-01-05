import cv2
import numpy as np
import tflite_runtime.interpreter as tflite
import time

# --- CONFIGURATION ---
MODEL_PATH = "/home/lima/LimaRaspAI/Quantized/model_quant.tflite"
IMAGE_PATH = "/home/lima/LimaRaspAI/Images/image1.jpg"
CONFIDENCE_THRESHOLD = 0.10 # Super low just to see ANY spark of life

# --- LOAD MODEL ---
print(f"Loading model: {MODEL_PATH}...")
interpreter = tflite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_shape = input_details[0]['shape']
input_height = input_shape[1]
input_width = input_shape[2]

# --- PREPROCESS IMAGE ---
image = cv2.imread(IMAGE_PATH)
if image is None:
    print("Error: Image not found!")
    exit()

# 1. FIX COLOR (BGR -> RGB)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# 2. RESIZE
input_image = cv2.resize(image, (input_width, input_height))
input_image = np.expand_dims(input_image, axis=0)

# 3. QUANTIZE INPUT
input_type = input_details[0]['dtype']
if input_type == np.int8:
    scale, zero_point = input_details[0]['quantization']
    print(f"Input: INT8 (Scale: {scale}, ZP: {zero_point})")
    input_image = (np.float32(input_image) / 255.0)
    input_image = (input_image / scale + zero_point)
    input_image = np.clip(input_image, -128, 127).astype(np.int8) # Clip is safe
else:
    print("Input: Float32")
    input_image = (np.float32(input_image) / 255.0)

# --- INFERENCE ---
interpreter.set_tensor(input_details[0]['index'], input_image)
interpreter.invoke()
output_data = interpreter.get_tensor(output_details[0]['index'])

# --- DE-QUANTIZE OUTPUT ---
output_type = output_details[0]['dtype']
if output_type == np.int8:
    scale, zero_point = output_details[0]['quantization']
    print(f"Output: INT8 (De-quantizing with Scale: {scale}, ZP: {zero_point})")
    output_data = (output_data.astype(np.float32) - zero_point) * scale

# Transpose [1, 6, 2100] -> [1, 2100, 6]
if output_data.shape[1] < output_data.shape[2]:
    output_data = np.transpose(output_data, (0, 2, 1))

# --- DEBUG REPORT ---
boxes = output_data[0]
scores = boxes[:, 4:] # All scores for all boxes
max_score = np.max(scores)
best_class = np.argmax(boxes[:, 4:]) % scores.shape[1]

print("\n" + "-"*30)
print(f"DEBUG REPORT:")
print(f"Highest Confidence: {max_score:.5f}")
print(f"Best Class Index: {best_class}")
print("-"*30)

if max_score == 0.0:
    print("❌ Model is still blind. The issue is likely the Export Calibration in Colab.")
else:
    print("✅ Model is seeing something!")