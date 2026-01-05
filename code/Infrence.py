import cv2
import numpy as np
import tflite_runtime.interpreter as tflite
import time

# --- CONFIGURATION ---
MODEL_PATH = "/home/lima/LimaRaspAI/Quantized/model_quant.tflite"
IMAGE_PATH = "/home/lima/LimaRaspAI/Images/image2.jpg"
CONFIDENCE_THRESHOLD = 0.25  # Lowered to standard YOLO threshold

# --- LOAD MODEL ---
print(f"Loading model: {MODEL_PATH}...")
interpreter = tflite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_height = input_details[0]['shape'][1]
input_width = input_details[0]['shape'][2]
print(f"Model expects input: {input_width}x{input_height}")

# --- PREPROCESS IMAGE ---
image = cv2.imread(IMAGE_PATH)
if image is None:
    print(f"Error: Image not found at {IMAGE_PATH}")
    exit()

# Resize to match model input
input_image = cv2.resize(image, (input_width, input_height))
input_image = np.expand_dims(input_image, axis=0)

# CHECK DATA TYPE AND CONVERT
input_type = input_details[0]['dtype']

if input_type == np.float32:
    print("Normalizing input to 0-1 (Float32)")
    input_image = (np.float32(input_image) / 255.0)

elif input_type == np.int8:
    # Signed Integer Model Logic
    scale, zero_point = input_details[0]['quantization']
    print(f"Converting to INT8 (Scale: {scale}, Zero Point: {zero_point})")
    
    input_image = (np.float32(input_image) / 255.0)
    input_image = (input_image / scale + zero_point)
    input_image = input_image.astype(np.int8)

else:
    print("Keeping input as Integer (Uint8)")
    input_image = input_image.astype(np.uint8)

# --- INFERENCE ---
interpreter.set_tensor(input_details[0]['index'], input_image)

start_time = time.time()
interpreter.invoke()
end_time = time.time()

# --- RAW OUTPUT ---
output_data = interpreter.get_tensor(output_details[0]['index'])

print("\n" + "="*30)
print(f"Inference Time: {(end_time - start_time) * 1000:.2f} ms")
print("="*30)

# Transpose if needed: [1, 6, 2100] -> [1, 2100, 6]
if output_data.shape[1] < output_data.shape[2]:
    print("Transposing output shape...")
    output_data = np.transpose(output_data, (0, 2, 1))

# --- DETECTION LOOP ---
boxes = output_data[0]
max_confidence_seen = 0.0
best_class_seen = -1
found_count = 0

for box in boxes:
    # box format: [x, y, w, h, class1_score, class2_score]
    scores = box[4:] 
    current_max = np.max(scores)
    
    # Track stats for debugging
    if current_max > max_confidence_seen:
        max_confidence_seen = current_max
        best_class_seen = np.argmax(scores)

    if current_max > CONFIDENCE_THRESHOLD:
        class_id = np.argmax(scores)
        
        # Convert xywh (normalized) to pixels
        cx = box[0] # * input_width (if model outputs normalized coords)
        cy = box[1] # * input_height
        w = box[2]  # * input_width
        h = box[3]  # * input_height
        
        # Note: Some quantized models output raw pixels (0-320), some output normalized (0-1).
        # If your output values are small (like 0.5), uncomment the multiplication above.
        
        print(f"✅ FOUND Class {class_id} | Conf: {current_max:.2f} | Center: ({int(cx)}, {int(cy)})")
        found_count += 1

print("\n" + "-"*30)
print(f"DEBUG REPORT:")
print(f"Total Detections > {CONFIDENCE_THRESHOLD}: {found_count}")
print(f"Highest Confidence seen in ANY box: {max_confidence_seen:.4f}")
print(f"Best Guess Class: {best_class_seen}")
print("-"*30 + "\n")

if max_confidence_seen < 0.1:
    print("⚠️ WARNING: The model is effectively blind (Max Conf < 10%).")
    print("Likely causes: Normalization mismatch or bad Quantization.")