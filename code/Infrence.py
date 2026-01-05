import cv2
import numpy as np
import tflite_runtime.interpreter as tflite
import time

# --- CONFIGURATION ---
MODEL_PATH = "/home/lima/LimaRaspAI/Quantized/model_quant.tflite"
IMAGE_PATH = "/home/lima/LimaRaspAI/Images/image1.jpg"
CONFIDENCE_THRESHOLD = 0.5

# --- LOAD MODEL ---
print(f"Loading model: {MODEL_PATH}...")
interpreter = tflite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Get target size
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
    # Standard Float Model
    print("Normalizing input to 0-1 (Float32)")
    input_image = (np.float32(input_image) / 255.0)

elif input_type == np.int8:
    # Signed Integer Model (This fixes your error!)
    scale, zero_point = input_details[0]['quantization']
    print(f"Converting to INT8 (Scale: {scale}, Zero Point: {zero_point})")
    
    # 1. Normalize to 0-1
    input_image = (np.float32(input_image) / 255.0)
    # 2. Quantize: (Value / Scale) + ZeroPoint
    input_image = (input_image / scale + zero_point)
    # 3. Cast to INT8
    input_image = input_image.astype(np.int8)

else:
    # Fallback for UINT8 models
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
print(f"Output Shape: {output_data.shape}")
print("="*30)

# If shape is [1, 84, 8400] -> It's standard YOLOv8
# We need to transpose it to [1, 8400, 84] to parse boxes easier
if output_data.shape[1] < output_data.shape[2]:
    print("Output format is [Channels, Anchors]. Transposing...")
    output_data = np.transpose(output_data, (0, 2, 1))

# --- SIMPLE DETECTION CHECK ---
# Checks for the highest confidence in the whole image
boxes = output_data[0]
found_any = False

# We iterate a few boxes just to see if it works
for i, box in enumerate(boxes):
    if i > 500: break # Don't check everything yet, just a quick sample
    
    # box format: [x, y, w, h, class1_score, class2_score...]
    scores = box[4:] 
    max_score = np.max(scores)
    
    if max_score > CONFIDENCE_THRESHOLD:
        class_id = np.argmax(scores)
        print(f"Found Class {class_id} with confidence {max_score:.2f}")
        found_any = True

if not found_any:
    print("No objects detected above threshold.")