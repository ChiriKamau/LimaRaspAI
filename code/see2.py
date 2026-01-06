import cv2
import numpy as np
import tflite_runtime.interpreter as tflite
import os  # <--- Added os library to handle folders

# --- CONFIGURATION ---
MODEL_PATH = "/home/lima/LimaRaspAI/Quantized/kagglemodelf16_quant.tflite"
IMAGE_PATH = "/home/lima/LimaRaspAI/Images/image1.jpg"
CONFIDENCE_THRESHOLD = 0.40

# ⚠️ UPDATE: Set your specific classes here
# If the model detects 'Green' as 'Ripe', swap these two words!
CLASSES = ["Green", "Ripe"] 

# --- LOAD MODEL ---
print(f"Loading model: {MODEL_PATH}...")
interpreter = tflite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_h = input_details[0]['shape'][1]
input_w = input_details[0]['shape'][2]

# --- PREPROCESS ---
image = cv2.imread(IMAGE_PATH)
if image is None:
    print(f"Error: Could not read image from {IMAGE_PATH}")
    exit()

original_h, original_w = image.shape[:2]

# Resize
image_resized = cv2.resize(image, (input_w, input_h))
input_data = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
input_data = np.expand_dims(input_data, axis=0)

# Quantization Handling
if input_details[0]['dtype'] == np.int8:
    scale, zero_point = input_details[0]['quantization']
    input_data = (np.float32(input_data) / 255.0)
    input_data = (input_data / scale + zero_point)
    input_data = np.clip(input_data, -128, 127).astype(np.int8)
else:
    input_data = (np.float32(input_data) / 255.0)

# --- INFERENCE ---
interpreter.set_tensor(input_details[0]['index'], input_data)
interpreter.invoke()
output_data = interpreter.get_tensor(output_details[0]['index'])

# De-quantize output
if output_details[0]['dtype'] == np.int8:
    scale, zero_point = output_details[0]['quantization']
    output_data = (output_data.astype(np.float32) - zero_point) * scale

# Transpose output if needed [1, 6, 2100] -> [1, 2100, 6]
if output_data.shape[1] < output_data.shape[2]:
    output_data = np.transpose(output_data, (0, 2, 1))

# --- DRAWING LOOP ---
boxes = output_data[0]
count = 0

# Auto-detect coordinate format
is_normalized = boxes[0][0] < 2.0 
if is_normalized:
    print("ℹ️ Coordinates are NORMALIZED (0-1). Scaling up...")
else:
    print("ℹ️ Coordinates are PIXELS. Scaling relative to input size...")

for box in boxes:
    scores = box[4:]
    max_score = np.max(scores)
    
    if max_score > CONFIDENCE_THRESHOLD:
        class_id = np.argmax(scores)
        
        # Safety check for class index
        if class_id >= len(CLASSES):
            continue 

        # Get Raw Box Coordinates
        cx, cy, w, h = box[0], box[1], box[2], box[3]
        
        # Scale Coordinates
        if is_normalized:
            cx = cx * original_w
            cy = cy * original_h
            w = w * original_w
            h = h * original_h
        else:
            scale_x = original_w / input_w
            scale_y = original_h / input_h
            cx = cx * scale_x
            cy = cy * scale_y
            w = w * scale_x
            h = h * scale_y
        
        # Convert to Top-Left
        x1 = int(cx - (w / 2))
        y1 = int(cy - (h / 2))
        x2 = int(cx + (w / 2))
        y2 = int(cy + (h / 2))
        
        # Clamp to image boundaries
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(original_w, x2)
        y2 = min(original_h, y2)
        
        # Set Color: Green for 'Green', Red for 'Ripe'
        # Note: BGR format (Blue, Green, Red)
        color = (0, 255, 0) # Default Green
        if CLASSES[class_id] == "Ripe":
            color = (0, 0, 255) # Red for Ripe
        
        # Draw Box and Label
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 3)
        
        label = f"{CLASSES[class_id]}: {int(max_score * 100)}%"
        (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(image, (x1, y1 - 20), (x1 + text_w, y1), color, -1)
        cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        count += 1

# --- SAVE RESULT ---
if count > 0:
    image_dir = os.path.dirname(IMAGE_PATH)
    output_dir = os.path.join(image_dir, "inferenced")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    original_filename = os.path.basename(IMAGE_PATH)
    filename_only, ext = os.path.splitext(original_filename)
    new_filename = f"{filename_only}_result2{ext}"
    save_path = os.path.join(output_dir, new_filename)
    
    cv2.imwrite(save_path, image)
    print(f"✅ Success! Saved {count} detection(s) to: {save_path}")
else:
    print("❌ No detections above threshold.")