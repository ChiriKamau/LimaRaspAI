import cv2
import numpy as np
import tflite_runtime.interpreter as tflite

# --- CONFIGURATION ---
MODEL_PATH = "/home/lima/LimaRaspAI/Quantized/model_quant.tflite"
IMAGE_PATH = "/home/lima/LimaRaspAI/Images/image1.jpg"
CONFIDENCE_THRESHOLD = 0.40  # We know your score is ~0.63, so 0.40 is safe
CLASSES = ["Tomato", "Fruit_2", "Fruit_3"] # ⚠️ UPDATE THIS with your real class names!

# --- LOAD MODEL ---
interpreter = tflite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_h = input_details[0]['shape'][1]
input_w = input_details[0]['shape'][2]

# --- PREPROCESS ---
image = cv2.imread(IMAGE_PATH)
original_h, original_w = image.shape[:2]

# Resize and Pad logic (Simplified resize)
image_resized = cv2.resize(image, (input_w, input_h))
input_data = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
input_data = np.expand_dims(input_data, axis=0)

# Quantization handling
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

# De-quantize output if needed
if output_details[0]['dtype'] == np.int8:
    scale, zero_point = output_details[0]['quantization']
    output_data = (output_data.astype(np.float32) - zero_point) * scale

# Transpose [1, 6, 2100] -> [1, 2100, 6]
if output_data.shape[1] < output_data.shape[2]:
    output_data = np.transpose(output_data, (0, 2, 1))

# --- DRAWING LOOP ---
boxes = output_data[0]
count = 0

for box in boxes:
    scores = box[4:]
    max_score = np.max(scores)
    
    if max_score > CONFIDENCE_THRESHOLD:
        class_id = np.argmax(scores)
        
        # Get Box Coordinates (Normalized 0-1 or Pixels)
        # YOLOv8 export usually gives center_x, center_y, width, height
        cx, cy, w, h = box[0], box[1], box[2], box[3]
        
        # If coordinates are normalized (0-1), multiply by image size
        # If they are already pixels (0-320), we just scale up to original image
        
        # Let's assume they are absolute pixels relative to the 320x320 input
        # We need to scale them back to the ORIGINAL image size
        scale_x = original_w / input_w
        scale_y = original_h / input_h
        
        cx = cx * scale_x
        cy = cy * scale_y
        w = w * scale_x
        h = h * scale_y
        
        # Convert Center-format to Top-Left format for OpenCV
        x1 = int(cx - (w / 2))
        y1 = int(cy - (h / 2))
        x2 = int(cx + (w / 2))
        y2 = int(cy + (h / 2))
        
        # Draw Rectangle
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Draw Label
        label = f"{CLASSES[class_id]}: {int(max_score * 100)}%"
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        count += 1

# --- SAVE RESULT ---
if count > 0:
    cv2.imwrite("detection_result.jpg", image)
    print(f"✅ Success! Saved {count} detection(s) to 'detection_result.jpg'")
else:
    print("❌ No detections above threshold.")