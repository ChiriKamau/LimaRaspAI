import cv2
import numpy as np
import tflite_runtime.interpreter as tflite
import os  # <--- Added os library to handle folders

# --- CONFIGURATION ---
MODEL_PATH = "/home/lima/LimaRaspAI/Quantized/kagglemodelf16_quant.tflite"
IMAGE_PATH = "/home/lima/LimaRaspAI/Images/image1.jpg"
CONFIDENCE_THRESHOLD = 0.30

NMS_THRESHOLD = 0.35  # Overlap threshold (lower = stricter)
CLASSES = ["Ripe", "Green"] 

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
if image is None: exit(f"Error reading {IMAGE_PATH}")
original_h, original_w = image.shape[:2]

# Resize & Normalize
input_data = cv2.resize(image, (input_w, input_h))
input_data = cv2.cvtColor(input_data, cv2.COLOR_BGR2RGB)
input_data = np.expand_dims(input_data, axis=0)

if input_details[0]['dtype'] == np.int8:
    scale, zero_point = input_details[0]['quantization']
    input_data = (np.float32(input_data) / 255.0 / scale + zero_point).astype(np.int8)
else:
    input_data = (np.float32(input_data) / 255.0)

# --- INFERENCE ---
interpreter.set_tensor(input_details[0]['index'], input_data)
interpreter.invoke()
output_data = interpreter.get_tensor(output_details[0]['index'])

# De-quantize & Transpose
if output_details[0]['dtype'] == np.int8:
    scale, zero_point = output_details[0]['quantization']
    output_data = (output_data.astype(np.float32) - zero_point) * scale

if output_data.shape[1] < output_data.shape[2]: # [1, 6, 2100] -> [1, 2100, 6]
    output_data = np.transpose(output_data, (0, 2, 1))

# --- PARSE BOXES ---
boxes_list = []
confidences_list = []
class_ids_list = []

predictions = output_data[0]
is_normalized = predictions[0][0] < 2.0 # Auto-detect format

for pred in predictions:
    scores = pred[4:]
    max_score = np.max(scores)
    
    if max_score > CONFIDENCE_THRESHOLD:
        class_id = np.argmax(scores)
        cx, cy, w, h = pred[0], pred[1], pred[2], pred[3]
        
        # Scale coordinates
        if is_normalized:
            cx, cy, w, h = cx*original_w, cy*original_h, w*original_w, h*original_h
        else:
            scale_x, scale_y = original_w/input_w, original_h/input_h
            cx, cy, w, h = cx*scale_x, cy*scale_y, w*scale_x, h*scale_y

        # Convert to Top-Left (x, y)
        x = int(cx - (w / 2))
        y = int(cy - (h / 2))
        
        boxes_list.append([x, y, int(w), int(h)])
        confidences_list.append(float(max_score))
        class_ids_list.append(class_id)

# --- APPLY NMS (REMOVE DUPLICATES) ---
indices = cv2.dnn.NMSBoxes(boxes_list, confidences_list, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)

# --- DRAW RESULTS ---
if len(indices) > 0:
    for i in indices.flatten():
        x, y, w, h = boxes_list[i]
        label = CLASSES[class_ids_list[i]]
        conf = confidences_list[i]
        
        # Color: Red for Ripe, Green for Green
        color = (0, 0, 255) if label == "Ripe" else (0, 255, 0)

        cv2.rectangle(image, (x, y), (x + w, y + h), color, 3)
        
        # Text Label
        text = f"{label} {int(conf * 100)}%"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(image, (x, y - 25), (x + tw, y), color, -1)
        cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Save
    save_dir = os.path.join(os.path.dirname(IMAGE_PATH), "inferenced")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{os.path.splitext(os.path.basename(IMAGE_PATH))[0]}_2result.jpg")
    cv2.imwrite(save_path, image)
    print(f"✅ Saved {len(indices)} clean detections to: {save_path}")

else:
    print("❌ No objects detected.")