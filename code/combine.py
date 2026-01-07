import cv2
import numpy as np
import tflite_runtime.interpreter as tflite
import os  # <--- Added os library to handle folders
# ==========================================
# --- CONFIGURATION ---
# ==========================================

# 1. PATHS
YOLO_MODEL_PATH = "/home/lima/LimaRaspAI/Quantized/Yolo/kagglemodelf16_quant.tflite"
RIPE_CNN_PATH   = "/home/lima/LimaRaspAI/Quantized/CNN/ripecnn_v2_quantized.tflite"
GREEN_CNN_PATH  = "/home/lima/LimaRaspAI/Quantized/CNN/greencnn_v2_quantized.tflite"
IMAGE_PATH = "/home/lima/LimaRaspAI/Images/image3.jpg"

# 2. LABELS (Swapped based on your previous finding)
YOLO_CLASSES = ["Green", "Ripe"] 

RIPE_CNN_CLASSES  = ['R_ber', 'R_healthy', 'R_spots'] 
GREEN_CNN_CLASSES = ['G_ber', 'G_healthy', 'G_lateblight', 'G_spots']

# 3. SETTINGS
CONFIDENCE_THRESHOLD = 0.20
NMS_THRESHOLD = 0.25 

# ==========================================
# --- HELPER FUNCTIONS ---
# ==========================================

def load_interpreter(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"❌ Model not found: {model_path}")
    interpreter = tflite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

def run_cnn_inference(interpreter, crop_img):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Auto-detect input size
    target_h = input_details[0]['shape'][1]
    target_w = input_details[0]['shape'][2]
    
    input_data = cv2.resize(crop_img, (target_w, target_h))
    input_data = cv2.cvtColor(input_data, cv2.COLOR_BGR2RGB)
    input_data = np.expand_dims(input_data, axis=0)

    if input_details[0]['dtype'] == np.int8:
        scale, zero_point = input_details[0]['quantization']
        input_data = (np.float32(input_data) / 255.0 / scale + zero_point).astype(np.int8)
    elif input_details[0]['dtype'] == np.float32:
        input_data = (np.float32(input_data) / 255.0)

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    
    output_data = interpreter.get_tensor(output_details[0]['index'])[0]

    if output_details[0]['dtype'] == np.int8:
        scale, zero_point = output_details[0]['quantization']
        output_data = (output_data.astype(np.float32) - zero_point) * scale

    idx = np.argmax(output_data)
    conf = output_data[idx]
    
    return idx, conf

# ==========================================
# --- MAIN EXECUTION ---
# ==========================================

print("⏳ Loading models...")
try:
    yolo_interpreter = load_interpreter(YOLO_MODEL_PATH)
    ripe_interpreter = load_interpreter(RIPE_CNN_PATH)
    green_interpreter = load_interpreter(GREEN_CNN_PATH)
    print("✅ All models loaded.")
except Exception as e:
    exit(f"❌ Critical Error: {e}")

# --- 1. YOLO SETUP ---
yolo_input = yolo_interpreter.get_input_details()
yolo_output = yolo_interpreter.get_output_details()
yolo_h = yolo_input[0]['shape'][1]
yolo_w = yolo_input[0]['shape'][2]

image = cv2.imread(IMAGE_PATH)
if image is None: exit(f"❌ Error reading {IMAGE_PATH}")
original_h, original_w = image.shape[:2]
final_img = image.copy()

input_data = cv2.resize(image, (yolo_w, yolo_h))
input_data = cv2.cvtColor(input_data, cv2.COLOR_BGR2RGB)
input_data = np.expand_dims(input_data, axis=0)

if yolo_input[0]['dtype'] == np.int8:
    scale, zero_point = yolo_input[0]['quantization']
    input_data = (np.float32(input_data) / 255.0 / scale + zero_point).astype(np.int8)
else:
    input_data = (np.float32(input_data) / 255.0)

# --- 2. RUN YOLO ---
print("🚀 Running YOLO Inference...")
yolo_interpreter.set_tensor(yolo_input[0]['index'], input_data)
yolo_interpreter.invoke()
output_data = yolo_interpreter.get_tensor(yolo_output[0]['index'])

if yolo_output[0]['dtype'] == np.int8:
    scale, zero_point = yolo_output[0]['quantization']
    output_data = (output_data.astype(np.float32) - zero_point) * scale

if output_data.shape[1] < output_data.shape[2]: 
    output_data = np.transpose(output_data, (0, 2, 1))

# --- 3. PARSE RESULTS ---
boxes_list = []
confidences_list = []
class_ids_list = []

predictions = output_data[0]
is_normalized = predictions[0][0] < 2.0 

for pred in predictions:
    scores = pred[4:]
    max_score = np.max(scores)
    
    if max_score > CONFIDENCE_THRESHOLD:
        class_id = np.argmax(scores)
        cx, cy, w, h = pred[0], pred[1], pred[2], pred[3]
        
        if is_normalized:
            cx, cy, w, h = cx*original_w, cy*original_h, w*original_w, h*original_h
        else:
            scale_x, scale_y = original_w/yolo_w, original_h/yolo_h
            cx, cy, w, h = cx*scale_x, cy*scale_y, w*scale_x, h*scale_y

        x = int(cx - (w / 2))
        y = int(cy - (h / 2))
        
        boxes_list.append([x, y, int(w), int(h)])
        confidences_list.append(float(max_score))
        class_ids_list.append(class_id)

indices = cv2.dnn.NMSBoxes(boxes_list, confidences_list, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)

# --- 4. CROP & CLASSIFY ---
if len(indices) > 0:
    print(f"🔍 Detected {len(indices)} tomatoes. Running health check...")
    
    for i in indices.flatten():
        x, y, w, h = boxes_list[i]
        yolo_label = YOLO_CLASSES[class_ids_list[i]]
        
        # Clamp Coordinates
        x = max(0, x)
        y = max(0, y)
        w = min(w, original_w - x)
        h = min(h, original_h - y)
        
        if w > 10 and h > 10:
            crop = image[y:y+h, x:x+w]
            
            final_label = ""
            final_conf = 0.0
            
            # --- ROUTING LOGIC ---
            if yolo_label == "Ripe":
                idx, final_conf = run_cnn_inference(ripe_interpreter, crop) 
                if idx < len(RIPE_CNN_CLASSES):
                    final_label = RIPE_CNN_CLASSES[idx]
                else:
                    final_label = f"Unknown"
            else:
                idx, final_conf = run_cnn_inference(green_interpreter, crop) 
                if idx < len(GREEN_CNN_CLASSES):
                    final_label = GREEN_CNN_CLASSES[idx]
                else:
                    final_label = f"Unknown"

            # --- DRAWING (FIXED VISIBILITY) ---
            # Ripe = Red Box, Green = Green Box
            color = (0, 0, 255) if yolo_label == "Ripe" else (0, 255, 0)
            
            # 1. Draw Bounding Box
            cv2.rectangle(final_img, (x, y), (x + w, y + h), color, 2)
            
            # 2. Prepare Label
            text = f"{final_label} ({int(final_conf * 100)}%)"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            
            # 3. Calculate Label Position (Prevent going off-screen)
            text_y = y - 10
            if text_y < 10: # If box is at the very top, move text inside the box
                text_y = y + th + 10
                
            # 4. Draw Background Rectangle (Same color as box)
            cv2.rectangle(final_img, (x, text_y - th - 5), (x + tw, text_y + 5), color, -1)
            
            # 5. Draw Text (ALWAYS WHITE for visibility)
            cv2.putText(final_img, text, (x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            print(f"   🍅 {yolo_label} -> {final_label}")

    # Save
    save_dir = os.path.join(os.path.dirname(IMAGE_PATH), "inferenced")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{os.path.splitext(os.path.basename(IMAGE_PATH))[0]}_full.jpg")
    cv2.imwrite(save_path, final_img)
    print(f"✅ Saved result to: {save_path}")

else:
    print("❌ No detections.")