import cv2
import numpy as np
import tflite_runtime.interpreter as tflite
import os
import csv

# ==========================================
# --- CONFIGURATION (RASPBERRY PI) ---
# ==========================================

YOLO_MODEL_PATH = "/home/lima/LimaRaspAI/Quantized/Yolo/kagglemodelf16_quant.tflite"
RIPE_CNN_PATH   = "/home/lima/LimaRaspAI/Quantized/CNN/ripecnn_v2_float32.tflite"
GREEN_CNN_PATH  = "/home/lima/LimaRaspAI/Quantized/CNN/greencnn_v2_float32.tflite"
IMAGE_PATH      = "/home/lima/LimaRaspAI/Images/image2.jpg"

YOLO_CLASSES = ["Green", "Ripe"]

RIPE_CNN_CLASSES  = ['R_ber', 'R_healthy', 'R_spots'] 
GREEN_CNN_CLASSES = ['G_ber', 'G_healthy', 'G_lateblight', 'G_spots']

DISPLAY_NAMES = {
    'R_ber': 'RIPE BER',
    'R_healthy': 'HEALTHY RIPE',
    'R_spots': 'RIPE SPOTS',
    'G_ber': 'GREEN BER',
    'G_healthy': 'HEALTHY GREEN',
    'G_lateblight': 'LATE BLIGHT',
    'G_spots': 'GREEN SPOTS'
}

CONFIDENCE_THRESHOLD = 0.25
NMS_THRESHOLD = 0.35

# ==========================================
# --- HELPER FUNCTIONS ---
# ==========================================

def load_interpreter(path):
    interpreter = tflite.Interpreter(model_path=path)
    interpreter.allocate_tensors()
    return interpreter

def run_cnn(interpreter, img):
    inp = interpreter.get_input_details()
    out = interpreter.get_output_details()
    h, w = inp[0]['shape'][1:3]

    x = cv2.resize(img, (w, h))
    x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
    x = np.expand_dims(x, axis=0)

    if inp[0]['dtype'] == np.int8:
        scale, zero = inp[0]['quantization']
        x = (x.astype(np.float32) / 255.0 / scale + zero).astype(np.int8)
    else:
        x = x.astype(np.float32) / 255.0

    interpreter.set_tensor(inp[0]['index'], x)
    interpreter.invoke()

    y = interpreter.get_tensor(out[0]['index'])[0]
    if out[0]['dtype'] == np.int8:
        scale, zero = out[0]['quantization']
        y = (y.astype(np.float32) - zero) * scale

    idx = int(np.argmax(y))
    return idx, float(y[idx])

def update_csv(path, row):
    exists = os.path.isfile(path)
    with open(path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not exists:
            writer.writeheader()
        writer.writerow(row)

def pct(a, b):
    return round((a / b) * 100, 2) if b > 0 else 0.0

# ==========================================
# --- LOAD MODELS ---
# ==========================================

yolo = load_interpreter(YOLO_MODEL_PATH)
ripe_cnn = load_interpreter(RIPE_CNN_PATH)
green_cnn = load_interpreter(GREEN_CNN_PATH)

yin = yolo.get_input_details()
yout = yolo.get_output_details()
yh, yw = yin[0]['shape'][1:3]

# ==========================================
# --- LOAD IMAGE ---
# ==========================================

image = cv2.imread(IMAGE_PATH)
if image is None:
    raise RuntimeError("Image not found")

h0, w0 = image.shape[:2]
final_img = image.copy()

# ==========================================
# --- YOLO INFERENCE ---
# ==========================================

blob = cv2.resize(image, (yw, yh))
blob = cv2.cvtColor(blob, cv2.COLOR_BGR2RGB)
blob = np.expand_dims(blob, axis=0)

if yin[0]['dtype'] == np.int8:
    scale, zero = yin[0]['quantization']
    blob = (blob.astype(np.float32) / 255.0 / scale + zero).astype(np.int8)
else:
    blob = blob.astype(np.float32) / 255.0

yolo.set_tensor(yin[0]['index'], blob)
yolo.invoke()

pred = yolo.get_tensor(yout[0]['index'])
if yout[0]['dtype'] == np.int8:
    scale, zero = yout[0]['quantization']
    pred = (pred.astype(np.float32) - zero) * scale

if pred.shape[1] < pred.shape[2]:
    pred = pred.transpose(0, 2, 1)

pred = pred[0]

boxes, scores, classes = [], [], []
normalized = pred[0][0] < 2.0

for p in pred:
    s = np.max(p[4:])
    if s > CONFIDENCE_THRESHOLD:
        cid = np.argmax(p[4:])
        cx, cy, w, h = p[:4]

        if normalized:
            cx, cy, w, h = cx*w0, cy*h0, w*w0, h*h0
        else:
            cx, cy, w, h = cx*(w0/yw), cy*(h0/yh), w*(w0/yw), h*(h0/yh)

        boxes.append([int(cx-w/2), int(cy-h/2), int(w), int(h)])
        scores.append(float(s))
        classes.append(cid)

idxs = cv2.dnn.NMSBoxes(boxes, scores, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)

# ==========================================
# --- METADATA COUNTERS ---
# ==========================================

total = ripe = unripe = healthy = disease = 0
disease_counter = Counter()
detections_log = []

# ==========================================
# --- DRAW + CLASSIFY ---
# ==========================================

if len(idxs) > 0:
    for i in idxs.flatten():
        x, y, w, h = boxes[i]
        x, y = max(0, x), max(0, y)
        w, h = min(w, w0-x), min(h, h0-y)
        if w < 10 or h < 10:
            continue

        crop = image[y:y+h, x:x+w]
        yolo_label = YOLO_CLASSES[classes[i]]

        if yolo_label == "Ripe":
            idx, conf = run_cnn(ripe_cnn, crop)
            label = RIPE_CNN_CLASSES[idx]
            ripe += 1
        else:
            idx, conf = run_cnn(green_cnn, crop)
            label = GREEN_CNN_CLASSES[idx]
            unripe += 1

        total += 1

        if "healthy" in label.lower():
            healthy += 1
        else:
            disease += 1
            disease_counter[label] += 1

        detections_log.append({
            "fruit_id": total,
            "ripeness": yolo_label,
            "disease_label": DISPLAY_NAMES[label],
            "confidence": round(conf * 100, 2)
        })

        box_color = (0,0,255) if yolo_label=="Ripe" else (0,255,0)
        thickness = max(4, w0//300)
        font_scale = max(0.8, w0/1200)

        text = f"{DISPLAY_NAMES[label]} {int(conf*100)}%"
        (tw, th), bl = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX,
                                       font_scale, thickness)

        ty = y - 10 if y - 10 > th else y + th + 10

        cv2.rectangle(final_img, (x, ty-th-bl), (x+tw+6, ty+bl),
                      box_color, -1)
        cv2.rectangle(final_img, (x, y), (x+w, y+h),
                      box_color, thickness)

        cv2.putText(final_img, text, (x+3, ty),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, (255,255,255),
                    thickness//2, cv2.LINE_AA)

# ==========================================
# --- SAVE OUTPUTS ---
# ==========================================

out_dir = os.path.join(os.path.dirname(IMAGE_PATH), "inferenced")
os.makedirs(out_dir, exist_ok=True)

base = os.path.splitext(os.path.basename(IMAGE_PATH))[0]
img_out = os.path.join(out_dir, f"{base}.output.jpg")

cv2.imwrite(img_out, final_img)

csv_out = os.path.join(out_dir, "metadata.csv")
update_csv(csv_out, {
    "image_name": os.path.basename(IMAGE_PATH),
    "total_fruits": total,
    "ripe": ripe,
    "unripe": unripe,
    "healthy": healthy,
    "disease": disease,
    "ripe_%": pct(ripe, total),
    "unripe_%": pct(unripe, total),
    "healthy_%": pct(healthy, total),
    "disease_%": pct(disease, total)
})

# ==========================================
# --- TERMINAL SUMMARY ---
# ==========================================

print("\n================ DETECTION SUMMARY ================")
print(f"Total Fruits Detected: {total}")
print(f"Ripe Fruits: {ripe}")
print(f"Green Fruits: {unripe}")
print(f"Healthy Fruits: {healthy}")
print(f"Diseased Fruits: {disease}")

print("\n--- Disease Breakdown ---")
if disease_counter:
    for k, v in disease_counter.items():
        print(f"{DISPLAY_NAMES[k]}: {v}")
else:
    print("No diseases detected")

print("\n--- Per Fruit Classification ---")
for d in detections_log:
    print(f"Fruit #{d['fruit_id']} | {d['ripeness']} | {d['disease_label']} | {d['confidence']}%")

print("===================================================")

print(f"\nSaved image → {img_out}")
print(f"Updated CSV → {csv_out}")
