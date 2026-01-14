import firebase_admin
from firebase_admin import credentials, storage, firestore, auth
import os
import csv
import datetime
import time

# ==========================================
# --- FIREBASE CONFIGURATION ---
# ==========================================

FIREBASE_KEY_PATH = "/home/lima/firebase-adminsdk.json"
FIREBASE_BUCKET = "espcam-69f58.appspot.com"

INFERENCED_DIR = "/home/lima/LimaRaspAI/Images/inferenced"
METADATA_CSV = os.path.join(INFERENCED_DIR, "metadata.csv")

UPLOAD_INTERVAL_SECONDS = 240  # 4 minutes
USER_EMAIL = "chiri.levisk@gmail.com"

# ==========================================
# --- FIREBASE INIT ---
# ==========================================

cred = credentials.Certificate(FIREBASE_KEY_PATH)
firebase_admin.initialize_app(cred, {
    "storageBucket": FIREBASE_BUCKET
})

db = firestore.client()
bucket = storage.bucket()

# ==========================================
# --- HELPERS ---
# ==========================================

def get_uid(email):
    try:
        user = auth.get_user_by_email(email)
        return user.uid
    except Exception as e:
        print("❌ Firebase auth error:", e)
        return None

def load_metadata_rows():
    if not os.path.exists(METADATA_CSV):
        return []

    with open(METADATA_CSV, newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)

def upload_image(uid, image_path):
    filename = os.path.basename(image_path)
    blob = bucket.blob(f"inferenced/{uid}/{filename}")

    blob.upload_from_filename(image_path, content_type="image/jpeg")
    blob.make_public()

    return blob.public_url

def upload_metadata(uid, metadata, image_url):
    doc_id = os.path.splitext(metadata["image_name"])[0]

    payload = {
        "image_name": metadata["image_name"],
        "image_url": image_url,
        "total_fruits": int(metadata["total_fruits"]),
        "ripe": int(metadata["ripe"]),
        "unripe": int(metadata["unripe"]),
        "healthy": int(metadata["healthy"]),
        "disease": int(metadata["disease"]),
        "ripe_percentage": float(metadata["ripe_%"]),
        "unripe_percentage": float(metadata["unripe_%"]),
        "healthy_percentage": float(metadata["healthy_%"]),
        "disease_percentage": float(metadata["disease_%"]),
        "timestamp": firestore.SERVER_TIMESTAMP
    }

    db.collection("users") \
      .document(uid) \
      .collection("inference_results") \
      .document(doc_id) \
      .set(payload)

def already_uploaded(uid, image_name):
    doc_id = os.path.splitext(image_name)[0]
    ref = db.collection("users") \
            .document(uid) \
            .collection("inference_results") \
            .document(doc_id)
    return ref.get().exists

# ==========================================
# --- MAIN LOOP ---
# ==========================================

def main():
    uid = get_uid(USER_EMAIL)
    if not uid:
        print("❌ User not found. Exiting.")
        return

    print(f"✅ Firebase connected. UID: {uid}")

    while True:
        rows = load_metadata_rows()

        for row in rows:
            image_name = row["image_name"]
            base = os.path.splitext(image_name)[0]
            image_file = os.path.join(INFERENCED_DIR, f"{base}.output.jpg")

            if not os.path.exists(image_file):
                continue

            if already_uploaded(uid, image_name):
                continue

            try:
                print(f"⬆ Uploading {image_name} ...")
                image_url = upload_image(uid, image_file)
                upload_metadata(uid, row, image_url)
                print(f"✅ Uploaded {image_name}")

            except Exception as e:
                print(f"❌ Upload failed for {image_name}: {e}")

        time.sleep(UPLOAD_INTERVAL_SECONDS)

# ==========================================
# --- ENTRY POINT ---
# ==========================================

if __name__ == "__main__":
    main()
