import firebase_admin
from firebase_admin import credentials, storage, auth, db
import os
import csv
import time

# ==========================================
# --- FIREBASE CONFIG ---
# ==========================================

FIREBASE_KEY_PATH = "/home/lima/firebase-adminsdk.json"
FIREBASE_BUCKET = "espcam-69f58.appspot.com"
DATABASE_URL = "https://espcam-69f58-default-rtdb.firebaseio.com"

INFERENCED_DIR = "/home/lima/LimaRaspAI/Images/inferenced"
METADATA_CSV = os.path.join(INFERENCED_DIR, "metadata.csv")

UPLOAD_INTERVAL = 240  # seconds
USER_EMAIL = "chiri.levisk@gmail.com"

# ==========================================
# --- INIT FIREBASE ---
# ==========================================

cred = credentials.Certificate(FIREBASE_KEY_PATH)
firebase_admin.initialize_app(cred, {
    "storageBucket": FIREBASE_BUCKET,
    "databaseURL": DATABASE_URL
})

bucket = storage.bucket()

# ==========================================
# --- HELPERS ---
# ==========================================

def get_uid(email):
    try:
        user = auth.get_user_by_email(email)
        return user.uid
    except Exception as e:
        print("Auth error:", e)
        return None

def load_metadata():
    if not os.path.exists(METADATA_CSV):
        return []

    with open(METADATA_CSV, newline="") as f:
        return list(csv.DictReader(f))

def upload_image(uid, image_path):
    name = os.path.basename(image_path)
    blob = bucket.blob(f"inferenced/{uid}/{name}")
    blob.upload_from_filename(image_path, content_type="image/jpeg")
    blob.make_public()
    return blob.public_url

def already_uploaded(uid, image_name):
    base = os.path.splitext(image_name)[0]
    ref = db.reference(f"users/{uid}/inference_results/{base}")
    return ref.get() is not None

def upload_metadata(uid, row, image_url):
    base = os.path.splitext(row["image_name"])[0]

    payload = {
        "image_name": row["image_name"],
        "image_url": image_url,
        "total_fruits": int(row["total_fruits"]),
        "ripe": int(row["ripe"]),
        "unripe": int(row["unripe"]),
        "healthy": int(row["healthy"]),
        "disease": int(row["disease"]),
        "ripe_percentage": float(row["ripe_%"]),
        "unripe_percentage": float(row["unripe_%"]),
        "healthy_percentage": float(row["healthy_%"]),
        "disease_percentage": float(row["disease_%"]),
        "timestamp": int(time.time())
    }

    db.reference(f"users/{uid}/inference_results/{base}").set(payload)

# ==========================================
# --- MAIN LOOP ---
# ==========================================

def main():
    uid = get_uid(USER_EMAIL)
    if not uid:
        print("User not found.")
        return

    print(f"✅ Firebase connected. UID: {uid}")

    while True:
        rows = load_metadata()

        for row in rows:
            image_name = row["image_name"]
            base = os.path.splitext(image_name)[0]
            image_path = os.path.join(INFERENCED_DIR, f"{base}.output.jpg")

            if not os.path.exists(image_path):
                continue

            if already_uploaded(uid, image_name):
                continue

            try:
                print(f"⬆ Uploading {image_name}")
                url = upload_image(uid, image_path)
                upload_metadata(uid, row, url)
                print(f"✅ Uploaded {image_name}")

            except Exception as e:
                print(f"❌ Upload failed for {image_name}: {e}")

        time.sleep(UPLOAD_INTERVAL)

# ==========================================
# --- ENTRY POINT ---
# ==========================================

if __name__ == "__main__":
    main()
