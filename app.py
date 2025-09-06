import cv2
import numpy as np
import base64
import io
from flask import Flask, request, jsonify, render_template, Response
from flask_cors import CORS
from PIL import Image
import datetime
import os
import csv
import subprocess

app = Flask(__name__, template_folder="templates")
CORS(app)

# Paths
MODEL_PATH = "trainer/trainer.yml"
STUDENTS_FILE = "students.csv"

# Load recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()
if os.path.exists(MODEL_PATH):
    recognizer.read(MODEL_PATH)
else:
    print("[WARNING] No trained model found. Run trainer.py first.")

# Haar cascade
haar_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(haar_path)

# Load student mapping
def load_students():
    students = {}
    if os.path.exists(STUDENTS_FILE):
        with open(STUDENTS_FILE, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    students[int(row["ID"])] = row["Name"]
                except:
                    continue
    return students

id_name_map = load_students()
attendance_log = {}  # {date: set(names)}

def decode_image(base64_string):
    """Convert base64 string to OpenCV image"""
    img_data = base64.b64decode(base64_string.split(",")[1])
    img = Image.open(io.BytesIO(img_data))
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

# ---------------- ROUTES ----------------

@app.route("/")
def index():
    """Serve the frontend"""
    return render_template("frontend.html")

@app.route("/api/mark_attendance", methods=["POST"])
def mark_attendance():
    data = request.get_json()
    image_b64 = data.get("image")
    frame = decode_image(image_b64)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, 1.2, 5)

    recognized = []
    today = datetime.date.today().isoformat()

    for (x, y, w, h) in faces:
        id_, conf = recognizer.predict(gray[y:y+h, x:x+w])
        if conf < 70:  # lower = better match
            name = id_name_map.get(id_, f"User {id_}")
            recognized.append(name)
            attendance_log.setdefault(today, set()).add(name)
        else:
            recognized.append("Unknown")

    return jsonify({
        "success": True,
        "recognized_students": recognized,
        "total_students": len(id_name_map),
        "timestamp": datetime.datetime.now().timestamp()
    })

@app.route("/api/add_student", methods=["POST"])
def add_student():
    data = request.get_json()
    name = data.get("name")
    photo_b64 = data.get("photo")

    if not name or not photo_b64:
        return jsonify({"success": False, "message": "Missing name or photo"}), 400

    # Assign a new numeric ID
    new_id = max(id_name_map.keys(), default=0) + 1
    id_name_map[new_id] = name

    # Save student in CSV
    file_exists = os.path.exists(STUDENTS_FILE)
    with open(STUDENTS_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["ID", "Name"])
        writer.writerow([new_id, name])

    # Save first photo to dataset
    frame = decode_image(photo_b64)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    os.makedirs("dataset", exist_ok=True)
    dataset_path = f"dataset/User.{new_id}.1.jpg"
    cv2.imwrite(dataset_path, gray)

    # --- Run dataset_creator.py and trainer.py ---
    try:
        subprocess.run(["python", "dataset_creator.py", str(new_id)], check=True)
        subprocess.run(["python", "trainer.py"], check=True)
    except subprocess.CalledProcessError as e:
        return jsonify({"success": False, "message": f"Error running scripts: {e}"}), 500

    return jsonify({"success": True, "message": f"{name} added successfully!", "id": new_id})

@app.route("/api/attendance_today", methods=["GET"])
def get_today_attendance():
    today = datetime.date.today().isoformat()
    students_present = list(attendance_log.get(today, []))
    return jsonify({
        "date": today,
        "present": students_present,
        "count": len(students_present)
    })

# ---------------------------------------

if __name__ == "__main__":
    app.run(debug=True)
