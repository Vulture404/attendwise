# webcam_attendance.py
import cv2
import os
import csv
from datetime import datetime

MODEL_FILE = "trainer/trainer.yml"
STUDENTS_FILE = "students.csv"
ATTENDANCE_FILE = "attendance.csv"
CONF_THRESHOLD = 75   # adjust 60-85 depending on accuracy

# Load students mapping (ID -> Name)
students = {}
if os.path.exists(STUDENTS_FILE):
    with open(STUDENTS_FILE, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                students[int(row["ID"])] = row["Name"]
            except Exception:
                continue

print(f"[INFO] Loaded {len(students)} students from {STUDENTS_FILE}")

# Load recognizer
if not os.path.exists(MODEL_FILE):
    print("[ERROR] Trained model not found. Run trainer.py first.")
    exit(0)

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(MODEL_FILE)

faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
cap = cv2.VideoCapture(0)

# Ensure attendance file exists and read today's marks to avoid duplicates
today = datetime.now().strftime("%Y-%m-%d")
marked_ids = set()
if os.path.exists(ATTENDANCE_FILE):
    with open(ATTENDANCE_FILE, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            try:
                if r.get("Date") == today:
                    marked_ids.add(int(r.get("ID")))
            except:
                pass
else:
    # create file with headers
    with open(ATTENDANCE_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["ID", "Name", "Date", "Time", "Status"])

print("[INFO] Starting webcam. Press 'q' to exit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Camera frame not available.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(80,80))

    for (x, y, w, h) in faces:
        roi = gray[y:y+h, x:x+w]
        # Some models require resizing, but LBPH works with variable sizes
        try:
            id_pred, confidence = recognizer.predict(roi)
        except Exception as e:
            # If prediction fails for any reason
            id_pred, confidence = None, 999

        # Debug print
        #print(f"[DEBUG] Predicted ID={id_pred}, confidence={confidence:.1f}")

        if id_pred is not None and isinstance(id_pred, (int,)) and confidence < CONF_THRESHOLD:
            name = students.get(id_pred, "Unknown")
            label = f"{name} ({100 - int(confidence)}%)"
            color = (0,255,0)
            # mark attendance if not already marked today
            if id_pred not in marked_ids and name != "Unknown":
                now = datetime.now()
                with open(ATTENDANCE_FILE, "a", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow([id_pred, name, now.strftime("%Y-%m-%d"), now.strftime("%H:%M:%S"), "Present"])
                marked_ids.add(id_pred)
                print(f"[INFO] Marked present: {id_pred} -> {name}")
        else:
            label = "Unknown"
            color = (0,0,255)

        cv2.rectangle(frame, (x,y), (x+w, y+h), color, 2)
        cv2.putText(frame, label, (x+5, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    cv2.imshow("Attendance (press q to quit)", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
