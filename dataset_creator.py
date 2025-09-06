# dataset_creator.py
import cv2
import os
import csv

DATASET_DIR = "dataset"
STUDENTS_FILE = "students.csv"
NUM_SAMPLES = 80   # change to 30-100 as you like

os.makedirs(DATASET_DIR, exist_ok=True)

def save_student_record(student_id: int, name: str):
    rows = []
    if os.path.exists(STUDENTS_FILE):
        with open(STUDENTS_FILE, "r", newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            rows = list(reader)

    # Ensure header
    if not rows or rows[0] != ["ID", "Name"]:
        rows = [["ID", "Name"]] + [r for r in rows if r != []]

    # Update or append
    updated = False
    for i, row in enumerate(rows):
        if i == 0:  # skip header
            continue
        if row[0] == str(student_id):
            rows[i][1] = name
            updated = True
            break

    if not updated:
        rows.append([str(student_id), name])

    with open(STUDENTS_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(rows)
    print(f"[INFO] Saved student: {student_id} -> {name}")

# Ask user
while True:
    sid = input("Enter numeric ID (e.g. 1): ").strip()
    try:
        student_id = int(sid)
        break
    except:
        print("ID must be a number. Try again.")

student_name = input("Enter Name (e.g. Aditya): ").strip()
if student_name == "":
    student_name = f"User{student_id}"

save_student_record(student_id, student_name)

# Start capture
cam = cv2.VideoCapture(0)
detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
print("[INFO] Starting capture. Press 'q' to stop early. Look at the camera...")

count = 0
while True:
    ret, img = cam.read()
    if not ret:
        print("[ERROR] Cannot read camera frame.")
        break

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(80,80))

    for (x, y, w, h) in faces:
        count += 1
        face_img = gray[y:y+h, x:x+w]
        filename = os.path.join(DATASET_DIR, f"User.{student_id}.{count}.jpg")
        cv2.imwrite(filename, face_img)
        cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
        cv2.putText(img, f"{student_name} {count}/{NUM_SAMPLES}", (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

    cv2.imshow("Capturing Faces - Press q to stop", img)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    if count >= NUM_SAMPLES:
        break

cam.release()
cv2.destroyAllWindows()
print(f"[INFO] Captured {count} images for {student_name} (ID: {student_id})")
