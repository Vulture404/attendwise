import cv2
import numpy as np
from PIL import Image
import os

# Path for face image database
dataset_path = "dataset"
trainer_path = "trainer"

if not os.path.exists(trainer_path):
    os.makedirs(trainer_path)

recognizer = cv2.face.LBPHFaceRecognizer_create()
haar_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
detector = cv2.CascadeClassifier(haar_path)

# function to get the images and label data
def getImagesAndLabels(path):
    imagePaths = [os.path.join(path,f) for f in os.listdir(path)]
    faceSamples = []
    ids = []

    for imagePath in imagePaths:
        try:
            PIL_img = Image.open(imagePath).convert('L')  # convert to grayscale
            img_numpy = np.array(PIL_img, 'uint8')

            # Extract ID
            id = int(os.path.split(imagePath)[-1].split(".")[1])

            faces = detector.detectMultiScale(img_numpy)
            if len(faces) == 0:
                #print(f"[WARNING] No face detected in {os.path.basename(imagePath)}, skipping.")
                continue

            for (x,y,w,h) in faces:
                faceSamples.append(img_numpy[y:y+h,x:x+w])
                ids.append(id)
                print(f"[INFO] Face detected in {os.path.basename(imagePath)}, added to training.")

        except Exception as e:
            print(f"[ERROR] Could not process {imagePath}: {e}")

    return faceSamples, ids

print("[INFO] Gathering faces and IDs from dataset...")
faces, ids = getImagesAndLabels(dataset_path)

if len(faces) == 0:
    print("[ERROR] No valid face data found. Please recapture dataset.")
else:
    recognizer.train(faces, np.array(ids))
    recognizer.save(f"{trainer_path}/trainer.yml")
    print(f"[INFO] Training complete. {len(set(ids))} unique ID(s) trained. Model saved to {trainer_path}/trainer.yml")
