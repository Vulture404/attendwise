import cv2
import os
import numpy as np

def train_model():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    faces = []
    ids = []
    labels = {}

    dataset_dir = "dataset"
    current_id = 0

    for root, dirs, files in os.walk(dataset_dir):
        for file in files:
            if file.endswith("jpg") or file.endswith("png"):
                path = os.path.join(root, file)
                name = os.path.basename(root)
                if name not in labels.values():
                    labels[current_id] = name
                    id_ = current_id
                    current_id += 1
                else:
                    id_ = list(labels.keys())[list(labels.values()).index(name)]

                img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                detected_faces = face_cascade.detectMultiScale(img, 1.3, 5)
                for (x, y, w, h) in detected_faces:
                    faces.append(img[y:y+h, x:x+w])
                    ids.append(id_)

    recognizer.train(faces, np.array(ids))
    recognizer.save("trainer.yml")

    with open("labels.txt", "w") as f:
        for k, v in labels.items():
            f.write(f"{v},{k}\n")

    print("Training complete. Model saved as trainer.yml")

if __name__ == "__main__":
    train_model()
