# attendwise
Automated attendance application 
# AttendWise – Automatic Attendance Portal

**Hackathon Project Guide**
*Automated attendance system using face recognition and live webcam feed.*

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Features](#features)
3. [Tech Stack](#tech-stack)
4. [Setup Instructions](#setup-instructions)
5. [Folder Structure](#folder-structure)
6. [Usage](#usage)
7. [Add New Student](#add-new-student)
8. [Troubleshooting](#troubleshooting)
9. [Future Improvements](#future-improvements)

---

## Project Overview

AttendWise is an automatic attendance management system that leverages **computer vision** to recognize students in a classroom using a **live webcam feed**.
It tracks attendance in real-time and logs the results automatically.

Students can be **added via a simple web interface**, with photos captured and the recognition model retrained automatically.

---

## Features

* Real-time face recognition using **OpenCV** and **LBPHFaceRecognizer**
* Automatic attendance logging per day
* User-friendly frontend interface with live camera feed
* Add new students dynamically
* Dataset creation and model retraining triggered automatically
* Attendance visualization and log display

---

## Tech Stack

* **Backend:** Python, Flask, OpenCV, Pillow (PIL), CSV
* **Frontend:** HTML, TailwindCSS, JavaScript
* **Face Recognition:** OpenCV LBPHFaceRecognizer, Haar Cascades
* **Data Storage:** CSV for student mapping, local dataset folder for images

---

## Setup Instructions

### Prerequisites

* Python 3.8+
* pip package manager
* Webcam (for live feed)

### Install Dependencies

```bash
pip install flask flask-cors opencv-python opencv-contrib-python pillow numpy
```

### Project Files

* `app.py` → Flask backend
* `dataset_creator.py` → Captures and processes student images
* `trainer.py` → Trains LBPH recognizer model
* `frontend.html` → Web interface (served via Flask)
* `students.csv` → Stores student ID-Name mapping
* `dataset/` → Folder storing student images
* `trainer/trainer.yml` → Trained face recognition model

---

## Folder Structure

```
AttendWise/
│
├─ app.py
├─ dataset_creator.py
├─ trainer.py
├─ templates/
│   └─ frontend.html
├─ dataset/
│   └─ User.1.1.jpg
├─ trainer/
│   └─ trainer.yml
├─ students.csv
├─ README.md
└─ requirements.txt
```

---

## Usage

1. **Run Flask backend**

```bash
python app.py
```

2. **Open Frontend**

Navigate to:

```
http://127.0.0.1:5000/
```

3. **Live Attendance**

* Webcam feed starts automatically
* Faces detected are recognized in real-time
* Attendance log updates dynamically

---

## Add New Student

1. Click **“Add New Student”** button on the header
2. Enter **Full Name**
3. Upload a **photo**
4. Click **Save Student**

**What happens in the backend:**

* Photo is saved to `dataset/`
* Student added to `students.csv`
* `dataset_creator.py` runs to add dataset
* `trainer.py` runs to retrain the model
* Modal displays success/error message

---

## Troubleshooting

| Issue                            | Solution                                                                       |
| -------------------------------- | ------------------------------------------------------------------------------ |
| **Camera not detected**          | Check webcam permissions and refresh page                                      |
| **“Unexpected token '<'” error** | Make sure frontend sends JSON and backend reads `request.get_json()` correctly |
| **No trained model found**       | Run `trainer.py` at least once before first attendance scan                    |
| **Face not recognized**          | Ensure the student dataset has enough images and quality lighting              |

---

## Future Improvements

* Add **multi-face detection** with better accuracy
* Use **deep learning models** for more robust recognition
* Integrate **cloud storage** for dataset and model
* Generate **daily/monthly attendance reports** automatically
* Add **admin login** and **role-based access control**

---

**AttendWise** is now ready for deployment and live classroom testing. This system is ideal for **hackathons**, **educational institutes**, or any scenario requiring automated attendance tracking with minimal manual effort.
