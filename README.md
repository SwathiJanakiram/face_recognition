# 🎯 Face Recognition System (Flask + OpenCV)

A mini project that performs real-time face capture, training, and recognition using Python, Flask, and OpenCV.

------------------------------------------------------------
📌 PROJECT OVERVIEW
------------------------------------------------------------

This system allows users to:
1. Capture face images using webcam
2. Train a face recognition model
3. Recognize faces in real time
4. Store user details (ID & Name) in Excel

It uses:
- Haar Cascade → Face Detection
- LBPH Algorithm → Face Recognition

------------------------------------------------------------
🏗️ PROJECT ARCHITECTURE
------------------------------------------------------------

User → Flask Web App → OpenCV → Webcam
                    ↓
               Face Images (data/)
                    ↓
             Train Model (LBPH)
                    ↓
            classifier.xml (Model)
                    ↓
           Real-Time Recognition

------------------------------------------------------------
📂 PROJECT STRUCTURE
------------------------------------------------------------

FaceRecognitionProject/
│
├── app.py
├── templates/
│   └── index.html
├── data/
│   └── user.ID.img.jpg
├── classifier.xml
├── haarcascade_frontalface_default.xml
└── detail.xlsx

------------------------------------------------------------
⚙️ TECHNOLOGIES USED
------------------------------------------------------------

- Python
- Flask
- OpenCV
- NumPy
- OpenPyXL
- Bootstrap

------------------------------------------------------------
🚀 FEATURES
------------------------------------------------------------

1️⃣ Capture Faces
- Opens webcam
- Detects face
- Captures 10 images
- Saves images in data/ folder
- Stores ID & Name in Excel

2️⃣ Train Model
- Reads images from data/
- Converts to grayscale
- Applies data augmentation:
  - Horizontal flip
  - Brightness adjustment
- Trains using LBPH
- Saves model as classifier.xml

3️⃣ Recognize Faces
- Loads trained model
- Opens webcam
- Detects face
- Predicts ID
- Displays Name if confidence < 45
- Displays "Unknown" otherwise
- Press ESC to exit

------------------------------------------------------------
🧠 ALGORITHM USED
------------------------------------------------------------

Haar Cascade:
Used for face detection.

LBPH (Local Binary Pattern Histogram):
Used for recognition.
- Works well for small datasets
- Fast and efficient
- Good for real-time applications

------------------------------------------------------------
📊 EXCEL INTEGRATION
------------------------------------------------------------

User details are stored in detail.xlsx:

ID | Name
--------------
1  | Swathi
2  | John

Used to map predicted ID to actual name.

------------------------------------------------------------
🖥️ HOW TO RUN
------------------------------------------------------------

1️⃣ Install Dependencies

pip install flask opencv-python numpy openpyxl pillow
pip install opencv-contrib-python

2️⃣ Ensure Required Files Exist
- haarcascade_frontalface_default.xml
- detail.xlsx
- data/ folder

3️⃣ Run Application

python app.py

4️⃣ Open Browser

http://127.0.0.1:5000/home

------------------------------------------------------------
🔐 CONFIDENCE LOGIC
------------------------------------------------------------

If prediction confidence < 45 → Known Person
Else → Unknown

Lower confidence means better match.

------------------------------------------------------------
⚠️ LIMITATIONS
------------------------------------------------------------

- Requires proper lighting
- Small dataset
- Excel-based storage (not scalable)
- Hardcoded Excel path
- Works best for single face

------------------------------------------------------------
🔮 FUTURE IMPROVEMENTS
------------------------------------------------------------

- Replace Excel with Database (MySQL/SQLite)
- Add Attendance Feature
- Add Login Authentication
- Use Deep Learning (CNN)
- Deploy on Cloud
- Improve UI/UX

------------------------------------------------------------
📚 LEARNING OUTCOMES
------------------------------------------------------------

- Flask Routing
- OpenCV Image Processing
- Face Detection & Recognition
- Data Augmentation
- Model Training
- File Handling
- Frontend + Backend Integration

