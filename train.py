import os
import cv2
import pickle
import face_recognition
from mtcnn import MTCNN
from database.db_connection import connect_db

# Initialize MTCNN for face detection
detector = MTCNN()

# Path to images and model storage
IMAGE_PATH = "images"
ENCODINGS_FILE = "models/encodings.pickle"

# Load existing encodings if available
if os.path.exists(ENCODINGS_FILE):
    with open(ENCODINGS_FILE, "rb") as f:
        encodeListKnown, classNames, student_ids = pickle.load(f)
    print("✅ Loaded existing face encodings.")
else:
    encodeListKnown, classNames, student_ids = [], [], {}
    print("🔄 No previous encodings found. Training from scratch.")

# Connect to MySQL database
conn = connect_db()
cursor = conn.cursor()

print("\n🔍 Scanning images for new faces...")

# Loop through each student's folder
for student_folder in os.listdir(IMAGE_PATH):
    student_path = os.path.join(IMAGE_PATH, student_folder)

    if os.path.isdir(student_path) and student_folder not in classNames:
        print(f"📌 New student detected: {student_folder}")
        classNames.append(student_folder)

        for img_name in os.listdir(student_path):
            img_path = os.path.join(student_path, img_name)

            # 🔴 Check if the file is a valid image
            if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                print(f"❌ Skipping invalid file: {img_name}")
                continue

            img = cv2.imread(img_path)

            # 🔴 Check if image is loaded correctly
            if img is None:
                print(f"❌ ERROR: Could not load image: {img_path}")
                continue  # Skip this image

            # Detect faces using MTCNN
            faces = detector.detect_faces(img)

            if faces:
                x, y, width, height = faces[0]['box']
                face_crop = img[y:y+height, x:x+width]  # Crop detected face
                face_crop = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)

                encodes = face_recognition.face_encodings(face_crop)

                if encodes:
                    encodeListKnown.append(encodes[0])
                    print(f"✔️ Encoding saved for {student_folder}")

        # Add student to MySQL database if not already present
        cursor.execute("SELECT id FROM students WHERE name = %s", (student_folder,))
        result = cursor.fetchone()

        if result is None:
            cursor.execute("INSERT INTO students (name) VALUES (%s)", (student_folder,))
            conn.commit()
            student_id = cursor.lastrowid
        else:
            student_id = result[0]

        student_ids[student_folder] = student_id

cursor.close()
conn.close()

# Save the updated encodings
with open(ENCODINGS_FILE, "wb") as f:
    pickle.dump((encodeListKnown, classNames, student_ids), f)

print("\n✅ Training Complete! New face encodings saved.")
