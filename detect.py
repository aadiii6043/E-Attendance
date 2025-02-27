import cv2
import numpy as np
import face_recognition
import pickle
from datetime import datetime
import time
from mtcnn import MTCNN
from database.db_connection import connect_db

# Load trained encodings
with open("models/encodings.pickle", "rb") as f:
    encodeListKnown, classNames, student_ids = pickle.load(f)

# Initialize MTCNN for face detection
detector = MTCNN()

# MySQL Attendance Marking
# ✅ Modify markAttendance to print debug info
def markAttendance(name):
    conn = connect_db()
    cursor = conn.cursor()

    student_id = student_ids.get(name)

    if student_id:
        try:
            cursor.execute("INSERT INTO attendance (student_id) VALUES (%s)", (student_id,))
            conn.commit()
            print(f"✅ Attendance marked for {name} at {datetime.now()}")
        except Exception as e:
            print(f"❌ Error marking attendance for {name}: {e}")

    cursor.close()
    conn.close()

# Start Webcam
cap = cv2.VideoCapture(0)
cap.set(3, 1280)  # Set higher resolution
cap.set(4, 720)

start_time = time.time()  # Track start time
run_time = 10  # Run for 30 seconds

while True:
    success, img = cap.read()

    if not success or img is None:
        print("❌ ERROR: No frame captured. Exiting...")
        break

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Detect faces using MTCNN
    faces = detector.detect_faces(imgRGB)

    if faces:
        x, y, width, height = faces[0]['box']
        print(f"✅ Face detected: {faces[0]['box']}")

        # Crop face properly
        y1, y2 = max(0, y), min(y + height, imgRGB.shape[0])
        x1, x2 = max(0, x), min(x + width, imgRGB.shape[1])
        face_crop = imgRGB[y1:y2, x1:x2]

        # 🔹 Detect face locations using `face_recognition`
        face_locations = face_recognition.face_locations(imgRGB)

        if not face_locations:
            print("⚠️ No face landmarks detected, skipping face.")
            continue

        # 🔹 Encode the detected face
        encodesCurFrame = face_recognition.face_encodings(imgRGB, face_locations)

        if encodesCurFrame:
            encodeFace = encodesCurFrame[0]
            matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
            faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)

            if len(faceDis) == 0:
                print("⚠️ No matching faces found.")
                continue  # Skip if no faces are recognized

            matchIndex = np.argmin(faceDis)

            if matchIndex < len(classNames) and matches[matchIndex]:  # Ensure matchIndex is valid
                name = classNames[matchIndex].upper()

                # Draw Rectangle & Name
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.rectangle(img, (x1, y1 - 35), (x2, y1), (0, 255, 0), cv2.FILLED)
                cv2.putText(img, name, (x1 + 6, y1 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

                markAttendance(name)

    else:
        print("⚠️ No face detected.")

    cv2.imshow('Webcam', img)

    # 🔹 Stop after 30 seconds
    if time.time() - start_time > run_time:
        print("✅ Camera session completed successfully.")
        break

    # 🔹 Press ESC to exit manually
    if cv2.waitKey(1) == 27:
        print("👋 Exiting face detection.")
        break

cap.release()
cv2.destroyAllWindows()
