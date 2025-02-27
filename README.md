# 🎯 E-Attendance System | Python - MySQL - OpenCV

## 📌 Project Overview
This **AI-powered E-Attendance System** uses **Face Recognition** to automate attendance marking. It eliminates **manual roll calls, fingerprint-based systems, and proxy attendance** by leveraging **computer vision and deep learning**. The system ensures **high accuracy, security, and efficiency** while reducing human errors.

---

## 🏗️ Project File Structure

```plaintext
E-Attendance/
│── images/                      # Folder for storing student images
│   ├── Aditya/                  # Student folder (name-based)
│   │   ├── img1.jpg
│   │   ├── img2.jpg
│   ├── Vishal/                  
│       ├── img1.jpg
│       ├── img2.jpg
│
│── models/                       # Stores trained encodings
│   ├── encodings.pickle
│
│── database/                     # Database connection file
│   ├── db_connection.py
│
│── train.py                       # Training script
│── detect.py                      # Face recognition & attendance marking
│── main.py                        # Main menu script
│── README.md                      # Project Documentation
