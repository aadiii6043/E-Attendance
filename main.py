import os
import argparse
import train
import detect
from database.db_connection import connect_db

def main():
    # Ensure the database is connected
    conn = connect_db()
    if conn:
        print("✅ Database connected successfully!\n")
        conn.close()
    else:
        print("❌ Database connection failed!")
        return

    print("📌 Welcome to the E-Attendance System")
    print("1️⃣ Train Face Recognition Model")
    print("2️⃣ Start Real-Time Face Recognition")
    print("3️⃣ Exit")
    
    choice = input("\nEnter your choice (1/2/3): ").strip()

    if choice == "1":
        print("\n🔄 Training model...")
        os.system("python train.py")  # Runs train.py
        print("✅ Training complete! Encodings saved.")

    elif choice == "2":
        print("\n📷 Starting real-time face recognition...")
        os.system("python detect.py")  # Runs detect.py
        print("✅ Face recognition session ended.")

    elif choice == "3":
        print("\n👋 Exiting program.")
        exit()

    else:
        print("\n❌ Invalid choice! Please enter 1, 2, or 3.")

if __name__ == "__main__":
    main()
