import mysql.connector

def connect_db():
    try:
        connection = mysql.connector.connect(
            host="localhost",
            user="root",  # Replace this with your actual MySQL username
            password="apple@123",  # Replace this with your actual MySQL password
            database="attendance_system"
        )
        if connection.is_connected():
            print("✅ Database connected successfully!")
        return connection
    except mysql.connector.Error as err:
        print(f"❌ Database Connection Error: {err}")
        return None

# Run the function to test the connection
if __name__ == "__main__":
    conn = connect_db()
    if conn:
        conn.close()
        print("🔌 Connection closed.")