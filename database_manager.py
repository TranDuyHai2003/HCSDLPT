# database_manager.py
import sqlite3
import os
import glob

# --- Cấu hình ---
AUDIO_FOLDER_PATH = "D:\\HCSDLDPT\\folder giong noi" # <<< THAY ĐỔI ĐƯỜNG DẪN NÀY
DATABASE_PATH = "voice_project.db"

def create_database_table():
    """Tạo CSDL và bảng audio_files nếu chưa tồn tại."""
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    # Tạo bảng với các cột cần thiết
    # id: Khóa chính tự tăng
    # filename: Tên file (ví dụ: 'female_voice_001.wav')
    # filepath: Đường dẫn đầy đủ đến file
    # feature_vector: Dữ liệu đặc trưng (dạng BLOB - Binary Large Object)
    # created_at: Dấu thời gian khi bản ghi được tạo
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS audio_files (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL UNIQUE,
            filepath TEXT NOT NULL,
            feature_vector BLOB,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    conn.commit()
    conn.close()
    print(f"CSDL và bảng đã sẵn sàng tại '{DATABASE_PATH}'.")

def populate_files_to_db():
    """Quét thư mục audio và điền thông tin file vào CSDL (chưa có đặc trưng)."""
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    print(f"Đang quét thư mục: {AUDIO_FOLDER_PATH}")
    wav_files = glob.glob(os.path.join(AUDIO_FOLDER_PATH, "*.wav"))
    
    if not wav_files:
        print("CẢNH BÁO: Không tìm thấy file .wav nào trong thư mục!")
        return

    for file_path in wav_files:
        filename = os.path.basename(file_path)
        # Dùng INSERT OR IGNORE để không báo lỗi nếu file đã tồn tại
        cursor.execute('''
            INSERT OR IGNORE INTO audio_files (filename, filepath)
            VALUES (?, ?)
        ''', (filename, file_path))
        
    conn.commit()
    conn.close()
    print(f"Đã thêm thông tin của {len(wav_files)} file vào CSDL.")

if __name__ == "__main__":
    create_database_table()
    populate_files_to_db()