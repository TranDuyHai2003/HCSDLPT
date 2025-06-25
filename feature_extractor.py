# feature_extractor.py
import sqlite3
import librosa  # Chỉ dùng ở đây!
import numpy as np
import io

DATABASE_PATH = "voice_project.db"

def extract_and_save_features():
    """
    Tìm các file chưa có đặc trưng trong CSDL, dùng librosa để tính,
    và lưu ngược lại vào CSDL.
    """
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    # Lấy danh sách các file chưa được xử lý (feature_vector IS NULL)
    cursor.execute("SELECT id, filepath FROM audio_files WHERE feature_vector IS NULL")
    files_to_process = cursor.fetchall()
    
    if not files_to_process:
        print("Tất cả các file đã được trích xuất đặc trưng.")
        return

    print(f"Bắt đầu trích xuất đặc trưng cho {len(files_to_process)} file...")
    
    for file_id, audio_path in files_to_process:
        try:
            print(f"Đang xử lý file ID: {file_id}...")
            
            # --- ĐIỂM DUY NHẤT SỬ DỤNG LIBROSA ---
            y, sr = librosa.load(audio_path, sr=None)
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20) # Lấy 20 hệ số
            # -----------------------------------------
            
            # Chuyển mảng numpy thành chuỗi bytes để lưu vào BLOB
            byte_io = io.BytesIO()
            np.save(byte_io, mfccs) # Dùng np.save để lưu vào bộ đệm bytes
            feature_bytes = byte_io.getvalue()
            
            # Cập nhật bản ghi trong CSDL với vector đặc trưng
            cursor.execute('''
                UPDATE audio_files
                SET feature_vector = ?
                WHERE id = ?
            ''', (feature_bytes, file_id))
            
        except Exception as e:
            print(f"Lỗi khi xử lý file {audio_path}: {e}")

    conn.commit()
    conn.close()
    print("Hoàn tất quá trình trích xuất đặc trưng.")

if __name__ == "__main__":
    extract_and_save_features()