# search_engine.py (phiên bản nâng cấp)
import sqlite3
import librosa
import numpy as np
import io
import math
import os # Cần import os để kiểm tra đường dẫn

DATABASE_PATH = "voice_project.db"

# ===================================================================
# CÁC HÀM BÊN TRÊN GIỮ NGUYÊN
# - cosine_similarity_from_scratch(vec1_flat, vec2_flat)
# - find_similar_voices(query_audio_path, top_n=3)
# Chúng ta không cần thay đổi logic cốt lõi này.
# ===================================================================

def cosine_similarity_from_scratch(vec1_flat, vec2_flat):
    """Tính Cosine Similarity từ hai vector đã được làm phẳng."""
    dot_product = sum(x * y for x, y in zip(vec1_flat, vec2_flat))
    mag_vec1 = math.sqrt(sum(x * x for x in vec1_flat))
    mag_vec2 = math.sqrt(sum(x * x for x in vec2_flat))
    if not mag_vec1 or not mag_vec2:
        return 0.0
    return dot_product / (mag_vec1 * mag_vec2)

def find_similar_voices(query_audio_path, top_n=3):
    """Tìm kiếm giọng nói tương đồng trong CSDL."""
    print(f"\nĐang trích xuất đặc trưng cho file query: {os.path.basename(query_audio_path)}")
    try:
        y, sr = librosa.load(query_audio_path, sr=None)
        query_mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        query_flat = query_mfccs.flatten()
    except Exception as e:
        print(f"Không thể xử lý file query: {e}")
        return []

    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT filename, feature_vector FROM audio_files WHERE feature_vector IS NOT NULL")
    all_db_features = cursor.fetchall()
    conn.close()
    
    if not all_db_features:
        print("CSDL đặc trưng trống.")
        return []

    print("Đang so sánh với CSDL...")
    similarities = []
    for filename, feature_blob in all_db_features:
        byte_io = io.BytesIO(feature_blob)
        db_mfccs = np.load(byte_io)
        db_flat = db_mfccs.flatten()
        sim_score = cosine_similarity_from_scratch(query_flat, db_flat)
        similarities.append((filename, sim_score))
        
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_n]

# ===================================================================
# PHẦN NÂNG CẤP BẮT ĐẦU TỪ ĐÂY
# ===================================================================

def get_filepath_from_db(identifier):
    """
    Thực hiện câu lệnh SQL SELECT để lấy đường dẫn file từ ID hoặc filename.
    """
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    query_path = None
    
    # Nếu identifier là một con số, tìm theo ID
    if identifier.isdigit():
        cursor.execute("SELECT filepath FROM audio_files WHERE id = ?", (int(identifier),))
        result = cursor.fetchone()
        if result:
            query_path = result[0]
    # Nếu không, tìm theo tên file
    else:
        # Thêm ký tự '%' để tìm kiếm linh hoạt (ví dụ: nhập 'voice_001' vẫn tìm ra 'female_voice_001.wav')
        search_term = f"%{identifier}%" 
        cursor.execute("SELECT filepath FROM audio_files WHERE filename LIKE ?", (search_term,))
        result = cursor.fetchone()
        if result:
            query_path = result[0]
            
    conn.close()
    return query_path

if __name__ == "__main__":
    # Tạo một vòng lặp để có thể tìm kiếm nhiều lần
    while True:
        print("\n=============================================")
        user_input = input("Nhập ID, tên file từ CSDL, hoặc đường dẫn đầy đủ để làm query (nhấn Enter để thoát): ")
        
        if not user_input:
            break # Thoát vòng lặp nếu người dùng không nhập gì

        query_path = None
        
        # Ưu tiên kiểm tra xem có phải là đường dẫn file tồn tại không
        if os.path.exists(user_input):
            query_path = user_input
        else:
            # Nếu không, thực hiện câu lệnh SQL để tìm trong CSDL
            print(f"'{user_input}' không phải là đường dẫn hợp lệ. Đang tìm trong CSDL...")
            query_path = get_filepath_from_db(user_input)

        # Sau khi đã có đường dẫn, tiến hành tìm kiếm
        if query_path and os.path.exists(query_path):
            top_results = find_similar_voices(query_path)
            
            print("\n--- KẾT QUẢ TÌM KIẾM ---")
            if not top_results:
                print("Không tìm thấy kết quả phù hợp.")
            else:
                for i, (filename, score) in enumerate(top_results):
                    print(f"{i+1}. File: {filename} (Độ tương đồng: {score:.4f})")
        else:
            print(f"LỖI: Không thể tìm thấy file tương ứng với '{user_input}' trong CSDL hoặc trên hệ thống.")
            