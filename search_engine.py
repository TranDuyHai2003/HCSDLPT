import os
import librosa
import numpy as np
import math

CSV_FILE_PATH = "features.csv"

def cosine_similarity_from_scratch(vec1_flat, vec2_flat):
    dot_product = sum(x * y for x, y in zip(vec1_flat, vec2_flat))
    mag_vec1 = math.sqrt(sum(x * x for x in vec1_flat))
    mag_vec2 = math.sqrt(sum(x * x for x in vec2_flat))
    if not mag_vec1 or not mag_vec2:
        return 0.0
    return dot_product / (mag_vec1 * mag_vec2)

def find_similar_voices_from_csv(query_audio_path, top_n=3):
    print(f"\nĐang trích xuất đặc trưng cho file query: {os.path.basename(query_audio_path)}")
    try:
        y, sr = librosa.load(query_audio_path, sr=None)
        query_mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        query_flat = query_mfccs.flatten()
    except Exception as e:
        print(f"Không thể xử lý file query: {e}")
        return []

    if not os.path.exists(CSV_FILE_PATH):
        print(f"Lỗi: File '{CSV_FILE_PATH}' không tồn tại.")
        return []

    db_features = []
    try:
        with open(CSV_FILE_PATH, mode='r', encoding='utf-8') as text_file:
            lines = text_file.readlines()
            # Bỏ qua dòng tiêu đề
            for line in lines[1:]:
                parts = line.strip().split(',')
                if len(parts) == 3 and parts[2]:
                    filename = parts[0]
                    feature_vector_str = parts[2]
                    db_flat = np.array([float(x) for x in feature_vector_str.split(';')])
                    db_features.append({'filename': filename, 'vector': db_flat})
    except Exception as e:
        print(f"Lỗi khi đọc file CSV: {e}")
        return []

    print("Đang so sánh với dữ liệu từ file CSV...")
    similarities = []
    for item in db_features:
        sim_score = cosine_similarity_from_scratch(query_flat, item['vector'])
        similarities.append((item['filename'], sim_score))
        
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_n]

if __name__ == "__main__":
    QUERY_FILE = r"D:\HCSDLDPT\folder giong noi\female_voice_0023.wav"
    
    if not os.path.exists(QUERY_FILE):
        print(f"LỖI: File query không tồn tại: {QUERY_FILE}")
    else:
        top_results = find_similar_voices_from_csv(QUERY_FILE)
        
        print("\n--- KẾT QUẢ TÌM KIẾM ---")
        if not top_results:
            print("Không tìm thấy kết quả phù hợp.")
        else:
            for i, (filename, score) in enumerate(top_results):
                print(f"{i+1}. File: {filename} (Độ tương đồng: {score:.4f})")