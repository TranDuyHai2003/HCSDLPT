# C:\Users\Acer\Desktop\HCSDLDPT\voice_project\search_similar.py

import librosa
import numpy as np
import os
from sklearn.metrics.pairwise import cosine_similarity

# --- Cấu hình (phải giống với extract_features.py) ---
FEATURE_DB_PATH = r"C:\Users\Acer\Desktop\HCSDLDPT\voice_project\feature_db.npz"

N_MFCC = 20
FIXED_NUM_FRAMES = 256
SAMPLE_RATE = 16000
N_FFT = 2048
HOP_LENGTH = 512
WIN_LENGTH = N_FFT
N_MELS = 128

# Hàm trích xuất đặc trưng (giống hệt trong extract_features.py)
def extract_mfcc_features(audio_path, n_mfcc, fixed_num_frames, sr, n_fft, hop_length, win_length, n_mels):
    try:
        y, current_sr = librosa.load(audio_path, sr=sr)
        # y, _ = librosa.effects.trim(y, top_db=20)
        mfccs = librosa.feature.mfcc(y=y, sr=current_sr, n_mfcc=n_mfcc,
                                     n_fft=n_fft, hop_length=hop_length,
                                     win_length=win_length, n_mels=n_mels)
        num_current_frames = mfccs.shape[1]
        if num_current_frames < fixed_num_frames:
            pad_width = fixed_num_frames - num_current_frames
            mfccs_padded = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant', constant_values=(0,))
        else:
            mfccs_padded = mfccs[:, :fixed_num_frames]
        return mfccs_padded
    except Exception as e:
        print(f"Lỗi xử lý file {os.path.basename(audio_path)}: {e}")
        return None

def find_similar_voices(query_audio_path, feature_db_path, top_n=3,
                        n_mfcc=N_MFCC, fixed_num_frames=FIXED_NUM_FRAMES,
                        sr=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH,
                        win_length=WIN_LENGTH, n_mels=N_MELS):
    """
    Tìm kiếm các giọng nói tương đồng với file query trong CSDL đặc trưng.
    """
    # 1. Tải CSDL đặc trưng
    try:
        db_data = np.load(feature_db_path, allow_pickle=True)
        # db_data.files trả về list các key (tên file) trong file .npz
        feature_db = {filename: db_data[filename] for filename in db_data.files}
        if not feature_db:
            print(f"CSDL đặc trưng tại '{feature_db_path}' rỗng.")
            return []
    except FileNotFoundError:
        print(f"LỖI: File CSDL đặc trưng '{feature_db_path}' không tìm thấy. Hãy chạy extract_features.py trước.")
        return []
    except Exception as e:
        print(f"Lỗi khi tải CSDL đặc trưng: {e}")
        return []

    # 2. Trích xuất đặc trưng cho file query
    print(f"\nĐang trích xuất đặc trưng cho file query: {os.path.basename(query_audio_path)}")
    query_features = extract_mfcc_features(query_audio_path, n_mfcc, fixed_num_frames, sr, n_fft, hop_length, win_length, n_mels)
    
    if query_features is None:
        print("Không thể trích xuất đặc trưng cho file query.")
        return []
    # print(f"  -> Đặc trưng query shape: {query_features.shape}")

    # 3. Làm phẳng ma trận MFCC để tính cosine similarity
    # Ma trận MFCC có shape (n_mfcc, fixed_num_frames)
    # Làm phẳng thành vector (1, n_mfcc * fixed_num_frames)
    query_features_flat = query_features.flatten().reshape(1, -1)
    
    similarities = []
    print("Đang so sánh với CSDL...")
    for filename, db_mfccs in feature_db.items():
        # db_mfccs cũng có shape (n_mfcc, fixed_num_frames)
        db_mfccs_flat = db_mfccs.flatten().reshape(1, -1)
        
        # Tính cosine similarity
        sim = cosine_similarity(query_features_flat, db_mfccs_flat)[0][0]
        similarities.append((filename, sim))
        
    # 4. Sắp xếp theo độ tương đồng giảm dần
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    return similarities[:top_n]

if __name__ == "__main__":
    print("--- Bắt đầu quá trình tìm kiếm giọng nói tương đồng ---")
    
    # --- Cấu hình file query cho DEMO ---
    # Thay đổi đường dẫn này đến file audio bạn muốn dùng để tìm kiếm
    # Ví dụ: lấy một file từ chính CSDL để test
    # QUERY_AUDIO_FILE_PATH = r"C:\Users\Acer\Desktop\HCSDLDPT\folder giong noi\female_voice_001.wav"
    
    # Hoặc một file mới bạn chuẩn bị
    # QUERY_AUDIO_FILE_PATH = r"C:\Users\Acer\Desktop\HCSDLDPT\voice_project\my_test_query.wav" 
    
    # Yêu cầu người dùng nhập đường dẫn file query
    query_path_input = input("Nhập đường dẫn đầy đủ đến file âm thanh query (ví dụ: D:\\audio\\test.wav): ")
    
    if not os.path.isfile(query_path_input):
        print(f"LỖI: File query '{query_path_input}' không tồn tại.")
    else:
        top_results = find_similar_voices(
            query_audio_path=query_path_input,
            feature_db_path=FEATURE_DB_PATH,
            top_n=3 # Tìm top 3
        )
        
        if top_results:
            print(f"\n--- Top {len(top_results)} kết quả giống nhất với '{os.path.basename(query_path_input)}' ---")
            for i, (filename, score) in enumerate(top_results):
                print(f"{i+1}. File: {filename} (Độ tương đồng: {score:.4f})")
        else:
            print("Không tìm thấy kết quả nào hoặc có lỗi xảy ra.")
            
    print("\n--- Kết thúc quá trình tìm kiếm ---")