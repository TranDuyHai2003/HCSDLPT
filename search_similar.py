import librosa
import numpy as np
import os

# --- Cấu hình (phải giống hệt với extract_features.py) ---
FEATURE_DB_PATH = r"D:\\HCSDLDPT\\feature_db.npz"

N_MFCC = 64
FIXED_NUM_FRAMES = 256
SAMPLE_RATE = 16000
N_FFT = 2048
HOP_LENGTH = 512
WIN_LENGTH = N_FFT
N_MELS = 128

# Hàm trích xuất đặc trưng (phải là bản sao y hệt trong extract_features.py đã được sửa)
def extract_mfcc_features(audio_path, n_mfcc, fixed_num_frames, sr, n_fft, hop_length, win_length, n_mels):
    try:
        y, current_sr = librosa.load(audio_path, sr=sr)
        mfccs = librosa.feature.mfcc(y=y, sr=current_sr, n_mfcc=n_mfcc,
                                     n_fft=n_fft, hop_length=hop_length,
                                     win_length=win_length, n_mels=n_mels)
        num_current_frames = mfccs.shape[1]
        if num_current_frames < fixed_num_frames:
            pad_width = fixed_num_frames - num_current_frames
            mfccs_padded = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant', constant_values=(0,))
        else:
            mfccs_padded = mfccs[:, :fixed_num_frames]
        
        mfccs_transposed = mfccs_padded.T
        
        features_flat = mfccs_transposed.flatten()
        return features_flat
        
    except Exception as e:
        print(f"Lỗi xử lý file {os.path.basename(audio_path)}: {e}")
        return None

def calculate_cosine_similarity(vec1, vec2):
    """
    Tính toán độ tương đồng cosine giữa hai vector 1D.
    """
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    
    if norm_vec1 == 0 or norm_vec2 == 0:
        return 0.0
        
    return dot_product / (norm_vec1 * norm_vec2)

def find_similar_voices(query_audio_path, feature_db_path, top_n=3):
    """
    Tìm kiếm các giọng nói tương đồng với file query trong CSDL đặc trưng.
    """
    try:
        # call tu db
        db_data = np.load(feature_db_path, allow_pickle=True)
        feature_db = {filename: db_data[filename] for filename in db_data.files}
        if not feature_db:
            print(f"CSDL đặc trưng tại '{feature_db_path}' rỗng.")
            return []
    except FileNotFoundError:
        print(f"LỖI: File CSDL đặc trưng '{feature_db_path}' không tìm thấy.")
        return []

    print(f"\nĐang trích xuất đặc trưng cho file query: {os.path.basename(query_audio_path)}")
    # Hàm này bây giờ trả về một vector 1D đã được làm phẳng
    query_features_flat = extract_mfcc_features(query_audio_path, N_MFCC, FIXED_NUM_FRAMES, SAMPLE_RATE, N_FFT, HOP_LENGTH, WIN_LENGTH, N_MELS)
    
    if query_features_flat is None:
        print("Không thể trích xuất đặc trưng cho file query.")
        return []

    similarities = []
    print("Đang so sánh với CSDL...")
    for filename, db_vector in feature_db.items():
        sim = calculate_cosine_similarity(query_features_flat, db_vector)
        similarities.append((filename, sim))
        
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    return similarities[:top_n]

if __name__ == "__main__":
    print("--- Bắt đầu quá trình tìm kiếm giọng nói tương đồng ---")
    
    query_path_input = input("Nhập đường dẫn đầy đủ đến file âm thanh query: ")
    
    if not os.path.isfile(query_path_input):
        print(f"LỖI: File query '{query_path_input}' không tồn tại.")
    else:
        top_results = find_similar_voices(
            query_audio_path=query_path_input,
            feature_db_path=FEATURE_DB_PATH,
            top_n=3
        )
        
        if top_results:
            print(f"\n--- Top {len(top_results)} kết quả giống nhất với '{os.path.basename(query_path_input)}' ---")
            for i, (filename, score) in enumerate(top_results):
                print(f"{i+1}. File: {filename} (Độ tương đồng: {score:.4f})")
        else:
            print("Không tìm thấy kết quả nào hoặc có lỗi xảy ra.")
            
    print("\n--- Kết thúc quá trình tìm kiếm ---")