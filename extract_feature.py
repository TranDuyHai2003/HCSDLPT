import librosa
import numpy as np
import os
import glob

# --- Cấu hình ---
AUDIO_FOLDER_PATH = r"D:\\HCSDLDPT\\folder giong noi"
OUTPUT_DB_PATH = r"D:\\HCSDLDPT\\feature_db.npz"

N_MFCC = 64
FIXED_NUM_FRAMES = 256
SAMPLE_RATE = 16000
N_FFT = 2048
HOP_LENGTH = 512
WIN_LENGTH = N_FFT
N_MELS = 128

def extract_mfcc_features(audio_path, n_mfcc, fixed_num_frames, sr, n_fft, hop_length, win_length, n_mels):
    """
    Trích xuất đặc trưng MFCC và trả về một VECTOR 1D đã được làm phẳng.
    """
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
        
        # Làm phẳng ma trận (256, 64) thành vector 1D (16384,) trước khi trả về.
        features_flat = mfccs_transposed.flatten()
        
        return features_flat
        
    except Exception as e:
        print(f"Lỗi xử lý file {os.path.basename(audio_path)}: {e}")
        return None

def build_feature_database(audio_folder, output_path, n_mfcc, fixed_num_frames, sr, n_fft, hop_length, win_length, n_mels):
    """
    Xây dựng cơ sở dữ liệu đặc trưng từ tất cả các file audio trong một folder.
    """
    feature_db = {}
    audio_files = glob.glob(os.path.join(audio_folder, "*.wav"))
    
    if not audio_files:
        print(f"Không tìm thấy file .wav nào trong thư mục: {audio_folder}")
        return

    print(f"Bắt đầu xây dựng CSDL đặc trưng từ {len(audio_files)} file...")
    for i, audio_file in enumerate(audio_files):
        filename = os.path.basename(audio_file)
        print(f"Đang xử lý file {i+1}/{len(audio_files)}: {filename}")
        
        # Hàm này bây giờ trả về một vector 1D
        features = extract_mfcc_features(audio_file, n_mfcc, fixed_num_frames, sr, n_fft, hop_length, win_length, n_mels)
        
        if features is not None:
            feature_db[filename] = features
        else:
            print(f"  -> Bỏ qua file {filename} do lỗi.")
            
    if not feature_db:
        print("Không có đặc trưng nào được trích xuất.")
        return

    np.savez_compressed(output_path, **feature_db)
    print(f"CSDL đặc trưng đã được lưu vào: {output_path}")
    print(f"Tổng số file được xử lý thành công: {len(feature_db)}")

if __name__ == "__main__":
    print("--- Bắt đầu quá trình xây dựng CSDL đặc trưng ---")
    
    if not os.path.isdir(AUDIO_FOLDER_PATH):
        print(f"LỖI: Thư mục audio '{AUDIO_FOLDER_PATH}' không tồn tại.")
    else:
        build_feature_database(
            audio_folder=AUDIO_FOLDER_PATH,
            output_path=OUTPUT_DB_PATH,
            n_mfcc=N_MFCC,
            fixed_num_frames=FIXED_NUM_FRAMES,
            sr=SAMPLE_RATE,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            win_length=WIN_LENGTH,
            n_mels=N_MELS
        )
    print("--- Kết thúc quá trình xây dựng CSDL đặc trưng ---")