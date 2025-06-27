import librosa
import numpy as np
import os
import glob

# --- Cấu hình ---
AUDIO_FOLDER_PATH = r"D:\\HCSDLDPT\\folder giong noi" # Đường dẫn tới folder chứa file wav
OUTPUT_DB_PATH = r"D:\\HCSDLDPT\\feature_db.npz"

N_MFCC = 20             # Số lượng hệ số MFCC (thử 13, 20, 40, hoặc 64)
FIXED_NUM_FRAMES = 256  # Số lượng frame cố định sau khi pad/truncate
SAMPLE_RATE = 16000     # Tần số lấy mẫu mong muốn (ví dụ 16kHz, 22.05kHz)
                          # Nếu None, librosa sẽ giữ nguyên tần số gốc của file
# Tham số cho trích xuất MFCC (có thể điều chỉnh nếu cần)
N_FFT = 2048            # Độ dài cửa sổ FFT
HOP_LENGTH = 512        # Bước nhảy giữa các frame
WIN_LENGTH = N_FFT      # Độ dài cửa sổ phân tích (thường bằng N_FFT)
N_MELS = 128            # Số lượng bộ lọc Mel (thường nhiều hơn N_MFCC)


def extract_mfcc_features(audio_path, n_mfcc, fixed_num_frames, sr, n_fft, hop_length, win_length, n_mels):
    """
    Trích xuất đặc trưng MFCC từ một file audio và chuẩn hóa số lượng frame.
    """
    try:
        # 1. Đọc file audio
        y, current_sr = librosa.load(audio_path, sr=sr) # sr=None để giữ nguyên, hoặc đặt cố định
                                                       # Nếu đặt sr cố định, librosa sẽ tự resample

        # (Tùy chọn) Loại bỏ khoảng lặng đầu cuối
        # y, _ = librosa.effects.trim(y, top_db=20)

        # 2. Trích xuất MFCCs
        # librosa.feature.mfcc trả về shape: (n_mfcc, num_frames)
        mfccs = librosa.feature.mfcc(y=y, sr=current_sr, n_mfcc=n_mfcc,
                                     n_fft=n_fft, hop_length=hop_length,
                                     win_length=win_length, n_mels=n_mels)

        # 3. Chuẩn hóa số lượng frames
        num_current_frames = mfccs.shape[1]
        if num_current_frames < fixed_num_frames:
            # Pad nếu số frames ít hơn fixed_num_frames
            pad_width = fixed_num_frames - num_current_frames
            # Pad theo chiều frame (chiều thứ 2), không pad theo chiều MFCC (chiều thứ 1)
            mfccs_padded = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant', constant_values=(0,))
        else:
            # Cắt nếu số frames nhiều hơn fixed_num_frames
            mfccs_padded = mfccs[:, :fixed_num_frames]
        
        return mfccs_padded
    except Exception as e:
        print(f"Lỗi xử lý file {os.path.basename(audio_path)}: {e}")
        return None

def build_feature_database(audio_folder, output_path, n_mfcc, fixed_num_frames, sr, n_fft, hop_length, win_length, n_mels):
    """
    Xây dựng cơ sở dữ liệu đặc trưng từ tất cả các file audio trong một folder.
    """
    feature_db = {}
    # Tìm tất cả các file .wav trong folder
    audio_files = glob.glob(os.path.join(audio_folder, "*.wav"))
    
    if not audio_files:
        print(f"Không tìm thấy file .wav nào trong thư mục: {audio_folder}")
        return

    print(f"Bắt đầu xây dựng CSDL đặc trưng từ {len(audio_files)} file...")
    for i, audio_file in enumerate(audio_files):
        filename = os.path.basename(audio_file)
        print(f"Đang xử lý file {i+1}/{len(audio_files)}: {filename}")
        
        features = extract_mfcc_features(audio_file, n_mfcc, fixed_num_frames, sr, n_fft, hop_length, win_length, n_mels)
        
        if features is not None:
            feature_db[filename] = features
            # print(f"  -> Trích xuất thành công, shape: {features.shape}")
        else:
            print(f"  -> Bỏ qua file {filename} do lỗi.")
            
    if not feature_db:
        print("Không có đặc trưng nào được trích xuất. Kiểm tra lại file audio hoặc thông báo lỗi.")
        return

    # Lưu CSDL đặc trưng (dưới dạng nén .npz)
    # **feature_db giải nén dictionary thành các keyword arguments
    np.savez_compressed(output_path, **feature_db)
    print(f"CSDL đặc trưng đã được lưu vào: {output_path}")
    print(f"Tổng số file được xử lý thành công: {len(feature_db)}")

if __name__ == "__main__":
    print("--- Bắt đầu quá trình xây dựng CSDL đặc trưng ---")
    
    # Kiểm tra xem thư mục audio có tồn tại không
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