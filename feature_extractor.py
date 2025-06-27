import os
import librosa
import numpy as np

CSV_FILE_PATH = "features.csv"
OUTPUT_CSV_FILE_PATH = "features_with_vectors.csv" 

def extract_and_save_features_minimal():
    if not os.path.exists(CSV_FILE_PATH):
        print(f"Lỗi: File '{CSV_FILE_PATH}' không tồn tại. Vui lòng chạy csv_manager.py trước.")
        return

    try:
        with open(CSV_FILE_PATH, mode='r', encoding='utf-8') as infile:
            lines_to_process = infile.readlines()
    except IOError as e:
        print(f"Lỗi khi đọc file: {e}")
        return

    header = lines_to_process[0]
    data_lines = lines_to_process[1:]

    print(f"Bắt đầu trích xuất đặc trưng cho {len(data_lines)} file...")

    try:
        with open(OUTPUT_CSV_FILE_PATH, mode='w', encoding='utf-8') as outfile:
            outfile.write(header)

            for line in data_lines:
                parts = line.strip().split(',')
                if len(parts) != 3:
                    continue
                
                filename, filepath, feature_vector_str = parts
                
                if feature_vector_str:
                    outfile.write(line)
                    continue

                try:
                    print(f"Đang xử lý file: {filename}...")
                    
                    y, sr = librosa.load(filepath, sr=None)
                    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
                    
                    feature_vector_flat = mfccs.flatten()
                    feature_vector_str_new = ";".join(map(str, feature_vector_flat))
                    
                    new_line = f"{filename},{filepath},{feature_vector_str_new}\n"
                    outfile.write(new_line)

                except Exception as e:
                    print(f"Lỗi khi xử lý file {filepath}: {e}")
                    outfile.write(line)
    except IOError as e:
        print(f"Lỗi khi ghi file tạm: {e}")
        return
    
    os.remove(CSV_FILE_PATH)
    os.rename(OUTPUT_CSV_FILE_PATH, CSV_FILE_PATH)
    
    print(f"Hoàn tất! Dữ liệu đặc trưng đã được lưu vào '{CSV_FILE_PATH}'.")

if __name__ == "__main__":
    extract_and_save_features_minimal()