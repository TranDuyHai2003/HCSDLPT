import os

AUDIO_FOLDER_PATH = "D:\\HCSDLDPT\\folder giong noi"
CSV_FILE_PATH = "features.csv"

def create_initial_csv_minimal():
    if os.path.exists(CSV_FILE_PATH):
        print(f"File '{CSV_FILE_PATH}' đã tồn tại. Bỏ qua việc tạo mới.")
        return

    print(f"Đang quét thư mục: {AUDIO_FOLDER_PATH}")
    
    try:
        all_files_in_dir = os.listdir(AUDIO_FOLDER_PATH)
        wav_files = [
            os.path.join(AUDIO_FOLDER_PATH, filename) 
            for filename in all_files_in_dir 
            if filename.lower().endswith('.wav')
        ]
    except FileNotFoundError:
        print(f"LỖI: Thư mục không tồn tại: {AUDIO_FOLDER_PATH}")
        return

    if not wav_files:
        print("CẢNH BÁO: Không tìm thấy file .wav nào trong thư mục!")
        return

    try:
        with open(CSV_FILE_PATH, mode='w', encoding='utf-8') as text_file:
            text_file.write('filename,filepath,feature_vector\n')
            
            for file_path in wav_files:
                filename = os.path.basename(file_path)
                line_to_write = f"{filename},{file_path},\n"
                text_file.write(line_to_write)
        
        print(f"Đã tạo file '{CSV_FILE_PATH}' với thông tin của {len(wav_files)} file.")

    except IOError as e:
        print(f"Lỗi khi ghi file: {e}")

if __name__ == "__main__":
    create_initial_csv_minimal()