import numpy as np
import csv
import os

# --- Cấu hình ---
# Đường dẫn đến file CSDL .npz của bạn (chứa các vector đã được làm phẳng)
NPZ_FILE_PATH = r"D:\\HCSDLDPT\\feature_db.npz"
# Tên file CSV tường minh sẽ được tạo ra
DETAILED_CSV_PATH = "features_detailed_explicit.csv"

# Kích thước của ma trận gốc
NUM_FRAMES = 256
NUM_MFCC = 64

def export_npz_to_detailed_csv(npz_path, csv_path):
    """
    Đọc file .npz chứa các vector đã làm phẳng và xuất ra file CSV
    với tên cột chi tiết (feature_X_frame_Y).
    """
    print(f"--- Bắt đầu quá trình xuất CSDL từ '{npz_path}' sang CSV chi tiết ---")

    # 1. Tải CSDL .npz
    try:
        db_data = np.load(npz_path)
        file_keys = db_data.files
    except FileNotFoundError:
        print(f"LỖI: Không tìm thấy file CSDL '{npz_path}'. Hãy chạy extract_features.py trước.")
        return

    print(f"Đã tải thành công {len(file_keys)} bản ghi từ CSDL.")

    # 2. Tạo dòng tiêu đề (header) một cách thông minh
    print("Đang tạo header chi tiết...")
    header = ['filename']
    # Vòng lặp ngoài cùng duyệt qua từng khung thời gian (từ 1 đến 256)
    for frame_index in range(1, NUM_FRAMES + 1):
        # Vòng lặp bên trong duyệt qua từng hệ số MFCC (từ 1 đến 64)
        for mfcc_index in range(1, NUM_MFCC + 1):
            column_name = f'feature_{mfcc_index}_frame_{frame_index}'
            header.append(column_name)
    
    print(f"Đã tạo header với {len(header)} cột.")

    # 3. Ghi dữ liệu vào file CSV
    try:
        with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            
            # Ghi dòng tiêu đề đã tạo
            writer.writerow(header)
            
            print("Đang ghi dữ liệu vào file CSV. Quá trình này có thể mất nhiều thời gian...")

            # Lặp qua từng file trong CSDL và ghi dữ liệu
            for filename in sorted(file_keys):
                # Lấy vector đặc trưng 1D
                feature_vector = db_data[filename]
                
                # Tạo một hàng để ghi: [tên file, số thứ 1, số thứ 2, ...]
                row_to_write = [filename] + feature_vector.tolist()
                
                writer.writerow(row_to_write)

        print(f"\n--- HOÀN TẤT ---")
        print(f"Đã xuất thành công CSDL ra file tường minh: '{os.path.abspath(csv_path)}'")
        print("Lưu ý: File CSV này rất lớn và có thể mở chậm trên các phần mềm bảng tính.")

    except IOError as e:
        print(f"Lỗi khi ghi file CSV: {e}")


if __name__ == "__main__":
    export_npz_to_detailed_csv(NPZ_FILE_PATH, DETAILED_CSV_PATH)