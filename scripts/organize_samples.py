import pandas as pd
import os
import shutil

def organize_samples():
    # 1. Setup Path
    csv_path = "results_eval.csv"
    source_dir = "data/localizations"  # Folder tempat kamu simpan hasil plot sebelumnya
    target_base = "data/samples"
    
    if not os.path.exists(csv_path):
        print("File CSV tidak ditemukan! Jalankan compare.py dulu.")
        return

    # Buat folder sampel
    categories = ['best', 'average', 'worst']
    for cat in categories:
        os.makedirs(os.path.join(target_base, cat), exist_ok=True)

    # 2. Baca Data
    df = pd.read_csv(csv_path)
    
    # Sort berdasarkan IoU
    df_sorted = df.sort_values(by='avg_iou', ascending=False).reset_index(drop=True)
    total_data = len(df_sorted)

    # 3. Ambil Index
    # Best: 5 teratas
    best_idx = df_sorted.head(5).index.tolist()
    
    # Worst: 5 terbawah
    worst_idx = df_sorted.tail(5).index.tolist()
    
    # Average: 5 di tengah-tengah (median)
    mid = total_data // 2
    avg_idx = list(range(mid - 2, mid + 3))

    # 4. Fungsi untuk copy file
    def copy_files(indices, category):
        print(f"\nCopying {category} cases:")
        for idx in indices:
            row = df_sorted.iloc[idx]
            filename = row['filename']
            # Sesuaikan dengan prefix 'eval_' yang kita buat di compare.py tadi
            src_file = f"eval_{filename}" 
            src_path = os.path.join(source_dir, src_file)
            dest_path = os.path.join(target_base, category, f"{category}_{idx}_{filename}")
            
            if os.path.exists(src_path):
                shutil.copy(src_path, dest_path)
                print(f"  [v] {filename} (IoU: {row['avg_iou']:.4f})")
            else:
                print(f"  [x] File {src_file} tidak ditemukan di {source_dir}")

    # Eksekusi
    copy_files(best_idx, 'best')
    copy_files(avg_idx, 'average')
    copy_files(worst_idx, 'worst')

    print(f"\nSelesai! Cek folder: {target_base}")

if __name__ == "__main__":
    organize_samples()