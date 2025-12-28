import os
import cv2
import matplotlib.pyplot as plt

def generate_evaluation_grid():
    base_dir = "data/samples"
    categories = ['best', 'average', 'worst']
    output_path = "evaluation_grid.png"
    
    # Setup subplot: 3 baris (kategori), 5 kolom (sampel)
    fig, axes = plt.subplots(3, 5, figsize=(20, 12))
    plt.subplots_adjust(wspace=0.05, hspace=0.2)

    for row_idx, cat in enumerate(categories):
        cat_dir = os.path.join(base_dir, cat)
        if not os.path.exists(cat_dir):
            print(f"Folder {cat} tidak ditemukan!")
            continue
            
        # List gambar di folder kategori (ambil max 5)
        images = sorted(os.listdir(cat_dir))[:5]
        
        for col_idx, img_name in enumerate(images):
            img_path = os.path.join(cat_dir, img_name)
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            ax = axes[row_idx, col_idx]
            ax.imshow(img)
            
            # Tambahkan label kategori hanya di kolom pertama
            if col_idx == 0:
                ax.set_ylabel(cat.upper(), fontsize=16, fontweight='bold')
            
            ax.set_xticks([])
            ax.set_yticks([])
            
            # Tambahkan info IoU dari nama file (opsional)
            # Karena nama file kita: category_idx_filename
            ax.set_title(img_name.split('_')[-1], fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Grid visualisasi berhasil disimpan di {output_path}")
    plt.show()

if __name__ == "__main__":
    generate_evaluation_grid()