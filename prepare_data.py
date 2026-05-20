import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split

# ── PATHS ──────────────────────────────────────────────
HAM_DIR = r"C:\Users\g\Desktop\HAM10000v2"
IMG_DIRS   = [
    os.path.join(HAM_DIR, "HAM10000_images_part_1"),
    os.path.join(HAM_DIR, "HAM10000_images_part_2"),
]
META_CSV   = os.path.join(HAM_DIR, "HAM10000_metadata.csv")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "isic")

# ── CLASS MAPPING ───────────────────────────────────────
# HAM10000 has 7 classes → we map to 3
CLASS_MAP = {
    "mel":   "malignant",   # Melanoma
    "bcc":   "malignant",   # Basal cell carcinoma
    "akiec": "suspicious",  # Actinic keratosis
    "bkl":   "suspicious",  # Benign keratosis
    "df":    "benign",      # Dermatofibroma
    "nv":    "benign",      # Melanocytic nevi
    "vasc":  "benign",      # Vascular lesions
}

def build_image_index(img_dirs):
    index = {}
    for d in img_dirs:
        for fname in os.listdir(d):
            if fname.lower().endswith(".jpg"):
                image_id = fname.replace(".jpg", "")
                index[image_id] = os.path.join(d, fname)
    return index

def prepare():
    df        = pd.read_csv(META_CSV)
    img_index = build_image_index(IMG_DIRS)

    df["label"] = df["dx"].map(CLASS_MAP)
    df = df[df["image_id"].isin(img_index)].dropna(subset=["label"])

    train_df, val_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["label"]
    )

    for split, split_df in [("train", train_df), ("val", val_df)]:
        for _, row in split_df.iterrows():
            label    = row["label"]
            src      = img_index[row["image_id"]]
            dst_dir  = os.path.join(OUTPUT_DIR, split, label)
            os.makedirs(dst_dir, exist_ok=True)
            shutil.copy2(src, os.path.join(dst_dir, os.path.basename(src)))

        print(f"{split}: {len(split_df)} images")

    print("\n✓ Dataset ready at data/isic/")
    for split in ["train", "val"]:
        for label in ["benign", "malignant", "suspicious"]:
            path  = os.path.join(OUTPUT_DIR, split, label)
            count = len(os.listdir(path)) if os.path.exists(path) else 0
            print(f"  {split}/{label}: {count} images")

if __name__ == "__main__":
    prepare()