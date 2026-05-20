import os
from torch.utils.data import Dataset
from PIL import Image
from data.transforms import train_transforms, val_transforms

CLASS_MAP = {"benign": 0, "malignant": 1, "suspicious": 2}

class ISICDataset(Dataset):
    """
    Expects folder structure:
        data/isic/train/benign/   *.jpg
        data/isic/train/malignant/*.jpg
        data/isic/train/suspicious/*.jpg
        data/isic/val/  ...same...
    """
    def __init__(self, root: str, split: str = "train"):
        self.samples = []
        self.transform = train_transforms if split == "train" else val_transforms
        split_dir = os.path.join(root, split)

        for label_name, label_idx in CLASS_MAP.items():
            folder = os.path.join(split_dir, label_name)
            if not os.path.isdir(folder):
                continue
            for fname in os.listdir(folder):
                if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    self.samples.append((os.path.join(folder, fname), label_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        return self.transform(image), label
