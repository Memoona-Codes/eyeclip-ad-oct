import os
import glob
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class OCTDataset(Dataset):
    def __init__(self, data_dir="data", transform=None):
        self.data_dir = data_dir

        # Search recursively for all .png and .PNG files inside data_dir
        all_pngs = (
            glob.glob(os.path.join(data_dir, "**/*.png"), recursive=True) +
            glob.glob(os.path.join(data_dir, "**/*.PNG"), recursive=True)
        )

        # Filter out generated synthetic outputs or visual comparison folders
        self.image_paths = sorted([
            p for p in all_pngs 
            if "synthetic" not in p.lower() and "visual_comparisons" not in p.lower()
        ])

        # Fallback: search root directory if no images found in data_dir
        if len(self.image_paths) == 0:
            all_pngs = glob.glob("**/*.png", recursive=True) + glob.glob("**/*.PNG", recursive=True)
            self.image_paths = sorted([
                p for p in all_pngs 
                if "synthetic" not in p.lower() and "visual_comparisons" not in p.lower()
            ])

        if len(self.image_paths) == 0:
            raise RuntimeError(f"No images found in '{data_dir}' or subdirectories!")

        print(f"📦 Loaded dataset with {len(self.image_paths)} images.")

        if transform is not None:
            self.transform = transform
        else:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor()
            ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        # Force 1-channel grayscale ("L")
        image = Image.open(img_path).convert("L")

        if self.transform:
            image = self.transform(image)

        return image, 0  # Dummy label for VAE
