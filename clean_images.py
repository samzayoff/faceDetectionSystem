import os
from PIL import Image

def remove_corrupt_images(folder):
    for root, _, files in os.walk(folder):
        for file in files:
            filepath = os.path.join(root, file)
            try:
                img = Image.open(filepath)
                img.verify()
            except Exception:
                print(f"Removing corrupt image: {filepath}")
                os.remove(filepath)

# Run on both mask and no-mask datasets
remove_corrupt_images(r"E:\Sameer\ArchTech\data\with_mask")
remove_corrupt_images(r"E:\Sameer\ArchTech\data\without_mask")
