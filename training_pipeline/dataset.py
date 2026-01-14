import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class ImageDatasetFromCSV(Dataset):
    def __init__(self, csv_path, transforms=None):
        self.df = pd.read_csv(csv_path)
        self.transforms = transforms

        required = {"image_path", "label"}
        if not required.issubset(self.df.columns):
            raise ValueError(f"CSV must contain columns: {required}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(row["image_path"]).convert("RGB")
        label = int(row["label"])

        if self.transforms:
            image = self.transforms(image)

        return image, label
