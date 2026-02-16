import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class TextureDataset(Dataset):
    def __init__(self, root):
        self.paths = []
        self.labels = []
        self.classes = os.listdir(root)

        for i, cls in enumerate(self.classes):
            cls_folder = os.path.join(root, cls)
            for img in os.listdir(cls_folder):
                self.paths.append(os.path.join(cls_folder, img))
                self.labels.append(i)

        self.transform = transforms.Compose([
            transforms.Resize((128,128)),
            transforms.ToTensor(),
            transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        img = self.transform(img)
        label = self.labels[idx]
        return img, label
    