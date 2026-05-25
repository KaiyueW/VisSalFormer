import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import json
import os

class ChartQADataset(Dataset):
    def __init__(self, img_dir, json_path, img_size=224, max_samples=None):
        self.img_dir = img_dir

        with open(json_path, 'r') as f:
            self.samples = json.load(f)
            # get {imgname, query, label}
        
        if max_samples is not None:
            self.samples = self.samples[:max_samples]

        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ]) # handle imgs to tensor

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        imgname = sample['imgname']
        query   = sample['query']
        label   = sample['label']

        img_path = os.path.join(self.img_dir, imgname) # get full img path
        img = Image.open(img_path).convert('RGB')
        img_tensor = self.transform(img)

        return img_tensor, query, imgname, label