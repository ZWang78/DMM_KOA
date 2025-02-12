import os
import csv
from PIL import Image
import torch
from torch.utils.data import Dataset

class KneeGradingDataset(Dataset):
    """
    Custom dataset to load pairs of images for diffusion training
    (e.g., x0 vs. x4, or any two knee X-rays).
    """
    def __init__(self, dataset, transform_train, transform_val, stage='train'):
        super(KneeGradingDataset, self).__init__()
        self.dataset = dataset
        self.stage = stage
        if self.stage == 'train':
            self.images0, self.labels0, self.images4, self.labels4 = self.load_csv("diffusion1.csv")
            self.transform = transform_train
        elif self.stage == 'val':
            self.images, self.labels = self.load_csv_val("valData_ViT_0_2.csv")
            self.transform = transform_val

    def load_csv(self, filename):
        images0, labels0, images4, labels4 = [], [], [], []
        with open(os.path.join(self.dataset, filename)) as f:
            reader = csv.reader(f)
            for row in reader:
                img0, label0, img4, label4 = row
                label0 = int(label0)
                label4 = int(label4)
                images0.append(img0)
                labels0.append(label0)
                images4.append(img4)
                labels4.append(label4)
        return images0, labels0, images4, labels4

    def load_csv_val(self, filename):
        images, labels = [], []
        with open(os.path.join(self.dataset, filename)) as f:
            reader = csv.reader(f)
            for row in reader:
                img, label = row
                label = int(label)
                images.append(img)
                labels.append(label)
        return images, labels

    def __getitem__(self, index):
        if self.stage == 'train':
            img0, label0 = self.images0[index], self.labels0[index]
            img4, label4 = self.images4[index], self.labels4[index]

            fname0 = os.path.join(self.dataset, img0)
            fname4 = os.path.join(self.dataset, img4)

            im0 = Image.open(fname0).convert("L")
            im4 = Image.open(fname4).convert("L")

            im0 = self.transform(im0)
            im4 = self.transform(im4)
            return im0, label0, im4, label4
        else:
            # for validation or test
            fname = os.path.join(self.dataset, self.images[index])
            im = Image.open(fname).convert("L")
            im = self.transform(im)
            return im, self.labels[index]

    def __len__(self):
        if self.stage == 'train':
            return len(self.images0)
        else:
            return len(self.images)
