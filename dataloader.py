import os, glob, time, random, math

import yaml
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

# load the image size config from the config file via yaml
cfg = yaml.load(open('config.yaml', 'r'), Loader=yaml.FullLoader)
# print(cfg)


def load_label(txt_path):
    # keep first element of each line
    with open(txt_path, 'r') as f:
        labels = f.readlines()
    labels = [label.strip().split(' ')[0] for label in labels]
    return labels


# Define your transforms
trans = transforms.Compose([
    transforms.RandomAutocontrast(),
    transforms.RandomSolarize(threshold=128),
    transforms.RandomAdjustSharpness(sharpness_factor=2),
    transforms.RandomAutocontrast(),
    transforms.RandomEqualize(),
    transforms.AutoAugment(),
])


class LPRDataset(Dataset):
    def __init__(self, image_dir, maxT=8, data_aug=True):
        self.image_dir = image_dir
        self.maxT = maxT
        self.data_aug = data_aug
        self.transform = trans
        self.image_files = glob.glob(os.path.join(image_dir, '*.jpg'))

        self.tensor2gray = transforms.Compose([
            transforms.Grayscale(),
            transforms.ToTensor(),
        ])

        # # keep 32 only, test only
        # self.image_files = self.image_files[:32]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # new a tensor to store the labels which length is maxT
        labels = torch.zeros(self.maxT, dtype=torch.long)

        img_name = self.image_files[idx]
        image = Image.open(img_name)
        image = transforms.Resize(cfg['img_size'], antialias=True)(image)

        txt_name = img_name.replace('images', 'labels').replace('.jpg', '.txt')
        labs = load_label(txt_name)
        # add 0 to the label if the label is less than maxT
        # labels[:len(labs)] = torch.tensor([int(lab) for lab in labs])
        labels[self.maxT - len(labs):] = torch.tensor([int(lab) for lab in labs])

        if self.data_aug: image = self.transform(image)

        return self.tensor2gray(image), labels


if __name__ == '__main__':
    dataset = LPRDataset(image_dir='datasets/lpr/images/val', data_aug=True)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    from matplotlib import pyplot as plt

    # Example usage
    for images, labels in dataloader:
        print(images.shape, labels)
        plt.imshow(images[0].permute(1, 2, 0))
        plt.savefig('test.png')
        break
