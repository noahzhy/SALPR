import os, glob, random, math
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.v2.functional as TF
import torchvision.transforms.v2 as transforms


class ResizeWithPad:
    def __init__(self, size:tuple, interpolation=Image.BILINEAR):
        self.size = size # H, W
        self.interpolation = interpolation

    def __call__(self, img:Image.Image):
        w, h = img.size
        target_h, target_w = self.size
        scale = min(target_w / w, target_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        img = img.resize((new_w, new_h), self.interpolation)
        pad_w = target_w - new_w
        pad_h = target_h - new_h
        left = pad_w // 2
        top = pad_h // 2
        right = pad_w - left
        bottom = pad_h - top
        img = TF.pad(img, (left, top, right, bottom), fill=0, padding_mode='constant')
        return img


class MaskDataset(Dataset):
    def __init__(self, image_dir, image_size=(128, 128), data_aug=True, seed=None):
        if seed is not None:
            random.seed(seed)
            torch.manual_seed(seed)

        self.root_dir = image_dir
        self.image_paths = glob.glob(os.path.join(self.root_dir, '*.jpg'))
        self.image_size = image_size  # (H, W)
        self.augment = data_aug

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        lbl_path = img_path.replace('.jpg', '.png')

        img = Image.open(img_path).convert('RGB')
        lbl = Image.open(lbl_path).convert('L')

        # check shape of img and lbl
        if img.size != lbl.size:
            # resize label to match image
            lbl = lbl.resize(img.size, Image.NEAREST)

        if self.augment:
            img = ResizeWithPad(self.image_size)(img)
            lbl = ResizeWithPad(self.image_size, interpolation=Image.NEAREST)(lbl)
        else:
            img = TF.resize(img, self.image_size, interpolation=Image.BILINEAR)
            lbl = TF.resize(lbl, self.image_size, interpolation=Image.NEAREST)

        if self.augment:
            # 随机水平/垂直翻转（同步）
            if random.random() > 0.5:
                img = TF.hflip(img); lbl = TF.hflip(lbl)
            if random.random() > 0.5:
                img = TF.vflip(img); lbl = TF.vflip(lbl)

        # to tensor
        img = TF.to_tensor(img)  # [0,1], C×H×W
        lbl = TF.to_tensor(lbl)  # [0,1], 1×H×W

        if self.augment:
            # 随机色相/亮度/对比度
            if random.random() > 0.5:
                hue = random.uniform(-0.2, 0.2)
                img = TF.adjust_hue(img, hue)
            if random.random() > 0.5:
                delta = random.uniform(-0.2, 0.2)
                img = TF.adjust_brightness(img, 1 + delta)
            if random.random() > 0.5:
                factor = random.uniform(0.5, 1.5)
                img = TF.adjust_contrast(img, factor)
    
            img = torch.clamp(img, 0.0, 1.0)

            # 随机灰度
            if random.random() > 0.1:
                gray = TF.rgb_to_grayscale(img)
                img = gray.repeat(3, 1, 1)

            # 随机 gamma 校正
            if random.random() > 0.5:
                img = TF.adjust_gamma(img, gamma=random.uniform(0.5, 1.5))

            # 随机 1 - image
            if random.random() > 0.5:
                img = 1.0 - img

        # 二值化标签
        lbl = (lbl > 0).float()
        return img, lbl


if __name__ == '__main__':
    PATH = '/Users/haoyu/Documents/datasets/lpr/mini_train'
    BATCH_SIZE = 12
    height = 128
    IMAGE_SIZE = (height, height*2)

    dataset = MaskDataset(PATH, image_size=IMAGE_SIZE, data_aug=True)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    import matplotlib.pyplot as plt

    for imgs, lbls in loader:
        print(imgs.shape, lbls.shape)  # (B,3,H,W), (B,1,H,W)
        fig, axs = plt.subplots(1,2,figsize=(8,4))
        axs[0].imshow(imgs[0].permute(1,2,0).cpu().numpy())
        axs[1].imshow(lbls[0,0].cpu().numpy(), cmap='gray')
        # plt.show()
        save_path = os.path.join('test')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        plt.savefig(os.path.join(save_path, f'test_{0}.png'))
        break