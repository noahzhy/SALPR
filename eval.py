import os, sys, random, time, glob, math
from pathlib import Path

import yaml
import torch
import numpy as np
from torch.utils.data import DataLoader
from PIL import Image
from tqdm import tqdm

from model import LPR_model
from dataloader import LPRDataset
from new_model.torch_model import TinyLPR


# load config
cfg = yaml.load(open('config.yaml', 'r'), Loader=yaml.FullLoader)
num_epochs = cfg['epochs']
eval_freq = cfg['eval_freq']
bs = cfg['batch_size']
print(cfg)

test_dataset = LPRDataset(**cfg['test'])
test_loader = DataLoader(test_dataset, batch_size=bs, shuffle=False)

# model = LPR_model(1, 68, 96, 32, 8).cuda()
model = TinyLPR().cuda()
# load model from checkpoint given by path
model.load_state_dict(torch.load('backup/model_100_acc_0.9915.pth', weights_only=True))
model.eval()

# export model to onnx
import onnx

dummy_input = torch.randn(1, 1, 32, 96).cuda()
torch.onnx.export(model, dummy_input,
    "model.onnx",
    verbose=False,
    input_names=['input'],
    output_names=['output']
)


# random pick one image from test dataset to test
def test_model(model, image_path):
    image = Image.open(image_path).convert('L')
    image = image.resize((96, 32))
    image = torch.from_numpy(np.array(image) / 255.).unsqueeze(0).unsqueeze(0).float().cuda()
    output = model(image)
    pred = torch.argmax(output, dim=-1).squeeze(0).cpu().numpy()
    return pred


# fn to eval pred and labels
def eval_model(model, loader):
    with torch.no_grad():
        outputs = []
        labels = []
        for data, label in tqdm(loader):
            data, label = data.cuda(), label.cuda()
            output = model(data)
            outputs.append(output)
            labels.append(label)
        outputs = torch.cat(outputs, dim=0)
        labels = torch.cat(labels, dim=0)
        acc = torch.mean(
            torch.all(torch.argmax(outputs, dim=-1) == labels, dim=-1).float()
        )
    return acc


if __name__ == '__main__':
    acc = eval_model(model, test_loader)
    print(f'Accuracy: {acc.item():.4f}')

    # img_paths = glob.glob('/workspace/datasets/lpr/images/val/*.jpg')
    # img_path = random.choice(img_paths)
    # print(img_path)
    # print(test_model(model, img_path))
