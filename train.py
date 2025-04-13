import os, sys, random, time, glob, math
from pathlib import Path

import yaml
import torch
torch.backends.cudnn.benchmark = True
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

from baseline.tiny_mbv4 import TinyLPR
from dataloader import LPRDataset


# load config
cfg = yaml.load(open('config.yaml', 'r'), Loader=yaml.FullLoader)
num_epochs = cfg['epochs']
eval_freq = cfg['eval_freq']
bs = cfg['batch_size']
seed = cfg['seed']
print(cfg)

# set seed
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

train_loader = DataLoader(LPRDataset(**cfg['train']), batch_size=bs, shuffle=True)
val_loader = DataLoader(LPRDataset(**cfg['val']), batch_size=bs, shuffle=False)

# model = LPR_model(1, 68, 96, 32, 8).cuda() for baseline raw
model = TinyLPR().cuda()
if cfg['checkpoint_path'] != '':
    print(f"\33[1;32mLoading checkpoint from {cfg['checkpoint_path']}\33[0m")
    model.load_state_dict(torch.load(cfg['checkpoint_path'], weights_only=True), strict=False)

train_loader = DataLoader(LPRDataset(**cfg['train']), batch_size=bs, shuffle=True)
val_loader = DataLoader(LPRDataset(**cfg['val']), batch_size=bs, shuffle=False)

optimizer = optim.NAdam(model.parameters(), lr=cfg['lr'])

# fn to eval pred and labels
def eval_model(model, data, labels):

    def eval_fn(preds, labels):
        preds = torch.argmax(preds, dim=-1)
        return torch.mean(
            torch.all(preds == labels, dim=-1).float()
        )

    with torch.no_grad():
        outputs = model(data)
        acc = eval_fn(outputs, labels)
    return acc


# loss ce for each label channel
class Losses(nn.Module):
    def __init__(self):
        super(Losses, self).__init__()

    def forward(self, preds, labels):
        preds = preds.reshape(-1, preds.size(-1))
        labels = labels.reshape(-1)
        return nn.CrossEntropyLoss()(preds, labels)


for epoch in range(1, num_epochs+1):
    model.train()
    pbar = tqdm(train_loader, total=len(train_loader))
    for inputs, labels in pbar:
        # data to cuda
        inputs = inputs.cuda()
        labels = labels.cuda()

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = Losses()(outputs, labels)
        loss.backward()
        optimizer.step()
        pbar.set_description(f'Epoch [{epoch}/{num_epochs}], Loss: {loss.item():.4f}')

    if epoch % eval_freq == 0:
        acc = []
        model.eval()
        for inputs, labels in val_loader:
            # data to cuda
            inputs = inputs.cuda()
            labels = labels.cuda()
            acc.append(eval_model(model, inputs, labels))

        acc = torch.tensor(acc).mean()
        Path('checkpoints/').mkdir(parents=True, exist_ok=True)
        print(f'\33[1;32mEpoch [{epoch}/{num_epochs}], Accuracy: {acc.item():.4f}\33[0m')
        torch.save(model.state_dict(), f'checkpoints/model_{epoch}_acc_{acc.item():.4f}.pth')
