import os, sys, random, time, glob, math
from pathlib import Path

import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

from new_model.torch_model import TinyLPR
from model import LPR_model
from dataloader import LPRDataset


# load config
cfg = yaml.load(open('config.yaml', 'r'), Loader=yaml.FullLoader)
num_epochs = cfg['epochs']
eval_freq = cfg['eval_freq']
bs = cfg['batch_size']
print(cfg)

train_dataset = LPRDataset(**cfg['train'])
train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True)

val_dataset = LPRDataset(**cfg['val'])
val_loader = DataLoader(val_dataset, batch_size=bs, shuffle=False)

# model = LPR_model(1, 68, 96, 32, 8).cuda()
model = TinyLPR().cuda()
# laod model from checkpoint given by path
model.load_state_dict(torch.load(cfg['checkpoint_path'], weights_only=True))

optimizer = optim.NAdam(model.parameters(), lr=cfg['lr'])

# fn to eval pred and labels
def eval_model(model, data, labels):

    def eval_fn(preds, labels):
        preds = torch.argmax(preds, dim=-1)
        acc = torch.mean(
            torch.all(preds == labels, dim=-1).float()
        )
        return acc

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