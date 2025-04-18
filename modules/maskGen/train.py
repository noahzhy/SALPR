import os, time
from pathlib import Path

import yaml
import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import TinyUNet
from dataloader import MaskDataset

# load config
cfg = yaml.load(open('config.yaml', 'r'), Loader=yaml.FullLoader)
print(cfg)
torch.manual_seed(cfg['seed'])

# setup device: prefer CUDA, then Apple MPS, else CPU
if torch.cuda.is_available():
    device = torch.device('cuda')
# elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
#     device = torch.device('mps')
else:
    device = torch.device('cpu')
print(f"Using device: {device}")

if device.type == 'cuda':
    torch.cuda.manual_seed(cfg['seed'])
    torch.backends.cudnn.benchmark = True

# datasets & loaders
train_ds = MaskDataset(**cfg['train'])
test_ds  = MaskDataset(**cfg['test'])
val_ds   = MaskDataset(**cfg['val'])
train_loader = DataLoader(train_ds, batch_size=cfg['batch_size'], shuffle=True,  num_workers=4)
test_loader  = DataLoader(test_ds,  batch_size=cfg['batch_size'], shuffle=False, num_workers=4)
val_loader   = DataLoader(val_ds,   batch_size=cfg['batch_size'], shuffle=False, num_workers=4)

# model, optimizer, scheduler, amp scaler
model = TinyUNet().to(device)
if 'checkpoint_path' in cfg and cfg['checkpoint_path']:
    print(f"Loading checkpoint from {cfg['checkpoint_path']}")
    model.load_state_dict(torch.load(cfg['checkpoint_path'], map_location=device), strict=False)

optimizer = optim.NAdam(model.parameters(), lr=cfg['lr'])
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg['epochs'])
scaler = torch.cuda.amp.GradScaler()


def tversky_loss(inputs, targets, alpha=0.5, beta=0.5, smooth=1e-6):
    """Tversky loss for imbalanced segmentation."""
    probs = torch.sigmoid(inputs)
    dims = tuple(range(1, targets.ndimension()))
    TP = (probs * targets).sum(dim=dims)
    FP = ((1 - targets) * probs).sum(dim=dims)
    FN = (targets * (1 - probs)).sum(dim=dims)
    tversky_index = (TP + smooth) / (TP + alpha * FN + beta * FP + smooth)
    return (1 - tversky_index).mean()


def mIoU_acc(pred, target):
    """Mean Intersection over Union (IoU) accuracy."""
    pred = np.array(pred)
    target = np.array(target)
    intersection = np.logical_and(pred, target)
    return intersection.sum() / (pred.sum() + target.sum() - intersection.sum() + 1e-6)

# loss & eval fn
criterion = tversky_loss

def eval_model(model, data, labels):
    with torch.no_grad():
        output = model(data)
        output = torch.sigmoid(output)
        pred = torch.argmax(output, dim=1)
        pred = pred.cpu().numpy()
        labels = labels.cpu().numpy()
        acc = mIoU_acc(pred, labels)

    return torch.tensor(acc, device=device)


# fn to eval pred and labels
def test_model(model_path, dataloader, device):
    model = TinyUNet(log_output=True).to(device)
    print(f"Loading checkpoint from {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
    model.eval()
    model.to(device)

    with torch.no_grad():
        outputs = []
        labels = []
        for data, label in tqdm(dataloader):
            data, label = data.to(device), label.to(device)
            output = model(data)
            outputs.append(output)
            labels.append(label)
        outputs = torch.cat(outputs, dim=0)
        labels = torch.cat(labels, dim=0)
        acc = torch.mean(
            torch.all(torch.argmax(outputs, dim=-1) == labels, dim=-1).float()
        )
    return acc


def train_model(model, train_loader, val_loader, device, cfg):
    # training loop
    for epoch in range(1, cfg['epochs'] + 1):
        model.train()
        t0 = time.time()
        pbar = tqdm(train_loader, desc=f'Epoch {epoch}/{cfg["epochs"]}')
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                if torch.isnan(loss):
                    print("Loss is NaN, skipping this batch.")
                    continue

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                parameters=model.parameters(), 
                max_norm=2.0,
                norm_type=2,
                error_if_nonfinite=False)
            scaler.step(optimizer)
            scaler.update()
            pbar.set_postfix(loss=f'{loss.item():.4f}')

        scheduler.step()
        print(f'Epoch {epoch} training time: {time.time()-t0:.1f}s')

        if epoch % cfg['eval_freq'] == 0:
            model.eval()
            accs = []

            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                accs.append(eval_model(model, inputs, labels))

            acc = torch.stack(accs).mean().item()
            Path('checkpoints').mkdir(exist_ok=True)
            ckpt = f'checkpoints/epoch_{epoch}_acc_{acc:.4f}.pth'
            torch.save(model.state_dict(), ckpt)
            print(f'→ Eval @epoch {epoch}: Accuracy={acc:.4f}, saved to {ckpt}')

    return model


if __name__ == '__main__':
    # train model
    model = train_model(model, train_loader, val_loader, device, cfg)

    # test model
    checkpoint_path = 'checkpoints/*.pth'
    acc = test_model(checkpoint_path, test_loader, device)
    print(f'Test Accuracy: {acc.item():.4f}')

    # rename checkpoint to save folder
    save_path = f"saved/model_{cfg['epochs']}_acc_{acc.item():.4f}.pth"
    Path('saved').mkdir(exist_ok=True)
    os.rename(checkpoint_path, save_path)
    print(f"Model saved to {save_path}")
