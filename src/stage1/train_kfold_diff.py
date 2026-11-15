#!/usr/bin/env python3
import glob
import os
import random
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models
import torchvision.transforms.functional as TF

# ================= 配置区域 =================
ROOT_DIR = Path(__file__).resolve().parents[2]
LABEL_CSV = Path(os.getenv("FCR_LABEL_CSV", ROOT_DIR / "data/stage1/reference_labels.csv"))
DEFAULT_IMG_ROOT = ROOT_DIR / "data/stage1/depth_frames"
IMG_ROOT = Path(os.getenv("FCR_IMG_ROOT", DEFAULT_IMG_ROOT))
DEFAULT_BASELINE = ROOT_DIR / "data/stage1/baseline_depth.npy"
BASELINE_PATH = Path(os.getenv("FCR_BASELINE_PATH", DEFAULT_BASELINE))
OUTPUT_DIR = Path(os.getenv("FCR_KFOLD_DIR", ROOT_DIR / "results/stage1_retraining"))
MODEL_ARCH = os.getenv("FCR_MODEL_ARCH", "resnet18")  # set to resnet18_cbam to enable attention blocks

FOLDS = 5
BATCH_SIZE = 16
LR = 1e-3
EPOCHS = 100
MAX_WEIGHT = 50.0
ALL_EXPS = ["Exp01", "Exp02", "Exp03"]
BLACKLIST = [] 
# ===========================================

class Augmentation:
    def __call__(self, diff_img):
        t_diff = torch.from_numpy(diff_img).unsqueeze(0)
        if random.random() > 0.5:
            t_diff = TF.hflip(t_diff)
        if random.random() > 0.5:
            noise = torch.randn_like(t_diff) * 0.005
            t_diff = t_diff + noise
        return t_diff

class FeedDiffDataset(Dataset):
    def __init__(self, sample_list, baseline_map, is_train=True):
        self.samples = sample_list
        self.baseline_map = baseline_map.astype(np.float32) / 4000.0 
        self.is_train = is_train
        self.augmentor = Augmentation() if is_train else None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, weight = self.samples[idx]
        weight = float(weight) 
        
        curr_img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if curr_img is None: curr_img = np.zeros_like(self.baseline_map)
        
        img_curr_norm = curr_img.astype(np.float32) / 4000.0
        diff = self.baseline_map - img_curr_norm
        diff[diff < 0] = 0
        
        if self.is_train and self.augmentor:
            t_diff = self.augmentor(diff)
        else:
            t_diff = torch.from_numpy(diff).unsqueeze(0)
            
        norm_weight = weight / MAX_WEIGHT
        return t_diff, torch.tensor(norm_weight, dtype=torch.float32)

class DiffResNet(nn.Module):
    def __init__(self):
        super(DiffResNet, self).__init__()
        self.backbone = models.resnet18(weights=None)
        self.backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.feature_extractor = nn.Sequential(*list(self.backbone.children())[:-1])
        self.regressor = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        feat = self.feature_extractor(x)
        feat = feat.view(feat.size(0), -1)
        # 保持 .flatten() 以兼容 Validation Batch=1
        return self.regressor(feat).flatten()

class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.shared_mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.shared_mlp(self.avg_pool(x))
        max_out = self.shared_mlp(self.max_pool(x))
        return self.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attn = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv(attn))


class CBAM(nn.Module):
    def __init__(self, channels, reduction=16, spatial_kernel=7):
        super().__init__()
        self.channel_attn = ChannelAttention(channels, reduction)
        self.spatial_attn = SpatialAttention(spatial_kernel)

    def forward(self, x):
        x = self.channel_attn(x) * x
        x = self.spatial_attn(x) * x
        return x


class DiffResNetCBAM(nn.Module):
    """ResNet18 backbone with CBAM attention after each stage."""

    def __init__(self):
        super().__init__()
        backbone = models.resnet18(weights=None)
        backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.avgpool = backbone.avgpool

        self.cbam1 = CBAM(64)
        self.cbam2 = CBAM(128)
        self.cbam3 = CBAM(256)
        self.cbam4 = CBAM(512)

        self.regressor = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.cbam1(self.layer1(x))
        x = self.cbam2(self.layer2(x))
        x = self.cbam3(self.layer3(x))
        x = self.cbam4(self.layer4(x))
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.regressor(x).flatten()


class DiffConvNeXtTiny(nn.Module):
    """ConvNeXt-Tiny backbone adapted to single-channel input."""

    def __init__(self):
        super().__init__()
        backbone = models.convnext_tiny(weights=None)
        # stem: patchify conv to accept 1 channel
        backbone.features[0][0] = nn.Conv2d(1, 96, kernel_size=4, stride=4)
        self.backbone = backbone
        self.head = nn.Sequential(
            nn.Linear(768, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        feat = self.backbone.features(x)
        feat = F.adaptive_avg_pool2d(feat, (1, 1)).flatten(1)
        return self.head(feat).flatten()


class ECA(nn.Module):
    """Efficient Channel Attention on spatial features."""

    def __init__(self, k_size=3):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)  # B,C,1,1
        y = self.conv(y.squeeze(-1).transpose(-1, -2))
        y = self.sigmoid(y).transpose(-1, -2).unsqueeze(-1)
        return x * y.expand_as(x)


class DiffConvNeXtTinyECA(nn.Module):
    """ConvNeXt-Tiny with a final ECA block."""

    def __init__(self):
        super().__init__()
        backbone = models.convnext_tiny(weights=None)
        backbone.features[0][0] = nn.Conv2d(1, 96, kernel_size=4, stride=4)
        self.backbone = backbone
        self.eca = ECA(k_size=3)
        self.head = nn.Sequential(
            nn.Linear(768, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        feat = self.backbone.features(x)
        feat = self.eca(feat)
        feat = F.adaptive_avg_pool2d(feat, (1, 1)).flatten(1)
        return self.head(feat).flatten()

def prepare_all_samples(df, img_root, target_exps):
    all_samples = []
    print(f"正在扫描实验数据: {target_exps} ...")
    
    for _, row in df.iterrows():
        bag_name = row['image'].replace('.bag', '')
        curr_exp = next((exp for exp in target_exps if f"_{exp}_" in bag_name), None)
        if not curr_exp: continue

        try:
            weight = float(row['total'])
            if np.isnan(weight): continue
        except: continue
        
        search_path = Path(img_root) / curr_exp / f"{bag_name}_*_depth.png"
        files = sorted(glob.glob(str(search_path)))
        
        if len(files) > 0:
            mid_idx = len(files) // 2
            all_samples.append((files[mid_idx], weight))
            
    print(f"-> 总共收集到有效样本: {len(all_samples)} 个")
    return all_samples 

def run_kfold():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    arch_key = MODEL_ARCH.lower()
    if arch_key == "resnet18_cbam":
        model_suffix = "_cbam"
    elif arch_key == "convnext_tiny":
        model_suffix = "_convnext"
    elif arch_key == "convnext_tiny_eca":
        model_suffix = "_convnext_eca"
    else:
        model_suffix = ""
    
    if not BASELINE_PATH.exists():
        print("Error: 找不到基准面文件"); return
    baseline_map = np.load(str(BASELINE_PATH))
    
    df = pd.read_csv(LABEL_CSV)
    all_samples = prepare_all_samples(df, IMG_ROOT, ALL_EXPS)
    
    kf = KFold(n_splits=FOLDS, shuffle=True, random_state=42)
    fold_results = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"\n>>> 开始 {FOLDS}-Fold 交叉验证 (Device: {device}, Arch: {MODEL_ARCH}) <<<")
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(all_samples)):
        print(f"\n=== Fold {fold+1}/{FOLDS} ===")
        print(f"Train样本数: {len(train_idx)} | Val样本数: {len(val_idx)}")
        
        train_data = [all_samples[i] for i in train_idx]
        val_data = [all_samples[i] for i in val_idx]
        
        train_dataset = FeedDiffDataset(train_data, baseline_map, is_train=True)
        val_dataset = FeedDiffDataset(val_data, baseline_map, is_train=False)
        
        # === 关键修复：训练集必须 drop_last=True 防止 Batch=1 炸 BN ===
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, drop_last=True)
        # 验证集不需要 drop_last，因为我们已经处理了模型输出维度
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
        
        if arch_key == "resnet18_cbam":
            model_cls = DiffResNetCBAM
        elif arch_key == "convnext_tiny":
            model_cls = DiffConvNeXtTiny
        elif arch_key == "convnext_tiny_eca":
            model_cls = DiffConvNeXtTinyECA
        else:
            model_cls = DiffResNet
        model = model_cls().to(device)
        optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
        criterion = nn.HuberLoss(delta=0.1)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10)
        
        best_fold_r2 = -float('inf')
        
        for epoch in range(EPOCHS):
            model.train()
            train_loss = 0.0
            for b_diff, b_w in train_loader:
                b_diff, b_w = b_diff.to(device), b_w.to(device)
                optimizer.zero_grad()
                out = model(b_diff)
                loss = criterion(out, b_w)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            model.eval()
            preds, targets = [], []
            with torch.no_grad():
                for b_diff, b_w in val_loader:
                    b_diff, b_w = b_diff.to(device), b_w.to(device)
                    out = model(b_diff)
                    preds.extend((out * MAX_WEIGHT).cpu().numpy())
                    targets.extend((b_w * MAX_WEIGHT).cpu().numpy())
            
            val_r2 = r2_score(targets, preds)
            scheduler.step(val_r2)
            
            if val_r2 > best_fold_r2:
                best_fold_r2 = val_r2
                torch.save(model.state_dict(), OUTPUT_DIR / f"best_model_fold{fold+1}{model_suffix}.pth")
            
            if (epoch+1) % 20 == 0:
                print(f"  Epoch {epoch+1} | Loss: {train_loss/len(train_loader):.4f} | Val R2: {val_r2:.4f}")
                
        print(f"--> Fold {fold+1} 完成。最佳 R2: {best_fold_r2:.4f}")
        fold_results.append(best_fold_r2)
        
        # 绘图
        model.load_state_dict(torch.load(OUTPUT_DIR / f"best_model_fold{fold+1}{model_suffix}.pth"))
        model.eval()
        final_preds, final_targets = [], []
        with torch.no_grad():
            for b_diff, b_w in val_loader:
                b_diff = b_diff.to(device)
                out = model(b_diff)
                final_preds.extend((out * MAX_WEIGHT).cpu().numpy())
                final_targets.extend((b_w * MAX_WEIGHT).cpu().numpy())
        
        plt.figure(figsize=(5,5))
        plt.scatter(final_targets, final_preds, alpha=0.6)
        plt.plot([0, MAX_WEIGHT], [0, MAX_WEIGHT], 'r--')
        plt.title(f"Fold {fold+1} (R2={best_fold_r2:.3f})")
        plt.xlabel("True"); plt.ylabel("Pred")
        plt.savefig(OUTPUT_DIR / f"scatter_fold{fold+1}.png")
        plt.close()

    fold_results = np.array(fold_results)
    mean_r2 = np.mean(fold_results)
    std_r2 = np.std(fold_results)
    
    print("\n" + "="*40)
    print(">>> 5-Fold Cross Validation 最终报告 <<<")
    print(f"各折 R2: {fold_results}")
    print(f"平均 R2 (Mean): {mean_r2:.4f}")
    print(f"标准差 (Std):   {std_r2:.4f}")
    print("="*40)
    
    with open(OUTPUT_DIR / "kfold_results.txt", "w") as f:
        f.write(f"Fold Results: {fold_results}\n")
        f.write(f"Mean R2: {mean_r2:.4f}\n")
        f.write(f"Std: {std_r2:.4f}\n")

if __name__ == "__main__":
    run_kfold()
