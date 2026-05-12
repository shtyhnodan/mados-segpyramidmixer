import os
import math
import random
import re
import warnings
from pathlib import Path
from multiprocessing import freeze_support

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms
from torchvision.transforms import functional as TF
from torchvision.transforms import InterpolationMode
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

torch.backends.cudnn.benchmark = True
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SEED = 42
EPOCHS = 121
WARMUP_EPOCHS = 5
EARLY_STOP_PATIENCE = 12
BATCH_SIZE = 2
NUM_WORKERS = 2

IMAGE_SIZE = 192
PATCH_SIZE = 4
DIM = 192
DEPTH = 8
HEADS = 4
GROUP_SIZE = 60

MADOS_ROOT = "./MADOS"
RESOLUTION = "10"
NUM_CLASSES = 20 

BASE_LR = 2e-4
MIN_LR = 1e-6
WEIGHT_DECAY = 0.05
DROP_RATE = 0.03
DROP_PATH_RATE = 0.02

CHECKPOINT_PATH = "checkpoint_mados.pth"
BEST_PATH = "best_mados.pth"


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# =====================
# DATASET HELPERS
# =====================
def parse_split_entry(entry: str):
    """
    Split entries look like: Scene_12_34
    Meaning: scene id = 12, crop id = 34
    """
    m = re.fullmatch(r"Scene_(\d+)_(\d+)", entry.strip())
    if m:
        scene_id = int(m.group(1))
        crop_id = int(m.group(2))
        return scene_id, crop_id
    raise ValueError(f"Cannot parse split entry: {entry}")

def build_sample_weights(train_dataset):
    weights = []

    for _, cl_path in train_dataset.samples:
        mask = np.array(Image.open(cl_path), dtype=np.int64)
        valid_pixels = np.sum((mask != 0) & (mask != 255))
        w = math.sqrt(float(valid_pixels))
        weights.append(w)

    weights = torch.tensor(weights, dtype=torch.double)
    weights = weights / weights.mean()
    return weights

# =====================
# DATASET
# =====================
class MADOSSegDataset(Dataset):
    def __init__(
        self,
        root: str,
        split: str,
        resolution: str,
        label_map: dict,
        image_size: int = 128,
        train: bool = True,
        filter_empty: bool = True,
    ):
        self.root = Path(root)
        self.resolution = resolution
        self.label_map = label_map
        self.image_size = image_size
        self.train = train
        self.filter_empty = filter_empty

        split_file = self.root / "splits" / f"{split}_X.txt"
        with open(split_file, "r", encoding="utf-8") as f:
            self.entries = [line.strip() for line in f if line.strip()]

        if len(self.entries) == 0:
            raise RuntimeError(f"No entries found in {split_file}")

        self.samples = []
        for entry in self.entries:
            scene_id, crop_id = parse_split_entry(entry)
            scene_dir = self.root / f"Scene_{scene_id}" / self.resolution
            if not scene_dir.exists():
                continue

            rgb_path = scene_dir / f"Scene_{scene_id}_L2R_rgb_{crop_id}.png"
            if not rgb_path.exists():
                candidates = sorted(scene_dir.glob(f"*_rgb_{crop_id}.png"))
                rgb_path = candidates[0] if candidates else None

            cl_path = scene_dir / f"Scene_{scene_id}_L2R_cl_{crop_id}.tif"
            if not cl_path.exists():
                candidates = sorted(scene_dir.glob(f"*_cl_{crop_id}.tif"))
                cl_path = candidates[0] if candidates else None

            if rgb_path is None or cl_path is None or not rgb_path.exists() or not cl_path.exists():
                continue

            # Drop empty examples up-front so they never reach the DataLoader.
            if self.filter_empty:
                mask = np.array(Image.open(cl_path), dtype=np.int64)
                valid_pixels = np.sum((mask != 0) & (mask != 255))
                if valid_pixels == 0:
                    continue

            self.samples.append((rgb_path, cl_path))

        if len(self.samples) == 0:
            raise RuntimeError(
                f"No samples found in {split_file}. "
                f"Check that filenames follow Scene_<scene>_L2R_rgb_<crop>.png and Scene_<scene>_L2R_cl_<crop>.tif"
            )

    def __len__(self):
        return len(self.samples)

    def _load_image(self, rgb_path: Path) -> torch.Tensor:
        img = Image.open(rgb_path).convert("RGB")
        arr = np.array(img, dtype=np.float32) / 255.0
        x = torch.from_numpy(arr.transpose(2, 0, 1)).float()

        mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)
        x = (x - mean) / std
        return x

    def _load_mask(self, cl_path: Path) -> torch.Tensor:
        mask = np.array(Image.open(cl_path), dtype=np.int64)
        remapped = np.full_like(mask, 255, dtype=np.int64)
        for old_label, new_label in self.label_map.items():
            remapped[mask == old_label] = new_label
        return torch.from_numpy(remapped).long()

    def _resize_pair(self, x: torch.Tensor, y: torch.Tensor):
        x = TF.resize(x, size=(self.image_size, self.image_size), interpolation=InterpolationMode.BILINEAR)
        y = TF.resize(
            y.unsqueeze(0).float(),
            size=(self.image_size, self.image_size),
            interpolation=InterpolationMode.NEAREST,
        ).squeeze(0).long()
        return x, y

    def _augment_pair(self, x: torch.Tensor, y: torch.Tensor):
        if random.random() < 0.5:
            x = torch.flip(x, dims=[-1])
            y = torch.flip(y, dims=[-1])
        return self._resize_pair(x, y)

    def __getitem__(self, idx):
        rgb_path, cl_path = self.samples[idx]
        x = self._load_image(rgb_path)
        y = self._load_mask(cl_path)

        if self.train:
            x, y = self._augment_pair(x, y)
        else:
            x, y = self._resize_pair(x, y)

        return x, y


# =====================
# MODEL UTILS
# =====================
class DropPath(nn.Module):
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor = random_tensor.floor()
        return x.div(keep_prob) * random_tensor


def make_drop_path_rates(depth, max_drop):
    if depth <= 1:
        return [0.0]
    return torch.linspace(0.0, max_drop, depth).tolist()


def best_factor_pair(n: int):
    root = int(math.sqrt(n))
    for h in range(root, 0, -1):
        if n % h == 0:
            return h, n // h
    return 1, n


# =====================
# PATCH EMBEDDING
# =====================
class PatchEmbed(nn.Module):
    def __init__(self, img_size, patch_size, dim, in_chans):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_chans, dim // 2, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            nn.Conv2d(dim // 2, dim // 2, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            nn.Conv2d(dim // 2, dim, kernel_size=patch_size, stride=patch_size),
        )
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size * self.grid_size

    def forward(self, x):
        return self.proj(x)


# =====================
# CHUNKED PATCH MIXER
# =====================
class ChunkPatchMixer(nn.Module):
    def __init__(self, dim, chunk_size=60, heads=4, drop_path=0.0, dropout=0.0):
        super().__init__()
        self.dim = dim
        self.chunk_size = chunk_size
        self.chunk_h, self.chunk_w = best_factor_pair(chunk_size)

        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)

        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.norm4 = nn.LayerNorm(dim)

        self.mlp1 = nn.Sequential(
            nn.Linear(self.chunk_w, self.chunk_w),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.chunk_w, self.chunk_w),
            nn.Dropout(dropout),
        )

        self.mlp2 = nn.Sequential(
            nn.Linear(self.chunk_h, self.chunk_h),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.chunk_h, self.chunk_h),
            nn.Dropout(dropout),
        )

        self.channel_mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout),
        )

        self.drop_path = DropPath(drop_path)

    def _to_tokens(self, x):
        if x.ndim == 4:
            b, c, h, w = x.shape
            return x.flatten(2).transpose(1, 2).contiguous(), (h, w)
        if x.ndim == 3:
            return x, None
        raise ValueError(f"Expected 3D or 4D tensor, got shape {tuple(x.shape)}")

    def _from_tokens(self, x, spatial):
        if spatial is None:
            return x
        b, n, c = x.shape
        h, w = spatial
        return x.transpose(1, 2).contiguous().view(b, c, h, w)

    def _pad_tokens(self, x):
        b, n, c = x.shape
        pad = (self.chunk_size - (n % self.chunk_size)) % self.chunk_size
        if pad > 0:
            x = torch.cat([x, x.new_zeros(b, pad, c)], dim=1)
        return x, pad

    def forward(self, x):
        x, spatial = self._to_tokens(x)
        b, n, c = x.shape

        x, pad = self._pad_tokens(x)
        n_pad = x.shape[1]
        g = n_pad // self.chunk_size

        # [B, N, C] -> [B, G, S, C], where S is the chunk size.
        xg = x.view(b, g, self.chunk_size, c)

        # Step 1: attention inside each chunk.
        y = self.norm1(xg).view(b * g, self.chunk_size, c)
        y, _ = self.attn(y, y, y, need_weights=False)
        y = y.view(b, g, self.chunk_size, c)
        xg = xg + self.drop_path(y)

        # Step 2: first transpose-style mixing + MLP1.
        y = self.norm2(xg).view(b, g, self.chunk_h, self.chunk_w, c)
        y = y.permute(0, 1, 2, 4, 3).contiguous()
        y = y.view(b * g * self.chunk_h, c, self.chunk_w)
        y = self.mlp1(y)
        y = y.view(b, g, self.chunk_h, c, self.chunk_w).permute(0, 1, 2, 4, 3).contiguous()
        y = y.view(b, g, self.chunk_size, c)
        xg = xg + self.drop_path(y)

        # Step 3: transpose again + MLP2.
        y = self.norm3(xg).view(b, g, self.chunk_h, self.chunk_w, c)
        y = y.transpose(2, 3).contiguous()
        y = y.view(b * g * self.chunk_w, c, self.chunk_h)
        y = self.mlp2(y)
        y = y.view(b, g, self.chunk_w, c, self.chunk_h).permute(0, 1, 4, 2, 3).contiguous()
        y = y.view(b, g, self.chunk_size, c)
        xg = xg + self.drop_path(y)

        # Step 4: channel mixing.
        y = self.channel_mlp(self.norm4(xg))
        xg = xg + self.drop_path(y)

        x = xg.view(b, n_pad, c)
        if pad > 0:
            x = x[:, :n, :]

        return self._from_tokens(x, spatial)


# =====================
# PATCH MERGE
# =====================
class PatchMerge(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(4 * dim)
        self.reduction = nn.Linear(4 * dim, dim)

    def forward(self, x):
        b, c, h, w = x.shape
        pad_h = h % 2
        pad_w = w % 2
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h))
            h, w = x.shape[-2], x.shape[-1]

        x0 = x[:, :, 0::2, 0::2]
        x1 = x[:, :, 1::2, 0::2]
        x2 = x[:, :, 0::2, 1::2]
        x3 = x[:, :, 1::2, 1::2]

        x = torch.cat([x0, x1, x2, x3], dim=1)
        x = x.permute(0, 2, 3, 1).contiguous()
        x = self.reduction(self.norm(x))
        x = x.permute(0, 3, 1, 2).contiguous()
        return x


# =====================
# DECODER
# =====================
class DecoderUpBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(dim * 2, dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
            nn.GELU(),
        )

    def forward(self, x, skip):
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.block(x)


# =====================
# MODEL
# =====================
class SegPyramidMixerAttention(nn.Module):
    def __init__(self, in_chans: int, num_classes: int):
        super().__init__()
        self.patch = PatchEmbed(IMAGE_SIZE, PATCH_SIZE, DIM, in_chans=in_chans)
        dpr = make_drop_path_rates(DEPTH, DROP_PATH_RATE)

        self.stage1 = nn.ModuleList([
            ChunkPatchMixer(DIM, chunk_size=GROUP_SIZE, heads=HEADS, drop_path=dpr[0], dropout=DROP_RATE),
            ChunkPatchMixer(DIM, chunk_size=GROUP_SIZE, heads=HEADS, drop_path=dpr[1], dropout=DROP_RATE),
        ])
        self.merge1 = PatchMerge(DIM)

        self.stage2 = nn.ModuleList([
            ChunkPatchMixer(DIM, chunk_size=GROUP_SIZE, heads=HEADS, drop_path=dpr[2], dropout=DROP_RATE),
            ChunkPatchMixer(DIM, chunk_size=GROUP_SIZE, heads=HEADS, drop_path=dpr[3], dropout=DROP_RATE),
        ])
        self.merge2 = PatchMerge(DIM)

        self.stage3 = nn.ModuleList([
            ChunkPatchMixer(DIM, chunk_size=GROUP_SIZE, heads=HEADS, drop_path=dpr[4], dropout=DROP_RATE),
            ChunkPatchMixer(DIM, chunk_size=GROUP_SIZE, heads=HEADS, drop_path=dpr[5], dropout=DROP_RATE),
        ])
        self.merge3 = PatchMerge(DIM)

        self.stage4 = nn.ModuleList([
            ChunkPatchMixer(DIM, chunk_size=GROUP_SIZE, heads=HEADS, drop_path=dpr[6], dropout=DROP_RATE),
            ChunkPatchMixer(DIM, chunk_size=GROUP_SIZE, heads=HEADS, drop_path=dpr[7], dropout=DROP_RATE),
        ])

        self.up3 = DecoderUpBlock(DIM)
        self.up2 = DecoderUpBlock(DIM)
        self.up1 = DecoderUpBlock(DIM)

        self.fuse = nn.Sequential(
            nn.Conv2d(DIM, DIM, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(DIM, DIM, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.head = nn.Conv2d(DIM, num_classes, kernel_size=1)

    @staticmethod
    def _run_stage(x, blocks):
        for blk in blocks:
            x = blk(x)
        return x

    def forward(self, x):
        x = self.patch(x)

        x = self._run_stage(x, self.stage1)
        skip1 = x
        x = self.merge1(x)

        x = self._run_stage(x, self.stage2)
        skip2 = x
        x = self.merge2(x)

        x = self._run_stage(x, self.stage3)
        skip3 = x
        x = self.merge3(x)

        x = self._run_stage(x, self.stage4)

        x = self.up3(x, skip3)
        x = self.up2(x, skip2)
        x = self.up1(x, skip1)

        x = self.fuse(x)
        x = F.interpolate(x, size=(IMAGE_SIZE, IMAGE_SIZE), mode="bilinear", align_corners=False)
        x = self.head(x)
        return x


# =====================
# METRICS
# =====================
@torch.no_grad()
def compute_metrics(logits, targets, num_classes=3, ignore_index=255):
    preds = logits.argmax(dim=1)
    valid = targets != ignore_index
    preds = preds[valid]
    targets = targets[valid]

    correct = (preds == targets).sum().item()
    total = targets.numel()
    pixel_acc = correct / max(total, 1)

    ious = []
    for cls in range(num_classes):
        pred_c = preds == cls
        targ_c = targets == cls
        inter = (pred_c & targ_c).sum().item()
        union = (pred_c | targ_c).sum().item()
        if union == 0:
            continue
        ious.append(inter / union)

    miou = float(np.mean(ious)) if len(ious) > 0 else 0.0
    return pixel_acc, miou


@torch.no_grad()
def compute_per_class_iou(model, loader, num_classes=3, ignore_index=255):
    model.eval()
    intersections = torch.zeros(num_classes, dtype=torch.float64, device=DEVICE)
    unions = torch.zeros(num_classes, dtype=torch.float64, device=DEVICE)

    for x, y in loader:
        x = x.to(DEVICE, non_blocking=True)
        y = y.to(DEVICE, non_blocking=True)
        logits = model(x)
        preds = logits.argmax(dim=1)
        valid = y != ignore_index

        for cls in range(num_classes):
            pred_c = preds == cls
            targ_c = y == cls
            inter = ((pred_c & targ_c) & valid).sum().item()
            union = (((pred_c | targ_c) & valid)).sum().item()
            intersections[cls] += inter
            unions[cls] += union

    iou = torch.full((num_classes,), float("nan"), dtype=torch.float64, device=DEVICE)
    nonzero = unions > 0
    iou[nonzero] = intersections[nonzero] / unions[nonzero]
    return iou.detach().cpu().numpy()


# =====================
# EMA
# =====================
class EMA:
    def __init__(self, model, decay=0.9999):
        self.decay = decay
        self.shadow = {}
        self.backup = {}

        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.detach().clone()

        for name, buffer in model.named_buffers():
            if buffer.dtype.is_floating_point:
                self.shadow[name] = buffer.detach().clone()

    @torch.no_grad()
    def update(self, model):
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if name not in self.shadow:
                self.shadow[name] = param.detach().clone()
            else:
                self.shadow[name].mul_(self.decay).add_(param.detach(), alpha=1.0 - self.decay)

        for name, buffer in model.named_buffers():
            if buffer.dtype.is_floating_point:
                if name not in self.shadow:
                    self.shadow[name] = buffer.detach().clone()
                else:
                    self.shadow[name].mul_(self.decay).add_(buffer.detach(), alpha=1.0 - self.decay)

    @torch.no_grad()
    def apply(self, model):
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.detach().clone()
                param.copy_(self.shadow[name])
        for name, buffer in model.named_buffers():
            if buffer.dtype.is_floating_point and name in self.shadow:
                self.backup[name] = buffer.detach().clone()
                buffer.copy_(self.shadow[name])

    @torch.no_grad()
    def restore(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.copy_(self.backup[name])
        for name, buffer in model.named_buffers():
            if buffer.dtype.is_floating_point and name in self.backup:
                buffer.copy_(self.backup[name])


# =====================
# EVALUATION
# =====================
@torch.inference_mode()
def evaluate_model(model, loader, criterion, num_classes=NUM_CLASSES, ignore_index=255):
    model.eval()
    total_loss = 0.0
    total_pix_acc = 0.0
    total_miou = 0.0
    batches = 0

    intersections = torch.zeros(num_classes, dtype=torch.float64, device=DEVICE)
    unions = torch.zeros(num_classes, dtype=torch.float64, device=DEVICE)

    for x, y in loader:
        x = x.to(DEVICE, non_blocking=True)
        y = y.to(DEVICE, non_blocking=True)

        logits = model(x)
        loss = criterion(logits, y)
        pix_acc, miou = compute_metrics(logits, y, num_classes=num_classes, ignore_index=ignore_index)

        preds = logits.argmax(dim=1)
        valid = y != ignore_index
        for cls in range(num_classes):
            pred_c = preds == cls
            targ_c = y == cls
            inter = ((pred_c & targ_c) & valid).sum().item()
            union = (((pred_c | targ_c) & valid)).sum().item()
            intersections[cls] += inter
            unions[cls] += union

        total_loss += loss.item()
        total_pix_acc += pix_acc
        total_miou += miou
        batches += 1

    per_class_iou = torch.full((num_classes,), float("nan"), dtype=torch.float64, device=DEVICE)
    nonzero = unions > 0
    per_class_iou[nonzero] = intersections[nonzero] / unions[nonzero]

    return (
        total_loss / max(batches, 1),
        total_pix_acc / max(batches, 1),
        total_miou / max(batches, 1),
        per_class_iou.detach().cpu().numpy(),
    )


# =====================
# TRAINING
# =====================
def cosine_with_warmup_factor(epoch: int):
    min_factor = MIN_LR / BASE_LR
    if epoch < WARMUP_EPOCHS:
        return max(min_factor, float(epoch + 1) / float(WARMUP_EPOCHS))
    progress = (epoch - WARMUP_EPOCHS) / max(1, EPOCHS - WARMUP_EPOCHS)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_factor + (1.0 - min_factor) * cosine


def train(model, train_loader, val_loader, criterion):
    model = model.to(DEVICE)
    ema = EMA(model, decay=0.9999)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=BASE_LR,
        betas=(0.9, 0.95),
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=cosine_with_warmup_factor)

    start_epoch = 0
    global_step = 0
    history = {
        "train_loss": [],
        "val_loss": [],
        "train_pixel_acc": [],
        "val_pixel_acc": [],
        "train_miou": [],
        "val_miou": [],
        "train_per_class_iou": [],
        "val_per_class_iou": [],
        "lr": [],
    }
    best_miou = 0.0
    epochs_no_improve = 0

    if os.path.exists(CHECKPOINT_PATH):
        print(f"Loading checkpoint: {CHECKPOINT_PATH}")
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        if "ema" in checkpoint:
            ema.shadow = {k: v.to(DEVICE) for k, v in checkpoint["ema"].items()}
        start_epoch = checkpoint["epoch"] + 1
        global_step = checkpoint.get("global_step", 0)
        history = checkpoint.get("history", history)
        best_miou = checkpoint.get("best_miou", 0.0)
        print(f"Resumed from epoch {start_epoch}")



    for epoch in range(start_epoch, EPOCHS):
        model.train()
        total_loss = 0.0
        valid_steps = 0

        for i, (x, y) in enumerate(train_loader):
            x = x.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            valid_pixels = (y != 255).sum()
            if valid_pixels.item() == 0:
                print(f"Empty target at epoch {epoch}, batch {i}, skipping")
                continue

            logits = model(x).float()
            loss = criterion(logits, y)

            if not torch.isfinite(loss):
                print(f"Non-finite loss at epoch {epoch}, batch {i}, skipping")
                optimizer.zero_grad(set_to_none=True)
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            ema.update(model)

            total_loss += loss.item()
            valid_steps += 1
            global_step += 1

        scheduler.step()
        avg_loss = total_loss / max(valid_steps, 1)
        current_lr = optimizer.param_groups[0]["lr"]

        ema.apply(model)
        train_loss_eval, train_pix_acc, train_miou, train_pc_iou = evaluate_model(model, train_loader, criterion)
        val_loss, val_pix_acc, val_miou, val_pc_iou = evaluate_model(model, val_loader, criterion)
        ema.restore(model)

        history["train_loss"].append(train_loss_eval)
        history["val_loss"].append(val_loss)
        history["train_pixel_acc"].append(train_pix_acc)
        history["val_pixel_acc"].append(val_pix_acc)
        history["train_miou"].append(train_miou)
        history["val_miou"].append(val_miou)
        history["train_per_class_iou"].append(train_pc_iou)
        history["val_per_class_iou"].append(val_pc_iou)
        history["lr"].append(current_lr)

        print(
            f"Epoch {epoch}: train_loss={train_loss_eval:.4f}, val_loss={val_loss:.4f}, "
            f"train_pixel_acc={train_pix_acc:.4f}, val_pixel_acc={val_pix_acc:.4f}, "
            f"train_miou={train_miou:.4f}, val_miou={val_miou:.4f}, lr={current_lr:.6e}"
        )

        if val_miou > best_miou:
            best_miou = val_miou
            epochs_no_improve = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "ema": {k: v.detach().cpu() for k, v in ema.shadow.items()},
                    "history": history,
                    "best_miou": best_miou,
                    "global_step": global_step,
                },
                BEST_PATH,
            )
        else:
            epochs_no_improve += 1

        if epoch % 5 == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "ema": {k: v.detach().cpu() for k, v in ema.shadow.items()},
                    "history": history,
                    "best_miou": best_miou,
                    "global_step": global_step,
                },
                CHECKPOINT_PATH,
            )

        if epochs_no_improve >= EARLY_STOP_PATIENCE:
            print(f"Early stopping at epoch {epoch} (no improvement for {EARLY_STOP_PATIENCE} epochs)")
            break

    return history


# =====================
# MAIN
# =====================
if __name__ == "__main__":
    freeze_support()
    seed_everything(SEED)

    print("Preparing MADOS dataset...")

    label_map = {i: i - 1 for i in range(1, NUM_CLASSES + 1)}

    train_dataset = MADOSSegDataset(
        root=MADOS_ROOT,
        split="train",
        resolution=RESOLUTION,
        label_map=label_map,
        image_size=IMAGE_SIZE,
        train=True,
    )
    val_dataset = MADOSSegDataset(
        root=MADOS_ROOT,
        split="val",
        resolution=RESOLUTION,
        label_map=label_map,
        image_size=IMAGE_SIZE,
        train=False,
    )
    test_dataset = MADOSSegDataset(
        root=MADOS_ROOT,
        split="test",
        resolution=RESOLUTION,
        label_map=label_map,
        image_size=IMAGE_SIZE,
        train=False,
    )

    sample_x, sample_y = train_dataset[0]
    in_chans = sample_x.shape[0]

    print(f"Device: {DEVICE}")
    print(f"Input channels: {in_chans}")
    print(f"Number of classes: {NUM_CLASSES}")
    print(f"Train samples: {len(train_dataset)}  Val samples: {len(val_dataset)}  Test samples: {len(test_dataset)}")
    print(f"Sample tensor shape: {tuple(sample_x.shape)}, mask shape: {tuple(sample_y.shape)}")
    print(f"y unique (sample): {torch.unique(sample_y)}")

    sample_weights = build_sample_weights(train_dataset)
    train_sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights) * 2,
        replacement=True,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=train_sampler,
        shuffle=False,
        drop_last=True,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE == "cuda"),
        persistent_workers=(NUM_WORKERS > 0),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE == "cuda"),
        persistent_workers=(NUM_WORKERS > 0),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE == "cuda"),
        persistent_workers=(NUM_WORKERS > 0),
    )

    print("Training SegPyramidMixer...")
    model = SegPyramidMixerAttention(in_chans=in_chans, num_classes=NUM_CLASSES)
    criterion = nn.CrossEntropyLoss(ignore_index=255)
    history = train(model, train_loader, val_loader, criterion)

    if os.path.exists(BEST_PATH):
        best_ckpt = torch.load(BEST_PATH, map_location=DEVICE, weights_only=False)
        model.load_state_dict(best_ckpt["model"])

    best_train_loss, best_train_pix_acc, best_train_miou, best_train_pc_iou = evaluate_model(model, train_loader, criterion)
    best_val_loss, best_val_pix_acc, best_val_miou, best_val_pc_iou = evaluate_model(model, val_loader, criterion)

    class_ids = np.arange(1, NUM_CLASSES + 1)
    width = 0.35
    x = np.arange(NUM_CLASSES)

    plt.figure(figsize=(14, 5))
    plt.bar(x - width / 2, best_train_pc_iou, width=width, label="Train")
    plt.bar(x + width / 2, best_val_pc_iou, width=width, label="Val")
    plt.xticks(x, class_ids)
    plt.title("Per-class IoU (best checkpoint)")
    plt.xlabel("Class")
    plt.ylabel("IoU")
    plt.ylim(0, 1.0)
    plt.legend()
    plt.tight_layout()
    plt.show()

    epochs = np.arange(1, len(history["train_loss"]) + 1)

    plt.figure()
    plt.plot(epochs, history["train_miou"], label="Train mIoU")
    plt.plot(epochs, history["val_miou"], label="Val mIoU")
    plt.legend()
    plt.title("mIoU")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.show()

    plt.figure()
    plt.plot(epochs, history["train_pixel_acc"], label="Train Pixel Acc")
    plt.plot(epochs, history["val_pixel_acc"], label="Val Pixel Acc")
    plt.legend()
    plt.title("Pixel Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.show()

    plt.figure()
    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.plot(epochs, history["val_loss"], label="Val Loss")
    plt.legend()
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.show()
