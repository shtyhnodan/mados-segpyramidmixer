import os
import math
import random
import warnings
from multiprocessing import freeze_support

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

EPOCHS = 351
BATCH_SIZE = 128

BASE_LR = 8e-4
MIN_LR = 1e-6

IMAGE_SIZE = 32
PATCH_SIZE = 4
DIM = 224
DEPTH = 8
HEADS = 4

WEIGHT_DECAY = 0.05
LABEL_SMOOTHING = 0.0

DROP_RATE = 0.03
DROP_PATH_RATE = 0.03

CUTMIX_ALPHA = 1.0

NUM_CLASSES = 10

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False     
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False


def rand_bbox(size, lam):
    H = size[2]
    W = size[3]

    cut_rat = math.sqrt(1.0 - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def cutmix_data(x, y, alpha=1.0):
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)

    shuffled_x = x[index]
    shuffled_y = y[index]

    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)

    x[:, :, bby1:bby2, bbx1:bbx2] = shuffled_x[:, :, bby1:bby2, bbx1:bbx2]

    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size(-1) * x.size(-2)))
    return x, y, shuffled_y, lam


def smooth_one_hot(labels, num_classes=10, smoothing=0.0):
    with torch.no_grad():
        confidence = 1.0 - smoothing
        off_value = smoothing / (num_classes - 1) if num_classes > 1 else 0.0
        y = torch.full((labels.size(0), num_classes), off_value, device=labels.device)
        y.scatter_(1, labels.unsqueeze(1), confidence)
    return y


def soft_cross_entropy(logits, soft_targets):
    log_probs = F.log_softmax(logits, dim=1)
    return -(soft_targets * log_probs).sum(dim=1).mean()


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


class PatchEmbed(nn.Module):
    def __init__(self, img_size, patch_size, dim):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(3, dim // 2, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            nn.Conv2d(dim // 2, dim // 2, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            nn.Conv2d(dim // 2, dim, kernel_size=patch_size, stride=patch_size),
        )
        self.num_patches = (img_size // patch_size) ** 2

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class MixerBlock(nn.Module):
    def __init__(self, num_patches, dim, drop_path=0.0, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.token_mlp = nn.Sequential(
            nn.Linear(num_patches, num_patches),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(num_patches, num_patches),
            nn.Dropout(dropout),
        )

        self.norm2 = nn.LayerNorm(dim)
        self.channel_mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout),
        )

        self.drop_path = DropPath(drop_path)

    def forward(self, x):
        y = self.norm1(x)
        y = y.transpose(1, 2)
        y = self.token_mlp(y)
        y = y.transpose(1, 2)
        x = x + self.drop_path(y)

        y = self.norm2(x)
        y = self.channel_mlp(y)
        x = x + self.drop_path(y)

        return x


class AttentionBlock(nn.Module):
    def __init__(self, dim, heads=4, drop_path=0.0, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)

        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout),
        )

        self.drop_path = DropPath(drop_path)

    def forward(self, x):
        y = self.norm1(x)
        y, _ = self.attn(y, y, y, need_weights=False)
        x = x + self.drop_path(y)

        y = self.norm2(x)
        y = self.ffn(y)
        x = x + self.drop_path(y)

        return x


class MLPMixer(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch = PatchEmbed(IMAGE_SIZE, PATCH_SIZE, DIM)
        num_patches = self.patch.num_patches

        dpr = make_drop_path_rates(DEPTH, DROP_PATH_RATE)
        self.blocks = nn.ModuleList([
            MixerBlock(num_patches, DIM, drop_path=dpr[i], dropout=DROP_RATE)
            for i in range(DEPTH)
        ])

        self.norm = nn.LayerNorm(DIM)
        self.head = nn.Linear(DIM, NUM_CLASSES)

    def forward(self, x):
        x = self.patch(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        x = x.mean(dim=1)
        return self.head(x)


class HybridModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.patch = PatchEmbed(IMAGE_SIZE, PATCH_SIZE, DIM)
        num_patches = self.patch.num_patches

        dpr = make_drop_path_rates(DEPTH, DROP_PATH_RATE)

        self.attn1 = AttentionBlock(DIM, HEADS, drop_path=0.0, dropout=DROP_RATE)
        self.blocks1 = nn.ModuleList([
            MixerBlock(num_patches, DIM, drop_path=dpr[i], dropout=DROP_RATE)
            for i in range(2)
        ])

        self.attn2 = AttentionBlock(DIM, HEADS, drop_path=0.0, dropout=DROP_RATE)
        self.blocks2 = nn.ModuleList([
            MixerBlock(num_patches, DIM, drop_path=dpr[2 + i], dropout=DROP_RATE)
            for i in range(2)
        ])

        self.attn3 = AttentionBlock(DIM, HEADS, drop_path=0.0, dropout=DROP_RATE)
        self.blocks3 = nn.ModuleList([
            MixerBlock(num_patches, DIM, drop_path=dpr[4 + i], dropout=DROP_RATE)
            for i in range(4)
        ])

        self.norm = nn.LayerNorm(DIM)
        self.head = nn.Linear(DIM, NUM_CLASSES)

    def forward(self, x):
        x = self.patch(x)

        x = self.attn1(x)
        for blk in self.blocks1:
            x = blk(x)

        x = self.attn2(x)
        for blk in self.blocks2:
            x = blk(x)

        x = self.attn3(x)
        for blk in self.blocks3:
            x = blk(x)

        x = self.norm(x)
        x = x.mean(dim=1)
        return self.head(x)


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


def train(model, run_name, seed=42):
    checkpoint_path = f"checkpoint_{run_name}.pth"
    best_path = f"best_{run_name}.pth"

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010))
    ])

    train_dataset = datasets.CIFAR10("./data", train=True, download=True, transform=transform_train)
    test_dataset = datasets.CIFAR10("./data", train=False, download=True, transform=transform_test)

    g = torch.Generator()
    g.manual_seed(seed)

    def seed_worker(worker_id):
        worker_seed = seed + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=(DEVICE == "cuda"),
        persistent_workers=True,
        generator=g,
        worker_init_fn=seed_worker,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=(DEVICE == "cuda"),
        persistent_workers=True,
        generator=g,
        worker_init_fn=seed_worker,
    )

    model = model.to(DEVICE)
    ema = EMA(model, decay=0.9999)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=BASE_LR,
        betas=(0.9, 0.95),
        weight_decay=WEIGHT_DECAY
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=100,
        T_mult=2,
        eta_min=MIN_LR
    )

    scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE == "cuda"))

    start_epoch = 0
    global_step = 0
    val_accs = []
    top2_accs = []
    val_losses = []
    train_losses = []
    lrs = []
    best_acc = 0.0

    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE)

        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])

        if "ema" in checkpoint:
            ema.shadow = {k: v.to(DEVICE) for k, v in checkpoint["ema"].items()}

        start_epoch = checkpoint["epoch"] + 1
        global_step = checkpoint.get("global_step", 0)
        val_accs = checkpoint.get("val_accs", [])
        top2_accs = checkpoint.get("top2_accs", [])
        val_losses = checkpoint.get("val_losses", [])
        train_losses = checkpoint.get("train_losses", [])
        lrs = checkpoint.get("lrs", [])
        best_acc = checkpoint.get("best_acc", 0.0)

        print(f"Resumed from epoch {start_epoch}")

    val_criterion = nn.CrossEntropyLoss()

    @torch.inference_mode()
    def evaluate():
        model.eval()
        total_loss = 0.0
        correct1, correct2, total = 0, 0, 0

        for x, y in test_loader:
            x = x.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True)

            logits = model(x)
            batch_size = y.size(0)

            total_loss += val_criterion(logits, y).item() * batch_size

            preds1 = logits.argmax(dim=1)
            correct1 += (preds1 == y).sum().item()

            top2 = logits.topk(2, dim=1).indices
            correct2 += top2.eq(y.unsqueeze(1)).any(dim=1).sum().item()

            total += batch_size

        val_loss = total_loss / total
        top1_acc = correct1 / total
        top2_acc = correct2 / total
        return val_loss, top1_acc, top2_acc

    for epoch in range(start_epoch, EPOCHS):
        model.train()
        total_loss = 0.0

        for i, (x, y) in enumerate(train_loader):
            x = x.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=(DEVICE == "cuda")):
                if np.random.rand() < 0.5:
                    x_cut, y_a, y_b, lam = cutmix_data(x, y, alpha=CUTMIX_ALPHA)
                    out = model(x_cut)

                    target_a = smooth_one_hot(y_a, NUM_CLASSES, LABEL_SMOOTHING)
                    target_b = smooth_one_hot(y_b, NUM_CLASSES, LABEL_SMOOTHING)
                    soft_targets = lam * target_a + (1.0 - lam) * target_b
                    loss = soft_cross_entropy(out, soft_targets)
                else:
                    out = model(x)
                    target = smooth_one_hot(y, NUM_CLASSES, LABEL_SMOOTHING)
                    loss = soft_cross_entropy(out, target)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step(epoch + i / len(train_loader))

            ema.update(model)
            total_loss += loss.item()
            global_step += 1

        avg_loss = total_loss / len(train_loader)
        current_lr = optimizer.param_groups[0]["lr"]

        train_losses.append(avg_loss)
        lrs.append(current_lr)

        ema.apply(model)
        val_loss, acc, top2_acc = evaluate()
        ema.restore(model)

        val_accs.append(acc)
        top2_accs.append(top2_acc)
        val_losses.append(val_loss)

        print(
            f"Epoch {epoch}: train_loss={avg_loss:.4f}, val_loss={val_loss:.4f}, "
            f"acc={acc:.4f}, top2={top2_acc:.4f}, lr={current_lr:.6e}"
        )

        if acc > best_acc:
            best_acc = acc
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "ema": {k: v.detach().cpu() for k, v in ema.shadow.items()},
                "val_accs": val_accs,
                "top2_accs": top2_accs,
                "val_losses": val_losses,
                "train_losses": train_losses,
                "lrs": lrs,
                "best_acc": best_acc,
                "global_step": global_step
            }, best_path)

        if epoch % 5 == 0:
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "ema": {k: v.detach().cpu() for k, v in ema.shadow.items()},
                "val_accs": val_accs,
                "top2_accs": top2_accs,
                "val_losses": val_losses,
                "train_losses": train_losses,
                "lrs": lrs,
                "best_acc": best_acc,
                "global_step": global_step
            }, checkpoint_path)

    return val_accs, top2_accs, train_losses, val_losses, lrs


if __name__ == "__main__":
    freeze_support()

    SEED = 42

    print("\nTraining standard Mixer...")
    set_seed(SEED)
    mixer = MLPMixer()
    acc_mixer, top2_mixer, loss_mixer, valloss_mixer, lr_mixer = train(mixer, "mixer_c10", seed=SEED)

    print("\nTraining Mixer + Attention...")
    set_seed(SEED) 
    hybrid = HybridModel()
    acc_hybrid, top2_hybrid, loss_hybrid, valloss_hybrid, lr_hybrid = train(hybrid, "hybrid_c10", seed=SEED)

    plt.figure()
    plt.plot(acc_mixer, label="Mixer")
    plt.plot(acc_hybrid, label="Mixer+Attention")
    plt.legend()
    plt.title("Top-1 Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.show()

    plt.figure()
    plt.plot(top2_mixer, label="Mixer")
    plt.plot(top2_hybrid, label="Mixer+Attention")
    plt.legend()
    plt.title("Top-2 Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Top-2 Accuracy")
    plt.show()

    plt.figure()
    plt.plot(loss_mixer, label="Mixer")
    plt.plot(loss_hybrid, label="Mixer+Attention")
    plt.legend()
    plt.title("Train Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.show()

    plt.figure()
    plt.plot(valloss_mixer, label="Mixer")
    plt.plot(valloss_hybrid, label="Mixer+Attention")
    plt.legend()
    plt.title("Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.show()