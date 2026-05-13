import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from multiprocessing import freeze_support
import warnings
warnings.filterwarnings("ignore")

SEED = 42

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

EPOCHS = 301
BATCH_SIZE = 128

BASE_LR = 1e-3
MIN_LR = 1e-6
WARMUP_EPOCHS = 5

IMAGE_SIZE = 32
PATCH_SIZE = 4
DIM = 128
DEPTH = 8
HEADS = 4

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

# =====================
# CUTMIX
# =====================
def rand_bbox(size, lam):
    W = size[2]
    H = size[3]

    cut_rat = np.sqrt(1. - lam)
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
    index = torch.randperm(batch_size).to(x.device)

    shuffled_x = x[index]
    shuffled_y = y[index]

    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    x[:, :, bbx1:bbx2, bby1:bby2] = shuffled_x[:, :, bbx1:bbx2, bby1:bby2]

    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size(-1) * x.size(-2)))
    return x, y, shuffled_y, lam

# =====================
# MODEL
# =====================
class PatchEmbed(nn.Module):
    def __init__(self, img_size, patch_size, dim):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class MixerBlock(nn.Module):
    def __init__(self, num_patches, dim):
        super().__init__()

        self.norm1 = nn.LayerNorm(dim)
        self.token_mlp = nn.Sequential(
            nn.Linear(num_patches, num_patches),
            nn.GELU(),
            nn.Linear(num_patches, num_patches)
        )

        self.norm2 = nn.LayerNorm(dim)
        self.channel_mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )

    def forward(self, x):
        y = self.norm1(x)
        y = y.transpose(1, 2)
        y = self.token_mlp(y)
        y = y.transpose(1, 2)
        x = x + y

        y = self.norm2(x)
        y = self.channel_mlp(y)
        x = x + y

        return x


class AttentionBlock(nn.Module):
    def __init__(self, dim, heads=4):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.gamma = nn.Parameter(torch.ones(1) * 0.5)

    def forward(self, x):
        y = self.norm(x)
        y, _ = self.attn(y, y, y)
        x = x + self.gamma * y
        return x


class MLPMixer(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch = PatchEmbed(IMAGE_SIZE, PATCH_SIZE, DIM)
        num_patches = self.patch.num_patches

        self.blocks = nn.Sequential(*[
            MixerBlock(num_patches, DIM) for _ in range(DEPTH)
        ])

        self.norm = nn.LayerNorm(DIM)
        self.head = nn.Linear(DIM, 10)

    def forward(self, x):
        x = self.patch(x)
        x = self.blocks(x)
        x = self.norm(x)
        x = x.mean(dim=1)
        return self.head(x)


class HybridModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch = PatchEmbed(IMAGE_SIZE, PATCH_SIZE, DIM)
        num_patches = self.patch.num_patches

        self.attn = AttentionBlock(DIM, HEADS)

        self.blocks = nn.Sequential(*[
            MixerBlock(num_patches, DIM) for _ in range(DEPTH)
        ])

        self.norm = nn.LayerNorm(DIM)
        self.head = nn.Linear(DIM, 10)

    def forward(self, x):
        x = self.patch(x)
        x = self.attn(x)
        x = self.blocks(x)
        x = self.norm(x)
        x = x.mean(dim=1)
        return self.head(x)


def train(model, checkpoint_path, log_path, seed):
    g = torch.Generator()
    g.manual_seed(seed)

    # DATA
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010))
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010))
    ])

    train_dataset = datasets.CIFAR10("./data", train=True, download=True, transform=transform_train)
    test_dataset = datasets.CIFAR10("./data", train=False, download=True, transform=transform_test)

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=2, worker_init_fn=seed_worker, generator=g
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE,
        num_workers=2, worker_init_fn=seed_worker, generator=g
    )

    model = model.to(DEVICE)

    optimizer = optim.Adam(
        model.parameters(),
        lr=BASE_LR,
        betas=(0.9, 0.99),
        weight_decay=5e-5
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS,
        eta_min=MIN_LR
    )

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    scaler = torch.cuda.amp.GradScaler()


    start_epoch = 0
    train_losses = []
    val_losses = []
    top1_accs = []
    top2_accs = []

    if start_epoch == 0:
        log_file_mode = 'w'  
    else:
        log_file_mode = 'a'  

    log_file = open(log_path, log_file_mode) if log_path else None

    if os.path.exists(checkpoint_path):
        print(f"🔄 Loading checkpoint: {checkpoint_path}")
        if log_file:
            log_file.write(f"🔄 Loading checkpoint: {checkpoint_path}\n")
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE)

        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])

        start_epoch = checkpoint["epoch"] + 1

        top1_accs = checkpoint.get("top1_accs", checkpoint.get("test_accs", []))
        train_losses = checkpoint.get("train_losses", [])
        val_losses = checkpoint.get("val_losses", [])
        top2_accs = checkpoint.get("top2_accs", [])

        print(f"✅ Resumed from epoch {start_epoch}")
        if log_file:
            log_file.write(f"✅ Resumed from epoch {start_epoch}\n")


    def evaluate():
        model.eval()
        total_loss = 0
        correct1, correct2, total = 0, 0, 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                out = model(x)
                loss = criterion(out, y)
                total_loss += loss.item() * x.size(0)

                preds = out.argmax(dim=1)
                correct1 += (preds == y).sum().item()

                _, top2 = out.topk(2, dim=1)
                correct2 += (top2 == y.unsqueeze(1)).any(dim=1).sum().item()
                total += y.size(0)
        return total_loss / total, correct1 / total, correct2 / total


    for epoch in range(start_epoch, EPOCHS):
        model.train()
        total_loss = 0

        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()

            if epoch < WARMUP_EPOCHS:
                lr = BASE_LR * (epoch + 1) / WARMUP_EPOCHS
                for param_group in optimizer.param_groups:
                    param_group["lr"] = lr

            with torch.cuda.amp.autocast():
                if np.random.rand() < 0.5:
                    x, y_a, y_b, lam = cutmix_data(x, y)
                    out = model(x)
                    loss = lam * criterion(out, y_a) + (1 - lam) * criterion(out, y_b)
                else:
                    out = model(x)
                    loss = criterion(out, y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

        if epoch >= WARMUP_EPOCHS:
            scheduler.step()

        train_loss = total_loss / len(train_loader)
        train_losses.append(train_loss)

        val_loss, acc1, acc2 = evaluate()
        val_losses.append(val_loss)
        top1_accs.append(acc1)
        top2_accs.append(acc2)

        log_line = f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, acc1={acc1:.4f}, acc2={acc2:.4f}"
        print(log_line)
        if log_file:
            log_file.write(log_line + "\n")
            log_file.flush() 

        if epoch % 5 == 0:
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "train_losses": train_losses,
                "val_losses": val_losses,
                "top1_accs": top1_accs,
                "top2_accs": top2_accs
            }, checkpoint_path)

    if log_file:
        log_file.close()

    return {
        "train_loss": train_losses,
        "val_loss": val_losses,
        "top1": top1_accs,
        "top2": top2_accs
    }


if __name__ == "__main__":
    freeze_support()

    print("Training baseline Mixer...")
    set_seed(SEED)
    mixer = MLPMixer()
    metrics_mixer = train(mixer, "checkpoint91_mixer.pth", "log_mixer.txt", seed=SEED)

    print("\nTraining Mixer + Attention...")
    set_seed(SEED)
    hybrid = HybridModel()
    metrics_hybrid = train(hybrid, "checkpoint91_hybrid.pth", "log_hybrid.txt", seed=SEED)

    fig1, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(metrics_mixer["train_loss"], label="Mixer")
    ax1.plot(metrics_hybrid["train_loss"], label="Mixer+Attention")
    ax1.set_title("Train Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True)

    fig2, ax2 = plt.subplots(figsize=(10, 6))
    ax2.plot(metrics_mixer["val_loss"], label="Mixer")
    ax2.plot(metrics_hybrid["val_loss"], label="Mixer+Attention")
    ax2.set_title("Validation Loss")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.legend()
    ax2.grid(True)

    fig3, ax3 = plt.subplots(figsize=(10, 6))
    ax3.plot(metrics_mixer["top1"], label="Mixer")
    ax3.plot(metrics_hybrid["top1"], label="Mixer+Attention")
    ax3.set_title("Top-1 Accuracy")
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("Accuracy")
    ax3.legend()
    ax3.grid(True)

    fig4, ax4 = plt.subplots(figsize=(10, 6))
    ax4.plot(metrics_mixer["top2"], label="Mixer")
    ax4.plot(metrics_hybrid["top2"], label="Mixer+Attention")
    ax4.set_title("Top-2 Accuracy")
    ax4.set_xlabel("Epoch")
    ax4.set_ylabel("Accuracy")
    ax4.legend()
    ax4.grid(True)

    plt.show()