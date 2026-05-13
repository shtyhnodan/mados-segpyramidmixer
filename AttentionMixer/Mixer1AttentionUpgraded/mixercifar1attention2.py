import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from multiprocessing import freeze_support
import warnings
warnings.filterwarnings("ignore")

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

EPOCHS = 301
BATCH_SIZE = 128

BASE_LR = 1e-3
MIN_LR = 1e-6
WARMUP_EPOCHS = 5

IMAGE_SIZE = 32
PATCH_SIZE = 4
DIM = 160
DEPTH = 8
HEADS = 4

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

        self.proj = nn.Sequential(
            nn.Conv2d(3, dim // 2, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            nn.Conv2d(dim // 2, dim, kernel_size=patch_size, stride=patch_size)
        )

        self.num_patches = (img_size // patch_size) ** 2

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class MixerBlock(nn.Module):
    def __init__(self, num_patches, dim, drop_prob=0.0):
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
    def __init__(self, dim, heads=4, drop_prob=0.0):
        super().__init__()

        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=0.0, batch_first=True)

        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )

    def forward(self, x):
        y = self.norm1(x)
        y, _ = self.attn(y, y, y)
        x = x + y

        y = self.norm2(x)
        y = self.ffn(y)
        x = x + y

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


class EMA:
    def __init__(self, model, decay=0.9999):
        self.model = model
        self.decay = decay
        self.shadow = {
            k: v.clone().detach()
            for k, v in model.state_dict().items()
        }

    def update(self):
        for k, v in self.model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k] = self.decay * self.shadow[k] + (1 - self.decay) * v.detach()

    def apply(self):
        self.backup = {}
        for k, v in self.model.state_dict().items():
            if v.dtype.is_floating_point:
                self.backup[k] = v.clone()
                v.copy_(self.shadow[k])

    def restore(self):
        for k, v in self.model.state_dict().items():
            if v.dtype.is_floating_point:
                v.copy_(self.backup[k])


def train(model, checkpoint_path):
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

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, num_workers=2)

    model = model.to(DEVICE)
    ema = EMA(model, decay=0.9999)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=BASE_LR,
        betas=(0.9, 0.999),
        weight_decay=0.05
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS - WARMUP_EPOCHS,
        eta_min=MIN_LR
    )

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    scaler = torch.cuda.amp.GradScaler()

    start_epoch = 0
    train_loss_hist = []
    val_loss_hist = []
    val_top1_hist = []
    val_top2_hist = []

    log_path = checkpoint_path.replace('.pth', '.log')

    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE)

        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])

        if "ema" in checkpoint:
            ema.shadow = checkpoint["ema"]
            ema.apply()
            ema.restore()

        start_epoch = checkpoint["epoch"] + 1

        if "train_loss_hist" in checkpoint:
            train_loss_hist = checkpoint["train_loss_hist"]
            val_loss_hist = checkpoint["val_loss_hist"]
            val_top1_hist = checkpoint["val_top1_hist"]
            val_top2_hist = checkpoint["val_top2_hist"]
        else:
            val_top1_hist = checkpoint.get("test_accs", [])
            train_loss_hist = []
            val_loss_hist = []
            val_top2_hist = []

        print(f"Resumed from epoch {start_epoch}")
        log_file = open(log_path, 'a')  
    else:
        log_file = open(log_path, 'w')
        log_file.write("epoch,train_loss,val_loss,top1_acc,top2_acc\n")


    def evaluate():
        model.eval()
        total_loss = 0.0
        correct_top1 = 0
        correct_top2 = 0
        total = 0
        criterion_eval = nn.CrossEntropyLoss() 
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                out = model(x)
                loss = criterion_eval(out, y)
                total_loss += loss.item() * x.size(0)

                # top-1
                pred = out.argmax(dim=1)
                correct_top1 += (pred == y).sum().item()

                # top-2
                top2_pred = out.topk(2, dim=1).indices
                correct_top2 += (top2_pred == y.unsqueeze(1)).any(dim=1).sum().item()

                total += y.size(0)
        return total_loss / total, correct_top1 / total, correct_top2 / total


    for epoch in range(start_epoch, EPOCHS):
        model.train()
        total_loss = 0

        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()

            # warmup
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
            ema.update()

            total_loss += loss.item()

        if epoch >= WARMUP_EPOCHS:
            scheduler.step()

        train_loss_hist.append(total_loss / len(train_loader))

        ema.apply()
        eval_loss, eval_top1, eval_top2 = evaluate()
        ema.restore()

        val_loss_hist.append(eval_loss)
        val_top1_hist.append(eval_top1)
        val_top2_hist.append(eval_top2)

        print(f"Epoch {epoch}: train_loss={train_loss_hist[-1]:.4f}, val_loss={eval_loss:.4f}, top1_acc={eval_top1:.4f}, top2_acc={eval_top2:.4f}")

        log_line = f"{epoch},{train_loss_hist[-1]:.6f},{eval_loss:.6f},{eval_top1:.6f},{eval_top2:.6f}\n"
        log_file.write(log_line)
        log_file.flush()

        if epoch % 5 == 0:
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "ema": ema.shadow,
                "train_loss_hist": train_loss_hist,
                "val_loss_hist": val_loss_hist,
                "val_top1_hist": val_top1_hist,
                "val_top2_hist": val_top2_hist,
            }, checkpoint_path)

    log_file.close()
    return train_loss_hist, val_loss_hist, val_top1_hist, val_top2_hist


if __name__ == "__main__":
    freeze_support()

    print("Training baseline Mixer...")
    mixer = MLPMixer()
    train_loss_m, val_loss_m, top1_m, top2_m = train(mixer, "checkpoint_mixer.pth")

    print("\nTraining Mixer + Attention...")
    hybrid = HybridModel()
    train_loss_h, val_loss_h, top1_h, top2_h = train(hybrid, "checkpoint_hybrid.pth")

    plt.figure("Train Loss")
    plt.plot(train_loss_m, label="Mixer")
    plt.plot(train_loss_h, label="Mixer+Attention")
    plt.title("Train Loss")
    plt.legend()

    plt.figure("Validation Loss")
    plt.plot(val_loss_m, label="Mixer")
    plt.plot(val_loss_h, label="Mixer+Attention")
    plt.title("Validation Loss")
    plt.legend()

    plt.figure("Top-1 Accuracy")
    plt.plot(top1_m, label="Mixer")
    plt.plot(top1_h, label="Mixer+Attention")
    plt.title("Top-1 Accuracy")
    plt.legend()

    plt.figure("Top-2 Accuracy")
    plt.plot(top2_m, label="Mixer")
    plt.plot(top2_h, label="Mixer+Attention")
    plt.title("Top-2 Accuracy")
    plt.legend()

    plt.show()