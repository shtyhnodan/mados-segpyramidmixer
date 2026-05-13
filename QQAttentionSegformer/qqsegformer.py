import os
import math
import json
import random
import types
import io
import contextlib
import warnings
import logging
from dataclasses import dataclass, field
from typing import Tuple, Optional, List, Dict, Any
from tqdm import tqdm

os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

from transformers.utils import logging as hf_logging
from huggingface_hub import logging as hub_logging

hf_logging.set_verbosity_error()
hub_logging.set_verbosity_error()
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from PIL import Image

from pycocotools.coco import COCO
from pycocotools import mask as maskUtils

from transformers import SegformerForSemanticSegmentation, AutoConfig


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("training.log", mode="w")]
)
logger = logging.getLogger(__name__)


@dataclass
class Config:
    coco_root: str = "./data/coco"
    train_img_dir: str = field(init=False)
    val_img_dir: str = field(init=False)
    train_ann: str = field(init=False)
    val_ann: str = field(init=False)
    out_dir: str = "./finetune_results"

    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    img_size: int = 384
    batch_size: int = 4
    num_workers: int = 0  
    epochs: int = 10
    weight_decay: float = 1e-4
    train_samples: int = 5000
    val_samples: int = 1000
    base_seed: int = 42
    compare_qq: bool = True

    segformer_backbone: str = "nvidia/segformer-b0-finetuned-ade-512-512"
    backbone_lr: float = 1e-5
    head_lr: float = 1e-4

    ce_weight: float = 1.0
    dice_weight: float = 1.0
    smooth: float = 1.0

    grad_clip_norm: float = 1.0
    warmup_ratio: float = 0.1

    def __post_init__(self):
        self.train_img_dir = os.path.join(self.coco_root, "train2017")
        self.val_img_dir = os.path.join(self.coco_root, "val2017")
        self.train_ann = os.path.join(self.coco_root, "annotations", "instances_train2017.json")
        self.val_ann = os.path.join(self.coco_root, "annotations", "instances_val2017.json")
        os.makedirs(self.out_dir, exist_ok=True)


cfg = Config()


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def worker_init_fn(worker_id: int):
    worker_seed = cfg.base_seed + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# =========================================================
# PIL helpers
# =========================================================
def resize_with_padding(img: Image.Image, size: int) -> Tuple[Image.Image, float, int, int]:
    w, h = img.size
    scale = size / max(w, h)
    new_w, new_h = int(w * scale), int(h * scale)
    img_resized = img.resize((new_w, new_h), Image.BILINEAR)

    canvas = Image.new("RGB", (size, size), (0, 0, 0))
    px = (size - new_w) // 2
    py = (size - new_h) // 2
    canvas.paste(img_resized, (px, py))
    return canvas, scale, px, py


def resize_mask(mask: np.ndarray, size: int, scale: float, px: int, py: int) -> np.ndarray:
    h, w = mask.shape
    pil = Image.fromarray(mask.astype(np.uint8), mode="L")  # 0/1 
    new_w, new_h = int(w * scale), int(h * scale)
    pil_resized = pil.resize((new_w, new_h), Image.NEAREST)

    canvas = Image.new("L", (size, size), 0)
    canvas.paste(pil_resized, (px, py))
    return np.array(canvas, dtype=np.uint8)  # 0 или 1


def coco_instance_annotations_to_binary_mask(coco: COCO, img_id: int) -> np.ndarray:
    ann_ids = coco.getAnnIds(imgIds=[img_id], iscrowd=False) 
    anns = coco.loadAnns(ann_ids)
    img_info = coco.loadImgs([img_id])[0]
    h, w = img_info["height"], img_info["width"]

    bin_mask = np.zeros((h, w), dtype=np.uint8)
    for ann in anns:
        rle = coco.annToRLE(ann)
        m = maskUtils.decode(rle)
        if m.ndim == 3:
            m = np.any(m, axis=2).astype(np.uint8)
        bin_mask = np.logical_or(bin_mask, m)
    return bin_mask.astype(np.uint8)


class ToTensorNormalize:
    def __init__(self):
        self.to_tensor = transforms.ToTensor()
        self.norm = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

    def __call__(self, sample):
        img, mask = sample["image"], sample["mask"]
        img_t = self.to_tensor(img)
        img_t = self.norm(img_t)
        mask_t = torch.from_numpy(mask.astype(np.int64))
        return {"image": img_t, "mask": mask_t}


class CocoBinaryDataset(Dataset):
    def __init__(
        self,
        img_dir: str,
        ann_file: str,
        img_size: int = cfg.img_size,
        transform=None,
        augment: bool = False,
        require_annotations: bool = True,
    ):
        with contextlib.redirect_stdout(io.StringIO()):
            self.coco = COCO(ann_file)
        self.img_dir = img_dir
        self.img_size = img_size
        self.transform = transform
        self.augment = augment

        if require_annotations:
            valid_ids = []
            for img_id in self.coco.getImgIds():
                ann_ids = self.coco.getAnnIds(imgIds=[img_id])
                if len(ann_ids) > 0:
                    valid_ids.append(img_id)
            self.ids = valid_ids
        else:
            self.ids = self.coco.getImgIds()

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        info = self.coco.loadImgs([img_id])[0]
        path = os.path.join(self.img_dir, info["file_name"])

        img = Image.open(path).convert("RGB")
        mask_full = coco_instance_annotations_to_binary_mask(self.coco, img_id)

        img_resized, scale, px, py = resize_with_padding(img, self.img_size)
        mask = resize_mask(mask_full, self.img_size, scale, px, py)

        if self.augment:
            if np.random.rand() < 0.5:
                img_resized = img_resized.transpose(Image.FLIP_LEFT_RIGHT)
                mask = np.fliplr(mask).copy()

        sample = {"image": img_resized, "mask": mask}
        if self.transform:
            sample = self.transform(sample)
        return sample


def collate_fn(batch):
    images = torch.stack([item["image"] for item in batch])
    masks = torch.stack([item["mask"].long() for item in batch])
    return {"image": images, "mask": masks}


def segmentation_losses(
    logits: torch.Tensor,
    targets: torch.Tensor,
    class_weights: torch.Tensor,
    ce_weight: float = cfg.ce_weight,
    dice_weight: float = cfg.dice_weight,
    smooth: float = cfg.smooth,
):
    ce_loss = F.cross_entropy(logits, targets.long(), weight=class_weights)

    probs = F.softmax(logits, dim=1)
    probs_fg = probs[:, 1:2]  # канал объекта
    targets_fg = (targets == 1).float().unsqueeze(1)

    dims = (0, 2, 3)
    intersection = (probs_fg * targets_fg).sum(dims)
    denom = probs_fg.sum(dims) + targets_fg.sum(dims)
    dice_score = (2.0 * intersection + smooth) / (denom + smooth)
    dice_loss = 1.0 - dice_score.mean()

    total = ce_weight * ce_loss + dice_weight * dice_loss
    return total, ce_loss.detach(), dice_loss.detach()


@torch.no_grad()
def evaluate_segformer(model, dataloader, device, class_weights):
    model.eval()

    total_loss = 0.0
    total_count = 0
    intersection = 0
    union = 0
    correct = 0
    total_pixels = 0
    dice_tp2 = 0
    pred_pos = 0
    gt_pos = 0

    for batch in dataloader:
        imgs = batch["image"].to(device)
        masks = batch["mask"].to(device)

        outputs = model(pixel_values=imgs)
        logits = outputs.logits
        logits_up = F.interpolate(logits, size=masks.shape[-2:], mode="bilinear", align_corners=False)

        loss, _, _ = segmentation_losses(logits_up, masks, class_weights)
        total_loss += loss.item() * masks.size(0)
        total_count += masks.size(0)

        preds = logits_up.argmax(dim=1)

        preds_f = preds.view(-1)
        masks_f = masks.view(-1)

        intersection += torch.logical_and(preds_f == 1, masks_f == 1).sum().item()
        union += torch.logical_or(preds_f == 1, masks_f == 1).sum().item()

        correct += (preds == masks).sum().item()
        total_pixels += masks.numel()

        tp = torch.logical_and(preds == 1, masks == 1).sum().item()
        dice_tp2 += 2 * tp
        pred_pos += (preds == 1).sum().item()
        gt_pos += (masks == 1).sum().item()

    val_loss = total_loss / max(1, total_count)
    val_iou = 1.0 if union == 0 else intersection / union
    val_acc = correct / total_pixels if total_pixels > 0 else 0.0
    val_dice = 1.0 if (pred_pos + gt_pos) == 0 else dice_tp2 / (pred_pos + gt_pos)

    return val_loss, val_iou, val_acc, val_dice



def patch_segformer_attention_with_qq(model):
    def qq_efficient_forward(
        self,
        hidden_states,
        height,
        width,
        output_attentions=False,
        **kwargs
    ):
        batch_size, seq_len, embed_dim = hidden_states.shape
        head_dim = self.attention_head_size  # 32
        all_head_size = self.all_head_size
        num_heads = all_head_size // head_dim
        scaling = head_dim ** -0.5

        # --- Q ---
        query_layer = self.query(hidden_states)
        query_layer = query_layer.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)

        # --- Sequence reduction для K и V (как в оригинале) ---
        if self.sr_ratio > 1:
            # Reshape -> sr -> reshape -> layer_norm
            kv_hidden = hidden_states.permute(0, 2, 1).reshape(batch_size, embed_dim, height, width)
            kv_hidden = self.sr(kv_hidden)
            kv_hidden = kv_hidden.reshape(batch_size, embed_dim, -1).permute(0, 2, 1)
            kv_hidden = self.layer_norm(kv_hidden)
        else:
            kv_hidden = hidden_states

        kv_seq_len = kv_hidden.shape[1]

        # --- QQ: K = Q(K) 
        key_layer = self.query(kv_hidden).view(batch_size, kv_seq_len, num_heads, head_dim).transpose(1, 2)

        # --- V ---
        value_layer = self.value(kv_hidden).view(batch_size, kv_seq_len, num_heads, head_dim).transpose(1, 2)

        query_layer = F.normalize(query_layer, dim=-1)
        key_layer = F.normalize(key_layer, dim=-1)

        # --- Attention ---
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2)) * scaling

        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_shape = context_layer.size()[:-2] + (all_head_size,)
        context_layer = context_layer.view(new_shape)

        return (context_layer, attention_probs) if output_attentions else (context_layer,)

    patched = 0
    for module in model.modules():
        if module.__class__.__name__ == "SegformerEfficientSelfAttention":
            module.forward = types.MethodType(qq_efficient_forward, module)
            patched += 1

    logger.info(f"QQ-attention patched in {patched} layers")


def compute_class_weights(dataset: CocoBinaryDataset, num_samples: int = None) -> torch.Tensor:

    bg = 0
    fg = 0
    indices = range(len(dataset))
    if num_samples is not None:
        indices = np.random.choice(len(dataset), size=min(num_samples, len(dataset)), replace=False)

    for i in tqdm(indices, desc="Computing class weights"):
        sample = dataset[i]  # без transform, чтобы получить сырой mask
        mask = sample["mask"]
        fg += (mask == 1).sum().item()
        bg += (mask == 0).sum().item()

    counts = np.array([bg, fg], dtype=np.float64)
    counts = np.clip(counts, 1.0, None)
    weights = counts.sum() / (len(counts) * counts)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


# =========================================================
# Optimizer and scheduler
# =========================================================
def build_param_groups(model, backbone_lr, head_lr, weight_decay):
    backbone_decay, backbone_no_decay = [], []
    head_decay, head_no_decay = [], []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        is_no_decay = (
            param.ndim == 1
            or name.endswith(".bias")
            or "layernorm" in name.lower()
            or "layer_norm" in name.lower()
            or "batch_norm" in name.lower()
            or name.endswith("norm.weight")
        )

        if name.startswith("decode_head."):
            if is_no_decay:
                head_no_decay.append(param)
            else:
                head_decay.append(param)
        else:
            if is_no_decay:
                backbone_no_decay.append(param)
            else:
                backbone_decay.append(param)

    return [
        {"params": backbone_decay, "lr": backbone_lr, "weight_decay": weight_decay},
        {"params": backbone_no_decay, "lr": backbone_lr, "weight_decay": 0.0},
        {"params": head_decay, "lr": head_lr, "weight_decay": weight_decay},
        {"params": head_no_decay, "lr": head_lr, "weight_decay": 0.0},
    ]


def build_scheduler(optimizer, total_steps: int, warmup_ratio: float = cfg.warmup_ratio):
    warmup_steps = max(1, int(total_steps * warmup_ratio))

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / max(1, warmup_steps - 1)  # начало с ~0
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def build_model(use_qq: bool = False) -> nn.Module:
    cfg_hf = AutoConfig.from_pretrained(cfg.segformer_backbone)
    cfg_hf.num_labels = 2
    cfg_hf.id2label = {0: "background", 1: "foreground"}
    cfg_hf.label2id = {"background": 0, "foreground": 1}

    model = SegformerForSemanticSegmentation.from_pretrained(
        cfg.segformer_backbone,
        config=cfg_hf,
        ignore_mismatched_sizes=True,
    ).to(cfg.device)

    if use_qq:
        patch_segformer_attention_with_qq(model)

    return model


def train_one_epoch(model, dataloader, optimizer, scheduler, device, class_weights, scaler=None):
    model.train()
    total_loss = 0.0
    total_count = 0
    autocast_enabled = scaler is not None and device.startswith("cuda")

    for batch in dataloader:
        imgs = batch["image"].to(device)
        masks = batch["mask"].to(device)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", enabled=autocast_enabled):
            outputs = model(pixel_values=imgs)
            logits = outputs.logits
            logits_up = F.interpolate(logits, size=masks.shape[-2:], mode="bilinear", align_corners=False)
            loss, _, _ = segmentation_losses(logits_up, masks, class_weights)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
            optimizer.step()

        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item() * masks.size(0)
        total_count += masks.size(0)

    return total_loss / max(1, total_count)


# =========================================================
# Data split helpers
# =========================================================
def build_subsets(full_train, full_val, train_samples, val_samples, seed=42):
    rng = np.random.default_rng(seed)
    train_n = min(train_samples, len(full_train))
    val_n = min(val_samples, len(full_val))
    train_indices = rng.choice(len(full_train), size=train_n, replace=False)
    val_indices = rng.choice(len(full_val), size=val_n, replace=False)
    return Subset(full_train, train_indices), Subset(full_val, val_indices)


def make_loaders(train_subset, val_subset, seed=42):
    g = torch.Generator()
    g.manual_seed(seed)

    train_loader = DataLoader(
        train_subset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        collate_fn=collate_fn,
        generator=g,
        pin_memory=(cfg.device == "cuda"),
        worker_init_fn=worker_init_fn,
    )

    val_loader = DataLoader(
        val_subset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        collate_fn=collate_fn,
        pin_memory=(cfg.device == "cuda"),
        worker_init_fn=worker_init_fn,
    )

    return train_loader, val_loader


def save_checkpoint(model, optimizer, scheduler, name, best_iou, epoch, history, use_qq, seed):
    path = os.path.join(cfg.out_dir, f"{name}_best.pth")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_iou": best_iou,
            "epoch": epoch,
            "history": history,
            "seed": seed,
            "use_qq": use_qq,
        },
        path,
    )
    return path


def run_experiment(name, use_qq, train_subset, val_subset, class_weights, seed):
    logger.info(f"=== {name} (seed={seed}) ===")
    set_seed(seed)

    train_loader, val_loader = make_loaders(train_subset, val_subset, seed=seed)

    model = build_model(use_qq=use_qq)
    class_weights_device = class_weights.to(cfg.device)

    param_groups = build_param_groups(
        model,
        backbone_lr=cfg.backbone_lr,
        head_lr=cfg.head_lr,
        weight_decay=cfg.weight_decay,
    )
    optimizer = optim.AdamW(param_groups)

    total_steps = cfg.epochs * len(train_loader)
    scheduler = build_scheduler(optimizer, total_steps=total_steps)
    scaler = torch.amp.GradScaler("cuda") if cfg.device == "cuda" else None

    history = []
    best_iou = -1.0
    best_path = None

    for epoch in range(1, cfg.epochs + 1):
        logger.info(f"{name} | epoch {epoch}/{cfg.epochs}")

        train_loss = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=cfg.device,
            class_weights=class_weights_device,
            scaler=scaler,
        )

        val_loss, val_iou, val_acc, val_dice = evaluate_segformer(
            model=model,
            dataloader=val_loader,
            device=cfg.device,
            class_weights=class_weights_device,
        )

        logger.info(
            f"{name} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
            f"IoU={val_iou:.4f} | Dice={val_dice:.4f} | Acc={val_acc:.4f}"
        )

        record = {
            "epoch": epoch,
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "val_iou": float(val_iou),
            "val_dice": float(val_dice),
            "val_acc": float(val_acc),
        }
        history.append(record)

        if val_iou > best_iou:
            best_iou = val_iou
            best_path = save_checkpoint(
                model, optimizer, scheduler, name, best_iou, epoch, history, use_qq, seed
            )

    history_path = os.path.join(cfg.out_dir, f"{name}_history.json")
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

    return history


def plot_histories(std_history, qq_history):
    metrics = [
        ("train_loss", "Train Loss", "train_loss_comparison.png", "Train Loss"),
        ("val_loss", "Val Loss", "val_loss_comparison.png", "Val Loss"),
        ("val_iou", "IoU", "iou_comparison.png", "IoU"),
        ("val_dice", "Dice", "dice_comparison.png", "Dice"),
        ("val_acc", "Acc", "acc_comparison.png", "Accuracy"),
    ]

    for key, title, filename, ylabel in metrics:
        plt.figure(figsize=(8, 5))
        plt.plot([x[key] for x in std_history], label="Standard")
        plt.plot([x[key] for x in qq_history], label="QQ")
        plt.title(title)
        plt.xlabel("Epoch")
        plt.ylabel(ylabel)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(cfg.out_dir, filename), dpi=200)
        plt.close()


def main():
    logger.info(f"Device: {cfg.device}, Workers: {cfg.num_workers}")
    set_seed(cfg.base_seed)

    weight_dataset = CocoBinaryDataset(
        cfg.train_img_dir,
        cfg.train_ann,
        img_size=cfg.img_size,  
        transform=None,          
        augment=False,
        require_annotations=True,
    )

    full_train = CocoBinaryDataset(
        cfg.train_img_dir,
        cfg.train_ann,
        img_size=cfg.img_size,
        transform=ToTensorNormalize(),
        augment=True,
        require_annotations=True,
    )
    full_val = CocoBinaryDataset(
        cfg.val_img_dir,
        cfg.val_ann,
        img_size=cfg.img_size,
        transform=ToTensorNormalize(),
        augment=False,
        require_annotations=True,
    )

    train_subset, val_subset = build_subsets(
        full_train,
        full_val,
        cfg.train_samples,
        cfg.val_samples,
        seed=cfg.base_seed,
    )
    logger.info(f"Train subset size: {len(train_subset)}")
    logger.info(f"Val subset size: {len(val_subset)}")

    class_weights = compute_class_weights(weight_dataset, num_samples=500)
    logger.info(f"Class weights: {class_weights.tolist()}")

    # Standard SegFormer
    standard_history = run_experiment(
        name="segformer_standard",
        use_qq=False,
        train_subset=train_subset,
        val_subset=val_subset,
        class_weights=class_weights,
        seed=cfg.base_seed,
    )

    if cfg.compare_qq:
        qq_history = run_experiment(
            name="segformer_QQ",
            use_qq=True,
            train_subset=train_subset,
            val_subset=val_subset,
            class_weights=class_weights,
            seed=cfg.base_seed,
        )

        std_best_iou = max(x["val_iou"] for x in standard_history)
        qq_best_iou = max(x["val_iou"] for x in qq_history)
        logger.info("=== FINAL COMPARISON ===")
        logger.info(f"Standard best IoU: {std_best_iou:.4f}")
        logger.info(f"QQ best IoU:       {qq_best_iou:.4f}")
        logger.info(f"Difference:        {qq_best_iou - std_best_iou:.4f}")

        plot_histories(standard_history, qq_history)

    logger.info("Done.")


if __name__ == "__main__":
    main()