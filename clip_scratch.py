"""
Minimal CLIP from scratch (PyTorch)
-----------------------------------
This module implements a compact, educational version of CLIP:
- Vision encoder: tiny ViT (patch embedding + TransformerEncoder)
- Text encoder: TransformerEncoder with learned token/pos embeddings
- Contrastive objective: symmetric InfoNCE with learnable temperature
- Lightweight tokenizer: whitespace w/ special tokens
- CSV dataset: columns [image_path, text]
- Training/evaluation utilities and CLI

NOTE: This is for learning & prototyping. For production, prefer
robust tokenizers (BPE/WordPiece), stronger vision backbones, and
mixed-precision/distributed training.

Usage (example):
  pip install torch torchvision pillow pandas
  python clip_scratch.py --csv data/train.csv --image-root . --epochs 10

CSV format example:
image_path,text
images/cat.jpg,"a photo of a cat"
images/dog.jpg,"a photo of a dog"

Author: Suraj Bhardwaj
License: MIT
"""
from __future__ import annotations

import argparse
import math
import os
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

# -----------------------------
# Repro & device helpers
# -----------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -----------------------------
# Tokenizer & Vocab (whitespace)
# -----------------------------
SPECIAL_TOKENS = {"<pad>": 0, "<bos>": 1, "<eos>": 2, "<unk>": 3}

class SimpleTokenizer:
    def __init__(self, vocab: Dict[str, int], max_len: int = 77):
        self.vocab = vocab
        self.inv_vocab = {i: t for t, i in vocab.items()}
        self.max_len = max_len

    @classmethod
    def build_from_texts(cls, texts: List[str], min_freq: int = 1, max_len: int = 77):
        freq: Dict[str, int] = {}
        for s in texts:
            for tok in s.lower().strip().split():
                freq[tok] = freq.get(tok, 0) + 1
        vocab = dict(SPECIAL_TOKENS)  # copy special
        idx = len(vocab)
        for tok, c in sorted(freq.items(), key=lambda x: (-x[1], x[0])):
            if c >= min_freq and tok not in vocab:
                vocab[tok] = idx
                idx += 1
        return cls(vocab=vocab, max_len=max_len)

    def encode(self, text: str) -> Tuple[torch.Tensor, torch.Tensor]:
        toks = [t for t in text.lower().strip().split()]
        ids = [SPECIAL_TOKENS["<bos>"]]
        for t in toks:
            ids.append(self.vocab.get(t, SPECIAL_TOKENS["<unk>"]))
        ids.append(SPECIAL_TOKENS["<eos>"])
        # pad/truncate
        if len(ids) < self.max_len:
            pad_len = self.max_len - len(ids)
            ids = ids + [SPECIAL_TOKENS["<pad>"]] * pad_len
            attn_mask = [1] * (len(ids) - pad_len) + [0] * pad_len
        else:
            ids = ids[:self.max_len]
            attn_mask = [1] * self.max_len
        return torch.tensor(ids, dtype=torch.long), torch.tensor(attn_mask, dtype=torch.long)

    def decode(self, ids: List[int]) -> str:
        toks = [self.inv_vocab.get(i, "<unk>") for i in ids]
        # strip specials
        toks = [t for t in toks if t not in SPECIAL_TOKENS]
        return " ".join(toks)


# -----------------------------
# Dataset
# -----------------------------
class CaptionImageDataset(Dataset):
    def __init__(self, csv_path: str | Path, image_root: str | Path, tokenizer: SimpleTokenizer,
                 image_size: int = 224):
        self.df = pd.read_csv(csv_path)
        assert {"image_path", "text"}.issubset(self.df.columns), \
            "CSV must contain columns: image_path,text"
        self.image_root = Path(image_root)
        self.tokenizer = tokenizer
        self.transform = transforms.Compose([
            transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                 std=[0.26862954, 0.26130258, 0.27577711]),  # CLIP-like
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = self.image_root / row["image_path"]
        text = str(row["text"])

        # Image
        with Image.open(img_path).convert("RGB") as im:
            image = self.transform(im)

        # Text
        ids, attn = self.tokenizer.encode(text)

        return {
            "image": image,                # (3, H, W)
            "input_ids": ids,             # (L,)
            "attention_mask": attn,       # (L,)
        }


# -----------------------------
# Vision Encoder: Tiny ViT
# -----------------------------
class PatchEmbed(nn.Module):
    def __init__(self, in_chans=3, embed_dim=384, patch_size=16, img_size=224):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        num_patches = (img_size // patch_size) ** 2
        self.num_patches = num_patches

    def forward(self, x):
        x = self.proj(x)  # (B, D, H/P, W/P)
        x = x.flatten(2).transpose(1, 2)  # (B, N, D)
        return x

class VisionEncoder(nn.Module):
    def __init__(self, img_size=224, patch_size=16, embed_dim=384, depth=6, num_heads=6, mlp_ratio=4.0, drop=0.0):
        super().__init__()
        self.patch = PatchEmbed(3, embed_dim, patch_size, img_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.randn(1, 1 + self.patch.num_patches, embed_dim) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=drop, activation="gelu", batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):  # x: (B,3,H,W)
        B = x.size(0)
        x = self.patch(x)  # (B,N,D)
        cls = self.cls_token.expand(B, -1, -1)  # (B,1,D)
        x = torch.cat([cls, x], dim=1) + self.pos_embed[:, : x.size(1)]
        x = self.encoder(x)
        x = self.norm(x[:, 0])  # CLS token
        return x  # (B,D)


# -----------------------------
# Text Encoder: Transformer Encoder (bidirectional)
# -----------------------------
class TextEncoder(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int = 384, max_len: int = 77,
                 depth: int = 6, num_heads: int = 6, mlp_ratio: float = 4.0, drop: float = 0.0):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, embed_dim)
        self.pos_emb = nn.Parameter(torch.randn(1, max_len, embed_dim) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=drop, activation="gelu", batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)
        self.max_len = max_len

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        # input_ids: (B,L) attention_mask: (B,L) with 1 for tokens, 0 for padding
        x = self.tok_emb(input_ids) + self.pos_emb[:, : input_ids.size(1)]
        # Convert attention mask to key padding mask: True for pads
        key_padding = (attention_mask == 0)  # (B,L) True where padding
        x = self.encoder(x, src_key_padding_mask=key_padding)
        # Simple pooling: take <eos> position where mask transitions or mean over valid tokens
        lengths = attention_mask.sum(dim=1).clamp(min=1)
        masked = x * attention_mask.unsqueeze(-1)
        pooled = masked.sum(dim=1) / lengths.unsqueeze(-1)
        pooled = self.norm(pooled)
        return pooled  # (B,D)


# -----------------------------
# CLIP Model
# -----------------------------
class CLIP(nn.Module):
    def __init__(self, vision_width=384, text_width=384, proj_dim=512, vocab_size=30522, max_len=77):
        super().__init__()
        self.visual = VisionEncoder(embed_dim=vision_width)
        self.text = TextEncoder(vocab_size=vocab_size, embed_dim=text_width, max_len=max_len)
        self.vision_proj = nn.Linear(vision_width, proj_dim, bias=False)
        self.text_proj = nn.Linear(text_width, proj_dim, bias=False)
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1/0.07)))  # initialize ~ 1/0.07

    def encode_image(self, image):
        x = self.visual(image)
        x = self.vision_proj(x)
        x = F.normalize(x, dim=-1)
        return x

    def encode_text(self, input_ids, attention_mask):
        x = self.text(input_ids, attention_mask)
        x = self.text_proj(x)
        x = F.normalize(x, dim=-1)
        return x

    def forward(self, image, input_ids, attention_mask):
        img = self.encode_image(image)
        txt = self.encode_text(input_ids, attention_mask)
        logit_scale = self.logit_scale.exp().clamp(1e-3, 1e3)
        logits_per_image = logit_scale * img @ txt.t()
        logits_per_text = logits_per_image.t()
        return logits_per_image, logits_per_text


# -----------------------------
# Contrastive Loss (symmetric InfoNCE)
# -----------------------------
def clip_contrastive_loss(logits_per_image: torch.Tensor, logits_per_text: torch.Tensor) -> torch.Tensor:
    batch = logits_per_image.size(0)
    labels = torch.arange(batch, device=logits_per_image.device)
    loss_i = F.cross_entropy(logits_per_image, labels)
    loss_t = F.cross_entropy(logits_per_text, labels)
    return (loss_i + loss_t) / 2.0


# -----------------------------
# Training / Eval loops
# -----------------------------
@dataclass
class TrainConfig:
    csv: str
    image_root: str = "."
    epochs: int = 5
    batch_size: int = 64
    lr: float = 5e-4
    wd: float = 0.1
    workers: int = 4
    seed: int = 42
    max_len: int = 77
    min_freq: int = 1
    image_size: int = 224
    save_dir: str = "checkpoints"

def build_tokenizer(csv_path: str, max_len: int, min_freq: int) -> SimpleTokenizer:
    df = pd.read_csv(csv_path)
    texts = [str(t) for t in df["text"].tolist()]
    tok = SimpleTokenizer.build_from_texts(texts, min_freq=min_freq, max_len=max_len)
    return tok

def train_one_epoch(model: CLIP, loader: DataLoader, opt: torch.optim.Optimizer, device: torch.device) -> Tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_correct_i = 0
    total_correct_t = 0
    total = 0
    for batch in loader:
        images = batch["image"].to(device)
        ids = batch["input_ids"].to(device)
        attn = batch["attention_mask"].to(device)

        opt.zero_grad(set_to_none=True)
        logits_i, logits_t = model(images, ids, attn)
        loss = clip_contrastive_loss(logits_i, logits_t)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        with torch.no_grad():
            bsz = images.size(0)
            labels = torch.arange(bsz, device=device)
            total_loss += loss.item() * bsz
            total += bsz
            total_correct_i += (logits_i.argmax(dim=1) == labels).sum().item()
            total_correct_t += (logits_t.argmax(dim=1) == labels).sum().item()

    avg_loss = total_loss / total
    acc_i = total_correct_i / total
    acc_t = total_correct_t / total
    return avg_loss, (acc_i + acc_t) / 2.0

@torch.no_grad()
def evaluate(model: CLIP, loader: DataLoader, device: torch.device) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct_i = 0
    total_correct_t = 0
    total = 0
    for batch in loader:
        images = batch["image"].to(device)
        ids = batch["input_ids"].to(device)
        attn = batch["attention_mask"].to(device)

        logits_i, logits_t = model(images, ids, attn)
        loss = clip_contrastive_loss(logits_i, logits_t)

        bsz = images.size(0)
        labels = torch.arange(bsz, device=device)
        total_loss += loss.item() * bsz
        total += bsz
        total_correct_i += (logits_i.argmax(dim=1) == labels).sum().item()
        total_correct_t += (logits_t.argmax(dim=1) == labels).sum().item()

    avg_loss = total_loss / total
    acc = (total_correct_i + total_correct_t) / (2 * total)
    return avg_loss, acc


# -----------------------------
# Data loaders
# -----------------------------
def build_dataloaders(cfg: TrainConfig, tokenizer: SimpleTokenizer):
    ds = CaptionImageDataset(cfg.csv, cfg.image_root, tokenizer, image_size=cfg.image_size)

    # Simple split (90/10). For proper experiments, pre-split files.
    n = len(ds)
    n_train = int(0.9 * n)
    n_val = n - n_train
    gen = torch.Generator().manual_seed(cfg.seed)
    train_ds, val_ds = torch.utils.data.random_split(ds, [n_train, n_val], generator=gen)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=cfg.workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                            num_workers=cfg.workers, pin_memory=True, drop_last=False)
    return train_loader, val_loader


# -----------------------------
# Save / Load
# -----------------------------
def save_checkpoint(model: CLIP, tokenizer: SimpleTokenizer, cfg: TrainConfig, path: str | Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "model_state": model.state_dict(),
        "logit_scale": model.logit_scale.detach().cpu(),
        "tokenizer_vocab": tokenizer.vocab,
        "config": asdict(cfg),
    }
    torch.save(ckpt, path)

def load_checkpoint(path: str | Path, device: torch.device) -> Tuple[CLIP, SimpleTokenizer, TrainConfig]:
    ckpt = torch.load(path, map_location=device)
    vocab = ckpt["tokenizer_vocab"]
    cfg_dict = ckpt["config"]
    cfg = TrainConfig(**cfg_dict)

    tokenizer = SimpleTokenizer(vocab=vocab, max_len=cfg.max_len)
    model = CLIP(vocab_size=len(vocab), max_len=cfg.max_len)
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.to(device)
    return model, tokenizer, cfg


# -----------------------------
# CLI
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="Train a minimal CLIP from scratch (educational)")
    parser.add_argument("--csv", type=str, required=True, help="CSV with columns [image_path,text]")
    parser.add_argument("--image-root", type=str, default=".", help="Root folder for images")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--wd", type=float, default=0.1)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-len", type=int, default=77)
    parser.add_argument("--min-freq", type=int, default=1)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--save-dir", type=str, default="checkpoints")
    parser.add_argument("--save-name", type=str, default="clip_minimal.pt")
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()
    print(f"[INFO] Using device: {device}")

    cfg = TrainConfig(
        csv=args.csv, image_root=args.image_root, epochs=args.epochs, batch_size=args.batch_size,
        lr=args.lr, wd=args.wd, workers=args.workers, seed=args.seed, max_len=args.max_len,
        min_freq=args.min_freq, image_size=args.image_size, save_dir=args.save_dir
    )

    # Build tokenizer from training texts
    tokenizer = build_tokenizer(cfg.csv, max_len=cfg.max_len, min_freq=cfg.min_freq)
    print(f"[INFO] Vocab size: {len(tokenizer.vocab)}")

    # Build data loaders
    train_loader, val_loader = build_dataloaders(cfg, tokenizer)

    # Build model & optimizer
    model = CLIP(vocab_size=len(tokenizer.vocab), max_len=cfg.max_len).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.wd, betas=(0.9, 0.98), eps=1e-8)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)

    best_val = float("inf")
    save_path = Path(cfg.save_dir) / args.save_name

    for epoch in range(1, cfg.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, device)
        scheduler.step()

        print(f"[EPOCH {epoch:03d}] "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} | "
              f"lr={scheduler.get_last_lr()[0]:.6f}")

        if val_loss < best_val:
            best_val = val_loss
            save_checkpoint(model, tokenizer, cfg, save_path)
            print(f"[INFO] Saved checkpoint to: {save_path}")

    print("[DONE] Training complete.")

if __name__ == "__main__":
    main()
