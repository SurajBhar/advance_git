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

----------- Extension ---------------------
CLIP from Scratch — Plus Edition (PyTorch)
------------------------------------------
Adds:
  - Image–text retrieval metrics: Recall@K (1,5,10) for I->T and T->I
  - Embeddings export to .npz (image/text matrices)
  - Byte Pair Encoding (BPE) tokenizer option (trained on CSV texts)
  - Configurable vision backbones: tiny/deeper ViT or ResNet18/ResNet50

Requirements:
  pip install torch torchvision pillow pandas

CSV format (two columns):
image_path,text

Example:
images/cat.jpg,"a photo of a cat"

Note:
  - This is an educational/minimal implementation. The BPE here is simple and
    CPU-only, suitable for small corpora. For production, use a robust tokenizer.
  - ResNet weights default to randomly-initialized to avoid downloads.

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
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models

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
# Tokenizers
# -----------------------------
SPECIAL_TOKENS = {"<pad>": 0, "<bos>": 1, "<eos>": 2, "<unk>": 3}

class SimpleTokenizer:
    """Whitespace tokenizer with frequency-based vocab."""
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
        vocab = dict(SPECIAL_TOKENS)
        idx = len(vocab)
        for tok, c in sorted(freq.items(), key=lambda x: (-x[1], x[0])):
            if c >= min_freq and tok not in vocab:
                vocab[tok] = idx
                idx += 1
        return cls(vocab=vocab, max_len=max_len)

    def encode(self, text: str) -> Tuple[torch.Tensor, torch.Tensor]:
        toks = text.lower().strip().split()
        ids = [SPECIAL_TOKENS["<bos>"]] + [self.vocab.get(t, SPECIAL_TOKENS["<unk>"]) for t in toks] + [SPECIAL_TOKENS["<eos>"]]
        if len(ids) < self.max_len:
            pad_len = self.max_len - len(ids)
            attn = [1] * (len(ids)) + [0] * pad_len
            ids = ids + [SPECIAL_TOKENS["<pad>"]] * pad_len
        else:
            ids = ids[: self.max_len]
            attn = [1] * self.max_len
        return torch.tensor(ids, dtype=torch.long), torch.tensor(attn, dtype=torch.long)

    def decode(self, ids: List[int]) -> str:
        toks = [self.inv_vocab.get(i, "<unk>") for i in ids if i in self.inv_vocab]
        toks = [t for t in toks if t not in SPECIAL_TOKENS]
        return " ".join(toks)


class BPETokenizer:
    """
    Minimal BPE tokenizer trainer/encoder (educational).
    Trains merges on a small corpus. Uses word boundary marker '</w>'.
    """
    def __init__(self, vocab: Dict[str, int], merges: List[Tuple[str, str]], max_len: int = 77):
        self.vocab = vocab
        self.inv_vocab = {i: t for t, i in vocab.items()}
        self.merges = merges  # list of pair merges in order
        self.max_len = max_len
        # Build a ranking dict for merges to encode greedily
        self.merge_ranks = {pair: i for i, pair in enumerate(merges)}

    @staticmethod
    def _get_stats(words: List[List[str]]) -> Dict[Tuple[str, str], int]:
        stats: Dict[Tuple[str, str], int] = {}
        for w in words:
            for i in range(len(w) - 1):
                pair = (w[i], w[i + 1])
                stats[pair] = stats.get(pair, 0) + 1
        return stats

    @staticmethod
    def _merge_word(word: List[str], pair: Tuple[str, str]) -> List[str]:
        i = 0
        merged: List[str] = []
        while i < len(word):
            if i < len(word) - 1 and (word[i], word[i + 1]) == pair:
                merged.append(word[i] + word[i + 1])
                i += 2
            else:
                merged.append(word[i])
                i += 1
        return merged

    @classmethod
    def train(cls, texts: List[str], vocab_size: int = 10000, max_len: int = 77, min_freq: int = 1):
        # Initialize with special tokens + character vocab with </w> word boundary
        vocab = dict(SPECIAL_TOKENS)
        idx = len(vocab)

        # Build initial corpus as list of words (characters + '</w>')
        corpus_words: List[List[str]] = []
        freq: Dict[Tuple[str, ...], int] = {}
        for s in texts:
            for token in s.lower().strip().split():
                chars = list(token) + ["</w>"]
                tup = tuple(chars)
                freq[tup] = freq.get(tup, 0) + 1
        # Construct initial lexicon
        lexicon = []
        for word, c in freq.items():
            if c >= min_freq:
                lexicon.append((list(word), c))

        # Initialize vocab with individual symbols
        symbols = set()
        for word, c in lexicon:
            symbols.update(word)
        for sym in sorted(symbols):
            if sym not in vocab:
                vocab[sym] = idx
                idx += 1

        merges: List[Tuple[str, str]] = []
        # Iteratively merge most frequent pairs until reaching vocab size
        while len(vocab) < vocab_size:
            words_expanded = []
            for word, c in lexicon:
                for _ in range(c):
                    words_expanded.append(word)
            stats = cls._get_stats(words_expanded)
            if not stats:
                break
            pair = max(stats.items(), key=lambda x: x[1])[0]
            merges.append(pair)
            # Update lexicon by merging the chosen pair in every word
            new_lexicon = []
            for word, c in lexicon:
                merged = cls._merge_word(word, pair)
                new_lexicon.append((merged, c))
            lexicon = new_lexicon
            # Add new symbol to vocab
            new_sym = pair[0] + pair[1]
            if new_sym not in vocab:
                vocab[new_sym] = idx
                idx += 1

        return cls(vocab=vocab, merges=merges, max_len=max_len)

    def _bpe(self, token: str) -> List[str]:
        # Encode a single whitespace token into BPE subwords
        word = list(token) + ["</w>"]
        if not word:
            return []
        pairs = [(word[i], word[i+1]) for i in range(len(word)-1)]
        if not pairs:
            return word

        while True:
            candidate = None
            best_rank = None
            for p in pairs:
                r = self.merge_ranks.get(p, None)
                if r is not None and (best_rank is None or r < best_rank):
                    best_rank = r
                    candidate = p
            if candidate is None:
                break
            # merge the best candidate
            new_word = []
            i = 0
            while i < len(word):
                if i < len(word)-1 and (word[i], word[i+1]) == candidate:
                    new_word.append(word[i] + word[i+1])
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            word = new_word
            if len(word) == 1:
                break
            pairs = [(word[i], word[i+1]) for i in range(len(word)-1)]
        return word

    def encode(self, text: str) -> Tuple[torch.Tensor, torch.Tensor]:
        tokens = []
        for w in text.lower().strip().split():
            tokens.extend(self._bpe(w))
        ids = [SPECIAL_TOKENS["<bos>"]] + [self.vocab.get(t, SPECIAL_TOKENS["<unk>"]) for t in tokens] + [SPECIAL_TOKENS["<eos>"]]
        if len(ids) < self.max_len:
            pad_len = self.max_len - len(ids)
            attn = [1] * (len(ids)) + [0] * pad_len
            ids = ids + [SPECIAL_TOKENS["<pad>"]] * pad_len
        else:
            ids = ids[: self.max_len]
            attn = [1] * self.max_len
        return torch.tensor(ids, dtype=torch.long), torch.tensor(attn, dtype=torch.long)


# -----------------------------
# Dataset
# -----------------------------
class CaptionImageDataset(Dataset):
    def __init__(self, csv_path: str | Path, image_root: str | Path, tokenizer, image_size: int = 224):
        self.df = pd.read_csv(csv_path)
        assert {"image_path", "text"}.issubset(self.df.columns), "CSV must contain columns: image_path,text"
        self.image_root = Path(image_root)
        self.tokenizer = tokenizer
        self.transform = transforms.Compose([
            transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                 std=[0.26862954, 0.26130258, 0.27577711]),
        ])

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = self.image_root / row["image_path"]
        text = str(row["text"])

        with Image.open(img_path).convert("RGB") as im:
            image = self.transform(im)

        ids, attn = self.tokenizer.encode(text)
        return {"image": image, "input_ids": ids, "attention_mask": attn}


# -----------------------------
# Vision Encoders
# -----------------------------
class PatchEmbed(nn.Module):
    def __init__(self, in_chans=3, embed_dim=384, patch_size=16, img_size=224):
        super().__init__();
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.num_patches = (img_size // patch_size) ** 2

    def forward(self, x):
        x = self.proj(x)        # (B, D, H/P, W/P)
        x = x.flatten(2).transpose(1, 2)  # (B, N, D)
        return x

class VisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, embed_dim=384, depth=6, num_heads=6, mlp_ratio=4.0, drop=0.0):
        super().__init__()
        self.patch = PatchEmbed(3, embed_dim, patch_size, img_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.randn(1, 1 + self.patch.num_patches, embed_dim) * 0.02)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=drop, activation="gelu", batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        B = x.size(0)
        x = self.patch(x)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1) + self.pos_embed[:, : x.size(1)]
        x = self.encoder(x)
        x = self.norm(x[:, 0])
        return x  # (B, D)

class ResNetEncoder(nn.Module):
    def __init__(self, variant: str = "resnet18", out_dim: int = 512, pretrained: bool = False):
        super().__init__()
        if variant == "resnet18":
            net = models.resnet18(weights=None if not pretrained else models.ResNet18_Weights.DEFAULT)
            feat_dim = 512
        elif variant == "resnet50":
            net = models.resnet50(weights=None if not pretrained else models.ResNet50_Weights.DEFAULT)
            feat_dim = 2048
        else:
            raise ValueError("variant must be 'resnet18' or 'resnet50'")
        # remove classifier head
        self.backbone = nn.Sequential(*(list(net.children())[:-1]))  # up to avgpool
        self.fc = nn.Linear(feat_dim, out_dim)

    def forward(self, x):
        x = self.backbone(x)        # (B, C, 1, 1)
        x = torch.flatten(x, 1)     # (B, C)
        x = self.fc(x)              # (B, out_dim)
        return x


# -----------------------------
# Text Encoder
# -----------------------------
class TextEncoder(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int = 384, max_len: int = 77,
                 depth: int = 6, num_heads: int = 6, mlp_ratio: float = 4.0, drop: float = 0.0):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, embed_dim)
        self.pos_emb = nn.Parameter(torch.randn(1, max_len, embed_dim) * 0.02)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=drop, activation="gelu", batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)
        self.max_len = max_len

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        x = self.tok_emb(input_ids) + self.pos_emb[:, : input_ids.size(1)]
        key_padding = (attention_mask == 0)
        x = self.encoder(x, src_key_padding_mask=key_padding)
        lengths = attention_mask.sum(dim=1).clamp(min=1)
        pooled = (x * attention_mask.unsqueeze(-1)).sum(dim=1) / lengths.unsqueeze(-1)
        pooled = self.norm(pooled)
        return pooled


# -----------------------------
# CLIP Model
# -----------------------------
class CLIP(nn.Module):
    def __init__(self, vision_backbone: str = "vit_tiny", proj_dim: int = 512,
                 vocab_size: int = 30522, max_len: int = 77,
                 vit_embed_dim: int = 384, vit_depth: int = 6, vit_heads: int = 6,
                 resnet_variant: str = "resnet18", resnet_out_dim: int = 512, resnet_pretrained: bool = False,
                 text_width: int = 384):
        super().__init__()

        if vision_backbone.startswith("vit"):
            # options: vit_tiny, vit_small
            if vision_backbone == "vit_tiny":
                embed_dim = vit_embed_dim
                depth, heads = vit_depth, vit_heads
            elif vision_backbone == "vit_small":
                embed_dim = max(512, vit_embed_dim)
                depth = max(8, vit_depth)
                heads = max(8, vit_heads)
            else:
                raise ValueError("Unknown vit backbone; use vit_tiny or vit_small")
            self.visual = VisionTransformer(embed_dim=embed_dim, depth=depth, num_heads=heads)
            vision_width = embed_dim
        elif vision_backbone.startswith("resnet"):
            self.visual = ResNetEncoder(variant=resnet_variant, out_dim=resnet_out_dim, pretrained=resnet_pretrained)
            vision_width = resnet_out_dim
        else:
            raise ValueError("vision_backbone must be 'vit_tiny', 'vit_small', or 'resnet18'/'resnet50'")

        self.text = TextEncoder(vocab_size=vocab_size, embed_dim=text_width, max_len=max_len)
        self.vision_proj = nn.Linear(vision_width, proj_dim, bias=False)
        self.text_proj = nn.Linear(text_width, proj_dim, bias=False)
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1/0.07)))

    def encode_image(self, image):
        x = self.visual(image)
        x = self.vision_proj(x)
        return F.normalize(x, dim=-1)

    def encode_text(self, input_ids, attention_mask):
        x = self.text(input_ids, attention_mask)
        x = self.text_proj(x)
        return F.normalize(x, dim=-1)

    def forward(self, image, input_ids, attention_mask):
        img = self.encode_image(image)
        txt = self.encode_text(input_ids, attention_mask)
        logit_scale = self.logit_scale.exp().clamp(1e-3, 1e3)
        logits_per_image = logit_scale * img @ txt.t()
        logits_per_text = logits_per_image.t()
        return logits_per_image, logits_per_text


# -----------------------------
# Loss
# -----------------------------
def clip_contrastive_loss(logits_per_image: torch.Tensor, logits_per_text: torch.Tensor) -> torch.Tensor:
    batch = logits_per_image.size(0)
    labels = torch.arange(batch, device=logits_per_image.device)
    loss_i = F.cross_entropy(logits_per_image, labels)
    loss_t = F.cross_entropy(logits_per_text, labels)
    return 0.5 * (loss_i + loss_t)


# -----------------------------
# Training / Eval
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
    vocab_size: int = 10000  # for BPE
    image_size: int = 224
    save_dir: str = "checkpoints"
    tokenizer: str = "bpe"  # 'bpe' or 'ws'
    vision_backbone: str = "vit_tiny"  # 'vit_tiny','vit_small','resnet18','resnet50'
    resnet_pretrained: bool = False

def build_tokenizer(cfg: TrainConfig):
    df = pd.read_csv(cfg.csv)
    texts = [str(t) for t in df["text"].tolist()]
    if cfg.tokenizer.lower() == "ws":
        tok = SimpleTokenizer.build_from_texts(texts, min_freq=cfg.min_freq, max_len=cfg.max_len)
    else:
        tok = BPETokenizer.train(texts, vocab_size=cfg.vocab_size, max_len=cfg.max_len, min_freq=cfg.min_freq)
    return tok

def build_dataloaders(cfg: TrainConfig, tokenizer):
    ds = CaptionImageDataset(cfg.csv, cfg.image_root, tokenizer, image_size=cfg.image_size)
    n = len(ds)
    n_train = max(1, int(0.9 * n))
    n_val = max(1, n - n_train)
    gen = torch.Generator().manual_seed(cfg.seed)
    train_ds, val_ds = torch.utils.data.random_split(ds, [n_train, n_val], generator=gen)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=cfg.workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                            num_workers=cfg.workers, pin_memory=True, drop_last=False)
    return train_loader, val_loader

def train_one_epoch(model: CLIP, loader: DataLoader, opt: torch.optim.Optimizer, device: torch.device):
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
    avg_acc = (total_correct_i + total_correct_t) / (2 * total)
    return avg_loss, avg_acc

@torch.no_grad()
def compute_embeddings(model: CLIP, loader: DataLoader, device: torch.device):
    model.eval()
    img_embs = []
    txt_embs = []
    for batch in loader:
        images = batch["image"].to(device)
        ids = batch["input_ids"].to(device)
        attn = batch["attention_mask"].to(device)
        img = model.encode_image(images)
        txt = model.encode_text(ids, attn)
        img_embs.append(img.cpu().numpy())
        txt_embs.append(txt.cpu().numpy())
    img_embs = np.vstack(img_embs)
    txt_embs = np.vstack(txt_embs)
    return img_embs, txt_embs

def recall_at_k(sim: np.ndarray, k: int, axis: int = 1) -> float:
    """
    sim: similarity matrix (N x N)
    axis=1 -> I->T (rows: images, cols: texts)
    axis=0 -> T->I
    """
    N = sim.shape[0]
    correct = 0
    if axis == 1:  # I->T
        ranks = np.argsort(-sim, axis=1)
        for i in range(N):
            if i in ranks[i, :k]:
                correct += 1
    else:  # T->I
        ranks = np.argsort(-sim, axis=0)
        for j in range(N):
            if j in ranks[:k, j]:
                correct += 1
    return correct / N

@torch.no_grad()
def evaluate(model: CLIP, loader: DataLoader, device: torch.device, outdir: Path, split_name: str = "val"):
    # Compute loss/acc like training but also export embeddings and Recall@K
    model.eval()
    total_loss = 0.0
    total_correct_i = 0
    total_correct_t = 0
    total = 0
    # Also accumulate logits for loss
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
    avg_acc = (total_correct_i + total_correct_t) / (2 * total)

    # Embeddings & retrieval
    img_embs, txt_embs = compute_embeddings(model, loader, device)
    sim = img_embs @ txt_embs.T
    r1_i2t = recall_at_k(sim, 1, axis=1)
    r5_i2t = recall_at_k(sim, 5, axis=1)
    r10_i2t = recall_at_k(sim, 10, axis=1)
    r1_t2i = recall_at_k(sim, 1, axis=0)
    r5_t2i = recall_at_k(sim, 5, axis=0)
    r10_t2i = recall_at_k(sim, 10, axis=0)

    # Export embeddings
    outdir.mkdir(parents=True, exist_ok=True)
    np.savez(outdir / f"embeddings_{split_name}.npz", image_embeddings=img_embs, text_embeddings=txt_embs)

    metrics = {
        "loss": avg_loss,
        "acc": avg_acc,
        "R@1_I2T": r1_i2t, "R@5_I2T": r5_i2t, "R@10_I2T": r10_i2t,
        "R@1_T2I": r1_t2i, "R@5_T2I": r5_t2i, "R@10_T2I": r10_t2i,
    }
    return metrics

# -----------------------------
# Save / Load
# -----------------------------
def save_checkpoint(model: CLIP, tokenizer, cfg: TrainConfig, path: str | Path):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    ckpt = {"model_state": model.state_dict(),
            "logit_scale": model.logit_scale.detach().cpu(),
            "tokenizer_type": type(tokenizer).__name__,
            "tokenizer_payload": {
                "vocab": tokenizer.vocab,
                "merges": getattr(tokenizer, "merges", None),
                "max_len": tokenizer.max_len,
            },
            "config": asdict(cfg)}
    torch.save(ckpt, path)

def load_checkpoint(path: str | Path, device: torch.device) -> Tuple[CLIP, object, TrainConfig]:
    ckpt = torch.load(path, map_location=device)
    cfg = TrainConfig(**ckpt["config"])
    tok_type = ckpt.get("tokenizer_type", "SimpleTokenizer")
    payload = ckpt["tokenizer_payload"]
    if tok_type == "BPETokenizer":
        tokenizer = BPETokenizer(vocab=payload["vocab"], merges=payload["merges"], max_len=payload["max_len"])
    else:
        tokenizer = SimpleTokenizer(vocab=payload["vocab"], max_len=payload["max_len"])
    model = CLIP(
        vision_backbone=cfg.vision_backbone,
        proj_dim=512,
        vocab_size=len(tokenizer.vocab),
        max_len=cfg.max_len,
        resnet_variant="resnet50" if cfg.vision_backbone == "resnet50" else "resnet18",
        resnet_pretrained=cfg.resnet_pretrained,
    ).to(device)
    model.load_state_dict(ckpt["model_state"], strict=True)
    return model, tokenizer, cfg


# -----------------------------
# CLI
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="CLIP from scratch — Plus Edition")
    parser.add_argument("--csv", type=str, required=True, help="CSV with [image_path,text]")
    parser.add_argument("--image-root", type=str, default=".", help="Root folder for images")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--wd", type=float, default=0.1)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-len", type=int, default=77)
    parser.add_argument("--min-freq", type=int, default=1)
    parser.add_argument("--vocab-size", type=int, default=10000, help="BPE vocab size (if tokenizer=bpe)")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--save-dir", type=str, default="checkpoints")
    parser.add_argument("--save-name", type=str, default="clip_plus.pt")
    parser.add_argument("--tokenizer", type=str, choices=["bpe", "ws"], default="bpe")
    parser.add_argument("--vision-backbone", type=str, choices=["vit_tiny","vit_small","resnet18","resnet50"], default="vit_tiny")
    parser.add_argument("--resnet-pretrained", action="store_true", help="Use torchvision pretrained weights (may require download)")

    args = parser.parse_args()
    set_seed(args.seed); device = get_device()
    print(f"[INFO] Device: {device}")

    cfg = TrainConfig(
        csv=args.csv, image_root=args.image_root, epochs=args.epochs, batch_size=args.batch_size,
        lr=args.lr, wd=args.wd, workers=args.workers, seed=args.seed, max_len=args.max_len,
        min_freq=args.min_freq, vocab_size=args.vocab_size, image_size=args.image_size,
        save_dir=args.save_dir, tokenizer=args.tokenizer, vision_backbone=args.vision_backbone,
        resnet_pretrained=args.resnet_pretrained
    )

    # Tokenizer
    tokenizer = build_tokenizer(cfg)
    print(f"[INFO] Tokenizer: {type(tokenizer).__name__} | Vocab size: {len(tokenizer.vocab)}")

    # Data
    train_loader, val_loader = build_dataloaders(cfg, tokenizer)

    # Model & Opt
    model = CLIP(
        vision_backbone=cfg.vision_backbone,
        proj_dim=512,
        vocab_size=len(tokenizer.vocab),
        max_len=cfg.max_len,
        resnet_variant="resnet50" if cfg.vision_backbone == "resnet50" else "resnet18",
        resnet_pretrained=cfg.resnet_pretrained,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.wd, betas=(0.9, 0.98), eps=1e-8)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)

    outdir = Path(cfg.save_dir); outdir.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")

    for epoch in range(1, cfg.epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, device)
        metrics = evaluate(model, val_loader, device, outdir=outdir, split_name="val")
        scheduler.step()

        print(f"[EPOCH {epoch:03d}] "
              f"train_loss={tr_loss:.4f} train_acc={tr_acc:.4f} | "
              f"val_loss={metrics['loss']:.4f} val_acc={metrics['acc']:.4f} | "
              f"R@1 I2T={metrics['R@1_I2T']:.3f} T2I={metrics['R@1_T2I']:.3f} | "
              f"R@5 I2T={metrics['R@5_I2T']:.3f} T2I={metrics['R@5_T2I']:.3f} | "
              f"R@10 I2T={metrics['R@10_I2T']:.3f} T2I={metrics['R@10_T2I']:.3f} | "
              f"lr={scheduler.get_last_lr()[0]:.6f}")

        if metrics["loss"] < best_val:
            best_val = metrics["loss"]
            save_path = Path(cfg.save_dir) / args.save_name
            save_checkpoint(model, tokenizer, cfg, save_path)
            print(f"[INFO] Saved checkpoint to: {save_path}")

    # Export final train embeddings too (for convenience)
    train_img, train_txt = compute_embeddings(model, train_loader, device)
    np.savez(outdir / "embeddings_train.npz", image_embeddings=train_img, text_embeddings=train_txt)

    print("[DONE] Training complete. Embeddings saved to:", str(outdir))

if __name__ == "__main__":
    main()

