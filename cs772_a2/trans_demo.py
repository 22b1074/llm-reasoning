# Cell 10: beam search decode and helpers

# Cell 1: imports & device & HP
import streamlit as st
import os
import json
import math
import random
from pathlib import Path
from typing import List, Tuple, Dict
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ================================
# HYPERPARAMETERS
# ================================
HP = {
    # Paths (adjust if your files are in different locations)

    #/content/hin/hin_train.json
    "train_path": r"C:\Users\CHAPPIDI PREETHI\Documents\CS772_A2\hin\hin_train_100k_sample.json",
    "valid_path": r"C:\Users\CHAPPIDI PREETHI\Documents\CS772_A2\hin\hin_valid.json",
    "test_path": r"C:\Users\CHAPPIDI PREETHI\Documents\CS772_A2\hin\hin_test.json",

    # Model architecture
    "d_model": 512,
    "nhead": 8,
    "num_encoder_layers": 2,
    "num_decoder_layers": 2,
    "dim_feedforward": 2048,
    "dropout": 0.1,
    "activation": "relu",
    "max_seq_length": 128,

    # Training
    "batch_size": 128,
    "learning_rate": 5e-4,
    "num_epochs": 20,
    "warmup_steps": 4000,
    "label_smoothing": 0.1,
    "max_target_len": 64,
    "grad_clip": 1.0,

    # Inference
    "beam_size": 5,
    "length_penalty": 0.6,

    # Other
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "save_path": r"C:\Users\CHAPPIDI PREETHI\Documents\CS772_A2\best_transformer_model.pt",
    "seed": 42,
}

# enforce constraints
HP["num_encoder_layers"] = min(HP["num_encoder_layers"], 2)
HP["num_decoder_layers"] = min(HP["num_decoder_layers"], 2)

# reproducibility
torch.manual_seed(HP["seed"])
np.random.seed(HP["seed"])
random.seed(HP["seed"])
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(HP["seed"])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

print("Device:", HP["device"])
# Cell 2: utilities
def levenshtein(a: str, b: str) -> int:
    """Compute Levenshtein distance between two strings."""
    if a == b:
        return 0
    la, lb = len(a), len(b)
    if la == 0:
        return lb
    if lb == 0:
        return la
    prev = list(range(lb + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i] + [0] * lb
        for j, cb in enumerate(b, start=1):
            add = prev[j] + 1
            delete = cur[j - 1] + 1
            replace = prev[j - 1] + (0 if ca == cb else 1)
            cur[j] = min(add, delete, replace)
        prev = cur
    return prev[lb]

# Cell 3: dataset and collate
class TranslitDataset(Dataset):
    def __init__(self, df: pd.DataFrame, src_vocab: Dict[str,int],
                 tgt_vocab: Dict[str,int], max_tgt_len: int):
        self.srcs = df["english word"].astype(str).tolist()
        self.tgts = df["native word"].astype(str).tolist()
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.max_tgt_len = max_tgt_len

    def __len__(self):
        return len(self.srcs)

    def encode_src(self, s: str) -> List[int]:
        return [self.src_vocab.get(ch, self.src_vocab["<unk>"])
                for ch in list(s.lower())]

    def encode_tgt(self, s: str) -> List[int]:
        chars = list(s.strip())[:self.max_tgt_len - 1]
        return [self.tgt_vocab.get(ch, self.tgt_vocab["<unk>"])
                for ch in chars]

    def __getitem__(self, idx):
        return {
            "src_raw": self.srcs[idx],
            "tgt_raw": self.tgts[idx],
            "src": torch.tensor(self.encode_src(self.srcs[idx]), dtype=torch.long),
            "tgt": torch.tensor(self.encode_tgt(self.tgts[idx]), dtype=torch.long),
        }

def collate_fn(batch):
    """Collate batch with padding."""
    PAD, SOS, EOS = 0, 1, 2

    srcs = [b["src"] for b in batch]
    tgts = [b["tgt"] for b in batch]
    src_lens = [len(s) for s in srcs]
    tgt_lens = [len(t) for t in tgts]

    max_src = max(src_lens)
    max_tgt = max(tgt_lens) + 1

    src_padded = torch.full((len(batch), max_src), PAD, dtype=torch.long)
    dec_in_padded = torch.full((len(batch), max_tgt), PAD, dtype=torch.long)
    dec_target_padded = torch.full((len(batch), max_tgt), PAD, dtype=torch.long)

    for i, (s, t) in enumerate(zip(srcs, tgts)):
        src_padded[i, :s.size(0)] = s
        dec_in_padded[i, 0] = SOS
        dec_in_padded[i, 1:1+t.size(0)] = t
        dec_target_padded[i, :t.size(0)] = t
        dec_target_padded[i, t.size(0)] = EOS

    return {
        "src": src_padded,
        "src_lens": torch.tensor(src_lens, dtype=torch.long),
        "dec_in": dec_in_padded,
        "dec_target": dec_target_padded,
        "src_raws": [b["src_raw"] for b in batch],
        "tgt_raws": [b["tgt_raw"] for b in batch],
    }

# Cell 4: build vocabs
def build_vocabs(train_path, valid_path, test_path):
    """Build character vocabularies from all data splits."""
    src_chars = set()
    tgt_chars = set()

    for path in [train_path, valid_path, test_path]:
        if not os.path.exists(path):
            continue
        try:
            df = pd.read_json(path, lines=True)
        except:
            df = pd.read_json(path)

        for s in df["english word"].astype(str):
            src_chars.update(list(s.lower()))
        for t in df["native word"].astype(str):
            tgt_chars.update(list(t))

    def make_vocab(chars):
        vocab = {"<pad>": 0, "<sos>": 1, "<eos>": 2, "<unk>": 3}
        for i, ch in enumerate(sorted(chars), start=4):
            vocab[ch] = i
        inv_vocab = {v: k for k, v in vocab.items()}
        return vocab, inv_vocab

    src_vocab, inv_src = make_vocab(src_chars)
    tgt_vocab, inv_tgt = make_vocab(tgt_chars)
    return src_vocab, inv_src, tgt_vocab, inv_tgt

# Cell 5: positional encoding
class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for transformers."""

    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)

        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        x: (batch, seq_len, d_model)
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

# Cell 6: Transformer model
class TransformerTransliterator(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, nhead=8,
                 num_encoder_layers=2, num_decoder_layers=2, dim_feedforward=2048,
                 dropout=0.1, activation="relu", max_seq_length=128):
        super().__init__()

        self.d_model = d_model
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size

        self.src_embed = nn.Embedding(src_vocab_size, d_model, padding_idx=0)
        self.tgt_embed = nn.Embedding(tgt_vocab_size, d_model, padding_idx=0)

        self.pos_encoder = PositionalEncoding(d_model, max_seq_length, dropout)

        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=True
        )

        self.output_proj = nn.Linear(d_model, tgt_vocab_size)
        self._init_parameters()

    def _init_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, tgt, src_key_padding_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None):
        """
        src: (batch, src_len)
        tgt: (batch, tgt_len)
        """
        src_emb = self.src_embed(src) * math.sqrt(self.d_model)
        tgt_emb = self.tgt_embed(tgt) * math.sqrt(self.d_model)

        src_emb = self.pos_encoder(src_emb)
        tgt_emb = self.pos_encoder(tgt_emb)

        tgt_len = tgt.size(1)
        tgt_mask = self.generate_square_subsequent_mask(tgt_len).to(tgt.device)

        output = self.transformer(
            src_emb, tgt_emb,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask
        )

        logits = self.output_proj(output)
        return logits

    @staticmethod
    def generate_square_subsequent_mask(sz):
        mask = torch.triu(torch.ones(sz, sz), diagonal=1)
        mask = mask.masked_fill(mask == 1, float("-inf"))
        return mask

# Cell 7: LR scheduler (transformer warmup style)
class TransformerLRScheduler:
    """Learning rate scheduler with warmup for transformers."""

    def __init__(self, optimizer, d_model, warmup_steps=4000):
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.step_num = 0

    def step(self):
        self.step_num += 1
        lr = self._get_lr()
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def _get_lr(self):
        return (self.d_model ** -0.5) * min(
            self.step_num ** -0.5,
            self.step_num * (self.warmup_steps ** -1.5)
        )

    def get_last_lr(self):
        return [self._get_lr()]
# Cell 11: main (data loading)

hp = HP
device = torch.device(hp["device"])
print(f"Device: {device}")
print(f"Hyperparameters:\n{json.dumps(hp, indent=2)}\n")


# Cell 12: build vocabs, dataloaders, model, optimizer, criterion
print("Building vocabularies...")
src_vocab, inv_src, tgt_vocab, inv_tgt = build_vocabs(
    hp["train_path"], hp["valid_path"], hp["test_path"]
)
print(f"Source vocab: {len(src_vocab):,} | Target vocab: {len(tgt_vocab):,}\n")

print("Building Transformer model...")
model = TransformerTransliterator(
    src_vocab_size=len(src_vocab),
    tgt_vocab_size=len(tgt_vocab),
    d_model=hp["d_model"],
    nhead=hp["nhead"],
    num_encoder_layers=hp["num_encoder_layers"],
    num_decoder_layers=hp["num_decoder_layers"],
    dim_feedforward=hp["dim_feedforward"],
    dropout=hp["dropout"],
    activation=hp["activation"],
    max_seq_length=hp["max_seq_length"]
).to(device)

n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Model parameters: {n_params:,}\n")

optimizer = torch.optim.Adam(model.parameters(), lr=hp["learning_rate"], betas=(0.9, 0.98), eps=1e-9)
scheduler = TransformerLRScheduler(optimizer, hp["d_model"], hp["warmup_steps"])
criterion = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=hp["label_smoothing"])

ckpt = torch.load(hp["save_path"], map_location=device)
model.load_state_dict(ckpt["model_state"])
model.eval()
def beam_search_decode(model, src_ids, inv_tgt, hp):
    """Beam search decoding for a single source sequence."""
    device = hp["device"]
    beam_size = hp["beam_size"]
    alpha = hp["length_penalty"]
    SOS, EOS = 1, 2
    max_len = hp["max_target_len"]

    model.eval()

    src_tensor = torch.tensor([src_ids], dtype=torch.long, device=device)
    src_pad_mask = (src_tensor == 0)

    with torch.no_grad():
        beams = [(0.0, [SOS])]
        completed = []

        for step in range(max_len):
            candidates = []

            for score, seq in beams:
                if seq[-1] == EOS:
                    completed.append((score, seq))
                    continue

                dec_input = torch.tensor([seq], dtype=torch.long, device=device)
                tgt_pad_mask = (dec_input == 0)

                logits = model(src_tensor, dec_input,
                             src_key_padding_mask=src_pad_mask,
                             tgt_key_padding_mask=tgt_pad_mask)

                log_probs = F.log_softmax(logits[0, -1, :], dim=-1)
                topk_probs, topk_ids = log_probs.topk(beam_size)

                for prob, idx in zip(topk_probs, topk_ids):
                    new_score = score + prob.item()
                    new_seq = seq + [idx.item()]
                    candidates.append((new_score, new_seq))

            candidates.sort(key=lambda x: x[0] / (len(x[1]) ** alpha), reverse=True)
            beams = candidates[:beam_size]

            if not beams:
                break

        all_hyps = completed + beams
        if not all_hyps:
            return ""

        best_seq = max(all_hyps, key=lambda x: x[0] / (len(x[1]) ** alpha))[1]

        pred_chars = []
        for pid in best_seq[1:]:  # Skip SOS
            if pid == EOS:
                break
            ch = inv_tgt.get(pid, "")
            if ch not in ("<pad>", "<sos>", "<eos>"):
                pred_chars.append(ch if ch != "<unk>" else "")

        return "".join(pred_chars)

def infer_word(word: str, model, src_vocab, inv_tgt, hp) -> str:
    ids = [src_vocab.get(ch, src_vocab["<unk>"]) for ch in list(word.lower())]
    return beam_search_decode(model, ids, inv_tgt, hp)

def infer_sentence(sentence: str, model, src_vocab, inv_tgt, hp) -> str:
    tokens = sentence.strip().split()
    preds = [infer_word(tok, model, src_vocab, inv_tgt, hp) for tok in tokens]
    return " ".join(preds)

st.title("Transliteration Demo")

sentence_input = st.text_area("Enter sentence to transliterate:", "")

# Submit button
if st.button("Transliterate"):
    if sentence_input.strip() == "":
        st.warning("Please enter some text!")
    else:
        # Call your inference function
        prediction = infer_sentence(sentence_input, model, src_vocab, inv_tgt, HP)
        st.success("Prediction:")
        st.write(prediction)