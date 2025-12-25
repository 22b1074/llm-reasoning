# imports & HP initialization
import streamlit as st
import os
import json
import math
import random
from typing import List, Tuple, Dict
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ---------------------------
# HYPERPARAMETERS (use your file paths)
# ---------------------------
HP = {
    "train_path": r"C:\Users\CHAPPIDI PREETHI\Documents\CS772_A2\hin\hin_train_100k_sample.json",
    "valid_path": r"C:\Users\CHAPPIDI PREETHI\Documents\CS772_A2\hin\hin_valid.json",
    "test_path": r"C:\Users\CHAPPIDI PREETHI\Documents\CS772_A2\hin\hin_test.json",
    "batch_size": 256,
    "embed_size": 128,
    "hidden_size": 256,
    "num_layers": 2,           # MUST be <= 2
    "bidirectional_encoder": True,
    "dropout": 0.4,
    "learning_rate": 1e-3,
    "num_epochs": 16,
    "teacher_forcing_ratio": 0.5,
    "max_target_len": 64,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "save_path": r"C:\Users\CHAPPIDI PREETHI\Documents\CS772_A2\sampled_lstm_model.pt",
    "seed": 42,
    "num_workers": 2,
}

if HP["num_layers"] > 2:
    raise ValueError("num_layers must be <= 2 per your constraint.")

torch.manual_seed(HP["seed"])
np.random.seed(HP["seed"])
random.seed(HP["seed"])
print("Device:", HP["device"])


# Cell 2: utilities
from typing import List
def levenshtein(a: str, b: str) -> int:
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

def ensure_cols(df: pd.DataFrame, cols: List[str]):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

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

# Cell 4: build_vocabs (from train+valid+test)
def build_vocabs(train_path, valid_path, test_path):
    paths = [train_path, valid_path, test_path]
    src_chars = set()
    tgt_chars = set()

    for p in paths:
        if not os.path.exists(p):
            continue
        try:
            df = pd.read_json(p, lines=True)
        except Exception:
            df = pd.read_json(p)
        ensure_cols(df, ["english word", "native word"])
        for s in df["english word"].astype(str).tolist():
            src_chars.update(list(s.lower()))
        for t in df["native word"].astype(str).tolist():
            tgt_chars.update(list(t))

    def make_vocab(chars):
        idx = 4
        vocab = {"<pad>":0, "<sos>":1, "<eos>":2, "<unk>":3}
        for ch in sorted(chars):
            if ch in vocab:
                continue
            vocab[ch] = idx
            idx += 1
        inv = {v:k for k,v in vocab.items()}
        return vocab, inv

    src_vocab, inv_src = make_vocab(src_chars)
    tgt_vocab, inv_tgt = make_vocab(tgt_chars)
    return src_vocab, inv_src, tgt_vocab, inv_tgt

# Cell 5: Encoder & Decoder
class Encoder(nn.Module):
    def __init__(self, input_dim, embed_dim, hidden_dim, num_layers=1, dropout=0.1, bidirectional=True):
        super().__init__()
        self.embed = nn.Embedding(input_dim, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(input_size=embed_dim,
                            hidden_size=hidden_dim,
                            num_layers=num_layers,
                            dropout=dropout if num_layers>1 else 0.0,
                            bidirectional=bidirectional,
                            batch_first=True)

    def forward(self, src, src_lens):
        embedded = self.embed(src)
        packed = nn.utils.rnn.pack_padded_sequence(embedded, src_lens, batch_first=True, enforce_sorted=False)
        packed_out, (h_n, c_n) = self.lstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        return out, (h_n, c_n)

class Decoder(nn.Module):
    def __init__(self, output_dim, embed_dim, hidden_dim, num_layers=1, dropout=0.1):
        super().__init__()
        self.embed = nn.Embedding(output_dim, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(input_size=embed_dim,
                            hidden_size=hidden_dim,
                            num_layers=num_layers,
                            dropout=dropout if num_layers>1 else 0.0,
                            batch_first=True)
        self.out = nn.Linear(hidden_dim, output_dim)

    def forward(self, dec_in, hidden):
        embedded = self.embed(dec_in)
        outputs, hidden = self.lstm(embedded, hidden)
        logits = self.out(outputs)
        return logits, hidden

#instantiate encoder, decoder, create projection layers once, optimizer, criterion
src_vocab, inv_src, tgt_vocab, inv_tgt=build_vocabs(HP['train_path'], HP['valid_path'], HP['test_path'])
input_dim = len(src_vocab)
output_dim = len(tgt_vocab)

enc = Encoder(input_dim=input_dim,
              embed_dim=HP["embed_size"],
              hidden_dim=HP["hidden_size"],
              num_layers=HP["num_layers"],
              dropout=HP["dropout"],
              bidirectional=HP["bidirectional_encoder"]).to(HP["device"])

# decide decoder hidden size: match encoder concat if bidir, else same
dec_hidden_size = HP["hidden_size"] * (2 if HP["bidirectional_encoder"] else 1)
dec = Decoder(output_dim=output_dim,
              embed_dim=HP["embed_size"],
              hidden_dim=dec_hidden_size,
              num_layers=HP["num_layers"],
              dropout=HP["dropout"]).to(HP["device"])

# PROJECTIONS: if encoder is bidir and encoder_hidden*2 != decoder_hidden, create trainable proj layers
proj_h = None
proj_c = None
if HP["bidirectional_encoder"] and (HP["hidden_size"] * 2 != dec_hidden_size):
    proj_h = nn.Linear(HP["hidden_size"] * 2, dec_hidden_size).to(HP["device"])
    proj_c = nn.Linear(HP["hidden_size"] * 2, dec_hidden_size).to(HP["device"])

# single optimizer covering encoder, decoder, and projection (if any)
params = list(enc.parameters()) + list(dec.parameters())
if proj_h is not None and proj_c is not None:
    params += list(proj_h.parameters()) + list(proj_c.parameters())

optimizer = torch.optim.Adam(params, lr=HP["learning_rate"])
criterion = nn.CrossEntropyLoss(ignore_index=0)

# Cell 7:  prepare decoder initial hidden/cell
def prepare_decoder_init(h_n, c_n, hp, proj_h=None, proj_c=None):
    """
    h_n/c_n from encoder: shape (num_layers * num_dirs, batch, enc_hidden)
    Return h_0, c_0 for decoder: (num_layers, batch, dec_hidden)
    """
    if not hp["bidirectional_encoder"]:
        return h_n.contiguous(), c_n.contiguous()

    num_layers = hp["num_layers"]
    batch_size = h_n.size(1)
    enc_h = hp["hidden_size"]
    # reshape (num_layers, 2, batch, enc_h)
    h_v = h_n.view(num_layers, 2, batch_size, enc_h)
    c_v = c_n.view(num_layers, 2, batch_size, enc_h)
    h_cat = torch.cat([h_v[:,0,:,:], h_v[:,1,:,:]], dim=2)  # (num_layers, batch, enc_h*2)
    c_cat = torch.cat([c_v[:,0,:,:], c_v[:,1,:,:]], dim=2)

    if proj_h is not None and proj_c is not None:
        h_0 = proj_h(h_cat)
        c_0 = proj_c(c_cat)
    else:
        h_0 = h_cat
        c_0 = c_cat

    return h_0.contiguous(), c_0.contiguous()

# Cell 10: inference helpers
def infer_word(word: str, enc, dec, src_vocab, inv_tgt, hp, proj_h=None, proj_c=None, use_beam=False):
    ids = [src_vocab.get(ch, src_vocab["<unk>"]) for ch in list(word.lower())]
    device = hp["device"]
    src_tensor = torch.tensor([ids], dtype=torch.long, device=device)
    src_lens = torch.tensor([len(ids)], dtype=torch.long)

    with torch.no_grad():
        enc_out, (h_n, c_n) = enc(src_tensor, src_lens)
        h_0, c_0 = prepare_decoder_init(h_n, c_n, hp, proj_h=proj_h, proj_c=proj_c)
        hidden = (h_0, c_0)
        SOS, EOS = 1, 2
        input_t = torch.tensor([[SOS]], dtype=torch.long, device=device)
        pred_chars = []
        for _ in range(hp["max_target_len"]):
            out, hidden = dec(input_t, hidden)
            logits = out.squeeze(1)
            top1 = logits.argmax(dim=1).item()
            if top1 == EOS:
                break
            ch = inv_tgt.get(top1, "")
            if ch in ("<pad>", "<sos>", "<eos>"):
                continue
            if ch == "<unk>":
                pred_chars.append("�")
            else:
                pred_chars.append(ch)
            input_t = torch.tensor([[top1]], dtype=torch.long, device=device)
        return "".join(pred_chars)

def infer_sentence(sentence: str, enc, dec, src_vocab, inv_tgt, hp, proj_h=None, proj_c=None, use_beam=False):
    tokens = sentence.strip().split()
    return " ".join(infer_word(tok, enc, dec, src_vocab, inv_tgt, hp, proj_h=proj_h, proj_c=proj_c, use_beam=use_beam) for tok in tokens)

ckpt = torch.load(HP["save_path"], map_location=HP["device"])
enc.load_state_dict(ckpt["enc_state"])
dec.load_state_dict(ckpt["dec_state"])
if ckpt.get("proj_h_state") is not None and proj_h is not None:
    proj_h.load_state_dict(ckpt["proj_h_state"])
    proj_c.load_state_dict(ckpt["proj_c_state"])

st.title("Transliteration LSTM Demo")

# Input text area with a unique key
sentence_input = st.text_area("Enter sentence to transliterate:", key="input_sentence")

# Submit button with unique key
if st.button("Transliterate", key="submit_button"):
    if sentence_input.strip() == "":
        st.warning("Please enter some text to transliterate!")
    else:
        prediction = infer_sentence(
            sentence_input, 
            enc, 
            dec, 
            src_vocab, 
            inv_tgt, 
            HP, 
            proj_h=proj_h, 
            proj_c=proj_c
        )
        st.success("Prediction:")
        st.write(prediction)

