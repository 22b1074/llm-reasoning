# Hindi-English Transliteration

> **CS772 Assignment 2** | Converting Romanized Hindi to Devanagari Script

---

## 📌 Problem Statement

- **Task:** Character-level transliteration from Latin script to Devanagari
- **Input:** Romanized Hindi words (e.g., `namaste`, `khana`, `khayenge`)
- **Output:** Devanagari script (e.g., `नमस्ते`, `खाना`, `खायेंगे`)

---

## 📁 Dataset

- **Source:** AI4Bharat/Aksharantar
- **Training Size:** 100,000 samples (stratified from full dataset)

### Stratified Sampling Strategy
- Used 3-factor stratification: **source + native word length bin + english word length bin**
- Word length bins: `0-5`, `6-10`, `11-15`, `16-20`, `20+` characters
- Preserved original distribution with <0.01% deviation across all strata
- Fixed random seed (42) for reproducibility

### Source Distribution
| Source | Percentage |
|--------|:----------:|
| IndicCorp | 73.60% |
| Samanantar | 11.76% |
| Existing | 10.14% |
| Dakshina | 1.90% |
| Wikidata | 1.89% |
| AK-Freq | 0.71% |

---

## 🏗️ Models Implemented

### 1. BiLSTM Encoder-Decoder
- Bidirectional LSTM encoder to capture context from both directions
- Unidirectional LSTM decoder with hidden state projection
- 2 layers (max constraint)
- Teacher forcing ratio of 0.5 during training
- Dropout of 0.4 for regularization

### 2. Transformer
- Standard encoder-decoder architecture
- 2 encoder layers + 2 decoder layers (max constraint)
- 8 attention heads with 512 model dimension
- Sinusoidal positional encoding
- Label smoothing (0.1) for better generalization
- Warmup learning rate scheduler (4000 steps)

### 3. LLM (Gemini 2.5)
- Few-shot prompting with 10 example pairs
- Explicit phonetic character mappings provided in prompt
- Tested across multiple temperature and top_p configurations
- Best results at temperature=0.5, top_p=0.95

---

## 📊 Results

### Final Test Set Performance

| Model | Word Accuracy | Character F1 | Levenshtein Similarity | CER |
|-------|:-------------:|:------------:|:----------------------:|:---:|
| BiLSTM | 41.45% | 89.35% | 0.8445 | 0.1549 |
| Transformer | 46.58% | 89.91% | 0.8580 | 0.1423 |
| **Gemini 2.5** | **75.00%** | **91.00%** | - | - |

### Decoding Strategy Comparison

**Transformer (N=1280 samples):**
| Method | Word Acc | Char Match | CER | Char F1 |
|--------|:--------:|:----------:|:---:|:-------:|
| Greedy | 45.70% | 76.22% | 0.1421 | 0.8906 |
| Beam Search (k=5) | **46.33%** | **76.37%** | **0.1294** | **0.8922** |

**BiLSTM (N=1250 samples):**
| Method | Word Acc | Char Match | CER | Char F1 |
|--------|:--------:|:----------:|:---:|:-------:|
| Greedy | **38.32%** | **74.78%** | **0.1549** | **0.8879** |
| Beam Search (k=5) | 35.92% | 67.40% | 0.2344 | 0.8045 |

### Key Observations
- **Beam search helps Transformer** — explores better candidates for ambiguous sequences
- **Beam search hurts LSTM** — propagates low-probability errors across recurrent steps
- **LLM dominates** — leverages pre-trained linguistic knowledge and explicit phonetic rules

---

## 🔍 Error Analysis

### Common Failure Patterns
- **Short vs Long Vowels:** `i` ambiguous between `ि` and `ी`
- **Nasal Markers:** `◌ं` vs `◌ँ` highly context-dependent
- **Retroflex Consonants:** `t` → `त` vs `ट` based on word origin
- **Conjunct Consonants:** Complex clusters like `क्ष`, `ज्ञ` often mishandled

### Most Confused Characters
- `ा` (aa matra) — over-predicted due to high frequency
- `र` (ra) — multiple positional forms
- `◌ं` (anusvara) — nasal sound ambiguity
- Vowel matras confused due to frequency bias toward vowels

---

## 📈 Training Details

### BiLSTM
- Epochs: 16
- Batch size: 256
- Optimizer: Adam (lr=1e-3)
- Early stopping with patience

### Transformer
- Epochs: 20
- Batch size: 128
- Optimizer: Adam (lr=5e-4, betas=0.9/0.98)
- Warmup scheduler (4000 steps)
- Gradient clipping (max_norm=1.0)
- Early stopping (patience=5)

---

## 📏 Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **Word Accuracy** | Exact match between prediction and ground truth |
| **Character F1** | Harmonic mean of char-level precision and recall |
| **Levenshtein Similarity** | `1 - (edit_distance / max_length)` |
| **CER** | Character Error Rate = `edit_distance / reference_length` |
| **Char Match Rate** | Position-wise matching characters / reference length |

---

## 🧠 Key Learnings

- Stratified sampling crucial for maintaining data distribution
- Transformer attention handles long-range dependencies better than LSTM
- Greedy decoding sufficient for well-trained models; beam search marginal gains
- LLMs with proper prompting outperform task-specific models significantly
- Character-level F1 remains high (~89-91%) even when word accuracy is moderate — most errors are minor

---
## 🎮 Demo

- Built interactive **Streamlit web app** for real-time transliteration
- For both **BiLSTM** and **Transformer** model
- Users can input Romanized Hindi text and get instant Devanagari output

<p align="center"><b>IIT Bombay | CS772 Deep Learning for NLP</b></p>
