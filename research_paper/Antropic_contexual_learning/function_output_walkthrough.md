# Function Output Walkthrough
### Using a single academic PDF as the example throughout

---

## The Example PDF

Imagine a 3-page academic paper called `transformer_attention.pdf`. Here is roughly what PyMuPDF sees when it reads the raw text off each page before any cleaning:

**Page 1 raw (before cleaning):**
```
Attention Is All You Need
Neural Information Processing Systems 2023
3.1 Model Architecture The proposed model follows an encoder-
decoder structure. We use multi-head attention with dk = 64, dv = 64.
Results show that the model achieves 94.3 BLEU on
the WMT benchmark.
```

**Page 2 raw (before cleaning):**
```
WMT benchmark results (Table 2):
Base 41.0 27.3 38.1
Large 94.3 89.1 91.2
the final layer applies a softmax function. The attention
weights are computed as follows: QK^T / sqrt(dk).
```

**Page 3 raw (before cleaning):**
```
References
Vaswani, A., Shazeer, N. (2017). Attention is all you
need. Advances in Neural Information Processing Systems.
Devlin, J. (2018). BERT. arXiv preprint.
```

---

## Function 1 — `extract_text_from_pdf`

**What it receives:** the file path `"transformer_attention.pdf"`

**What it does step by step:**

**Page 1:**
- Header `"Attention Is All You Need"` → Y position is in top 7% → **dropped**
- Header `"Neural Information Processing Systems 2023"` → Y position in top 7% → **dropped**
- Body block detected, lines collected:
  ```
  3.1 Model Architecture The proposed model follows an encoder-
  decoder structure. We use multi-head attention with dk = 64, dv = 64.
  Results show that the model achieves 94.3 BLEU on
  the WMT benchmark.
  ```
- Hyphenation fix: `"encoder-"` + `"decoder structure"` → `"encoder-decoder structure"`
- Stray newline fix: `"94.3 BLEU on"` ends with no punctuation, next line `"the WMT benchmark."` starts lowercase → joined with space
- Final page 1 text:
  ```
  3.1 Model Architecture The proposed model follows an encoder-decoder structure.
  We use multi-head attention with dk = 64, dv = 64.
  Results show that the model achieves 94.3 BLEU on the WMT benchmark.
  ```

**Page 2:**
- Block `"WMT benchmark results (Table 2):"` → passes filters, kept
- Block `"Base 41.0 27.3 38.1"` → matches `[\d\s\.\,\%\-]+` and length < 40 → **table noise, dropped**
- Block `"Large 94.3 89.1 91.2"` → same → **dropped**
- Remaining lines:
  ```
  the final layer applies a softmax function.
  The attention weights are computed as follows: QK^T / sqrt(dk).
  ```
- Stray newline check: `"WMT benchmark results (Table 2):"` ends with `:` (not `.!?`), next line `"the final layer..."` starts lowercase → **joined**
- Final page 2 text:
  ```
  WMT benchmark results (Table 2): the final layer applies a softmax function.
  The attention weights are computed as follows: QK^T / sqrt(dk).
  ```

**Page 3:**
- First block `"References"` → matches `REFERENCE_HEADINGS` → **flag set, all remaining blocks dropped**
- references_reached = True → page 3 produces nothing

**What `extract_text_from_pdf` returns:**

```python
[
    {
        "page_num": 1,
        "text": "3.1 Model Architecture The proposed model follows an encoder-decoder structure. We use multi-head attention with dk = 64, dv = 64. Results show that the model achieves 94.3 BLEU on the WMT benchmark."
    },
    {
        "page_num": 2,
        "text": "WMT benchmark results (Table 2): the final layer applies a softmax function. The attention weights are computed as follows: QK^T / sqrt(dk)."
    }
]
```

**Key things to notice:**
- It is a list of dicts — one dict per page
- Page 3 is completely absent — references discarded
- The table rows are gone — numeric noise filtered
- The page header is gone — Y position filtered
- `"encoder-decoder"` is one word — hyphenation fixed
- `"94.3 BLEU on the WMT benchmark."` is one line — newline joined

---

## Function 2 — `split_into_sentences`

**What it receives:** the joined string built inside `semantic_chunk_pdf` — all page texts concatenated with page markers between them:

```
"3.1 Model Architecture The proposed model follows an encoder-decoder structure. We use multi-head attention with dk = 64, dv = 64. Results show that the model achieves 94.3 BLEU on the WMT benchmark. |||PAGE_2||| WMT benchmark results (Table 2): the final layer applies a softmax function. The attention weights are computed as follows: QK^T / sqrt(dk)."
```

**What spaCy does with this:**

spaCy reads the full string as one document and identifies sentence boundaries using its dependency parser. It correctly handles:
- `"dk = 64, dv = 64."` — the decimal-like tokens are not sentence boundaries
- `"(Table 2):"` — the parenthetical is kept with its sentence
- `"QK^T / sqrt(dk)."` — the symbols are kept together
- The `|||PAGE_2|||` marker — treated as an unknown token, doesn't confuse the parser

**Raw output from spaCy (before marker recovery):**

```python
[
    "3.1 Model Architecture The proposed model follows an encoder-decoder structure.",
    "We use multi-head attention with dk = 64, dv = 64.",
    "Results show that the model achieves 94.3 BLEU on the WMT benchmark.",
    "|||PAGE_2||| WMT benchmark results (Table 2): the final layer applies a softmax function.",
    "The attention weights are computed as follows: QK^T / sqrt(dk)."
]
```

**What `split_into_sentences` returns:**

```python
[
    "3.1 Model Architecture The proposed model follows an encoder-decoder structure.",
    "We use multi-head attention with dk = 64, dv = 64.",
    "Results show that the model achieves 94.3 BLEU on the WMT benchmark.",
    "|||PAGE_2||| WMT benchmark results (Table 2): the final layer applies a softmax function.",
    "The attention weights are computed as follows: QK^T / sqrt(dk)."
]
```

**Key things to notice:**
- It is a flat list of strings — no page info yet, that comes next
- Markers are still present — they get stripped in `semantic_chunk_pdf`
- `"dk = 64, dv = 64."` stayed as one sentence — spaCy handled the decimals correctly
- The oversized fallback didn't fire — all sentences are short enough

---

## Function 3 — `semantic_chunk_pdf`

### Part A — Marker recovery (page references rebuilt)

The function walks through the sentence list and tracks which page each sentence belongs to, using the markers:

| Sentence | Contains marker? | Assigned page |
|---|---|---|
| `"3.1 Model Architecture..."` | No | Page 1 |
| `"We use multi-head attention..."` | No | Page 1 |
| `"Results show that the model..."` | No | Page 1 |
| `"|||PAGE_2||| WMT benchmark results..."` | Yes — PAGE_2 | Page 1 (it started on page 1, marker removed, current_page advances to 2) |
| `"The attention weights..."` | No | Page 2 |

After marker recovery, the clean sentences and page refs are:

```python
sentences = [
    "3.1 Model Architecture The proposed model follows an encoder-decoder structure.",
    "We use multi-head attention with dk = 64, dv = 64.",
    "Results show that the model achieves 94.3 BLEU on the WMT benchmark.",
    "WMT benchmark results (Table 2): the final layer applies a softmax function.",
    "The attention weights are computed as follows: QK^T / sqrt(dk)."
]

page_refs = [1, 1, 1, 1, 2]
```

---

### Part B — How chunking actually works (your detailed question)

This is the core of the function. The answer to "how many sentences go into one chunk" is: **as many as fit within `max_chunk_tokens = 512` tokens**, estimated by character count divided by 4.

Here is the process sentence by sentence for our example:

**`max_chunk_tokens = 512`, so the budget is 512 tokens ≈ 2048 characters**

---

**Iteration 1 — sentence added:**
```
current_sents = ["3.1 Model Architecture The proposed model follows an encoder-decoder structure."]
current_pages = [1]
token_est = 80 chars // 4 = 20 tokens
20 < 512 → keep going, do not flush yet
```

**Iteration 2 — sentence added:**
```
current_sents = [
    "3.1 Model Architecture The proposed model follows an encoder-decoder structure.",
    "We use multi-head attention with dk = 64, dv = 64."
]
current_pages = [1, 1]
token_est = (80 + 51) chars // 4 = 32 tokens
32 < 512 → keep going
```

**Iteration 3 — sentence added:**
```
current_sents = [
    "3.1 Model Architecture...",
    "We use multi-head attention...",
    "Results show that the model achieves 94.3 BLEU on the WMT benchmark."
]
current_pages = [1, 1, 1]
token_est = (80 + 51 + 69) chars // 4 = 50 tokens
50 < 512 → keep going
```

*(In this small example, all 5 sentences together are only ~200 characters ≈ 50 tokens, so the budget is never hit and everything ends up in one chunk. In a real paper, sentences fill up the budget and chunks get flushed. Here is what a flush looks like when it does happen:)*

---

**What a flush looks like in a larger real example:**

Suppose after adding the 28th sentence the token estimate crosses 512:

```
token_est = 530 tokens  ← over budget
→ save_chunk() is called
→ chunk 0 is saved with sentences 1–28

overlap kicks in:
current_sents = [sentence_27, sentence_28]   ← last 2 sentences carried over
current_pages = [page of s27, page of s28]

→ next chunk starts fresh from sentence 27
→ sentence 29 is added to current_sents next iteration
```

This is the overlap mechanism. Sentences 27 and 28 appear in **both** chunk 0 and chunk 1. This means if a thought spans a chunk boundary, the context is not completely lost — the next chunk has a running start.

---

**What `semantic_chunk_pdf` returns for our small example:**

Because this PDF only has 5 short sentences totalling ~50 tokens, one chunk is produced:

```python
[
    {
        "chunk_id": 0,
        "text": "3.1 Model Architecture The proposed model follows an encoder-decoder structure. We use multi-head attention with dk = 64, dv = 64. Results show that the model achieves 94.3 BLEU on the WMT benchmark. WMT benchmark results (Table 2): the final layer applies a softmax function. The attention weights are computed as follows: QK^T / sqrt(dk).",
        "page_start": 1,
        "page_end": 2,
        "sentences": 5
    }
]
```

In a real 10-page paper, you would typically see 15–30 chunks depending on sentence density.

---

## End-to-End Summary

```
transformer_attention.pdf
        │
        ▼
extract_text_from_pdf()
        │  returns: list of {page_num, text} — 2 pages, cleaned
        │
        ▼
[joined into one string with |||PAGE_N||| markers]
        │
        ▼
split_into_sentences()
        │  returns: flat list of sentence strings (markers still present)
        │
        ▼
[marker recovery loop — strips markers, rebuilds page_refs]
        │
        ▼
[chunking loop — fills budget, flushes, overlaps]
        │
        ▼
semantic_chunk_pdf()
        returns: list of {chunk_id, text, page_start, page_end, sentences}
```
