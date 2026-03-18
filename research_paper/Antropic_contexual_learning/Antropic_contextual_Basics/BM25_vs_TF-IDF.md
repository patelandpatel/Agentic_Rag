**BM25:**
- BM25 does use term frequency (how often a word appears in a chunk)
- It does penalize common/dominant words

**What's slightly off:**
- BM25 doesn't use TF-IDF directly — it's a **separate algorithm inspired by TF-IDF** but with two important improvements

---

**The 2 key things BM25 adds over TF-IDF:**

**1. Term Frequency Saturation**
In TF-IDF, if a word appears 10x it scores 10x more than appearing 1x. BM25 says — after a word appears a few times, extra repetitions stop mattering as much.

```
TF-IDF:  "cat" appears 1x → score 1,  10x → score 10
BM25:    "cat" appears 1x → score 1,  10x → score ~2.5  (diminishing returns)
```

**2. Document Length Normalization**
A long chunk has more words, so naturally any word appears more. BM25 penalizes longer chunks so they don't unfairly dominate.

```
Chunk A (50 words):  "cat" appears 2x  → high score
Chunk B (500 words): "cat" appears 2x  → lower score (diluted by length)
```

---

**Quick example — query: "black cat"**

| Chunk | Content | BM25 score |
|---|---|---|
| A | "The black cat sat on the mat" | High — both words, short doc |
| B | "The cat cat cat cat cat cat" | Medium — "cat" saturates |
| C | "A black bear and a black dog and a black cat were seen in a very long wildlife report..." | Lower — words present but doc is long |
| D | "The weather is nice today" | Zero — no matching words |

---

Does this clear it up? Once you confirm, we can go into how Elasticsearch wraps BM25 and then walk through the code.