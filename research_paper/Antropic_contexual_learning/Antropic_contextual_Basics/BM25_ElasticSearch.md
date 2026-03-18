Great! So think of Elasticsearch as a **search engine built on top of BM25** — it handles all the messy infrastructure so you don't have to implement BM25 yourself.

Here's how to think about it in layers:

---

**What Elasticsearch gives you on top of raw BM25:**

**1. Indexing**
When you add documents, Elasticsearch doesn't store them as plain text. It builds an **inverted index** behind the scenes:

```
"cat"   → [chunk_1, chunk_4, chunk_7]
"black" → [chunk_1, chunk_3]
"dog"   → [chunk_2, chunk_4]
```
So when a query comes in, it instantly knows which chunks to look at — no scanning every document.

**2. Analyzer / Tokenizer**
Before indexing, Elasticsearch processes text through an **analyzer pipeline:**
```
"The black cats are running"
        ↓
Remove stopwords → "black cats running"
        ↓
Stemming → "black cat run"      ← notice "cats"→"cat", "running"→"run"
        ↓
Lowercase → "black cat run"
```
In your code this is set to `"english"` analyzer — which handles English stemming and stopwords automatically.

**3. Scoring with BM25**
Once it knows which chunks match, it applies BM25 scoring with length normalization and saturation. 

**3.1 Term Frequency Saturation**
In TF-IDF, if a word appears 10x it scores 10x more than appearing 1x. BM25 says — after a word appears a few times, extra repetitions stop mattering as much.

```
TF-IDF:  "cat" appears 1x → score 1,  10x → score 10
BM25:    "cat" appears 1x → score 1,  10x → score ~2.5  (diminishing returns)
```

**3.2 Document Length Normalization**
A long chunk has more words, so naturally any word appears more. BM25 penalizes longer chunks so they don't unfairly dominate.

```
Chunk A (50 words):  "cat" appears 2x  → high score
Chunk B (500 words): "cat" appears 2x  → lower score (diluted by length)
```

**4. Multi-field search**
Raw BM25 works on one field. Elasticsearch lets you search across **multiple fields simultaneously** — in your code it searches both `content` and `contextualized_content` fields at the same time and merges the scores.

---

**The simple mental model:**

```
Your query
    ↓
Elasticsearch analyzer (stem, lowercase, remove stopwords)
    ↓
Inverted index lookup (which chunks have these words?)
    ↓
BM25 scoring (rank those chunks)
    ↓
Returns top-k ranked results
```
