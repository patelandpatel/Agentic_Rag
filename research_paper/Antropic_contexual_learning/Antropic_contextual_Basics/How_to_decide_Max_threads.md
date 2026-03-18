## Deciding `max_threads`

---

### Your Core Insight is Correct

```
Large document → more chunks → more API calls with same document in cache
                             → cache hit rate is HIGH
                             → fewer threads = more sequential = better cache usage
                             → save money, pay with time

Small document → fewer chunks → less benefit from caching same document
                              → cache expires before you even finish the doc
                              → more threads = faster, cache barely helps anyway
                              → save time, minimal cache penalty
```

---

### Is Large Document = More Chunks Always True?

**Generally yes, but not always.** It depends on your `DocumentProcessor` settings:

```
Large document (50 pages)
    → more sentences
    → more coarse blocks
    → more semantic splits
    → MORE chunks ✅ (usually)

Small document (3 pages)
    → fewer sentences
    → fewer blocks
    → FEWER chunks ✅ (usually)
```

BUT exceptions exist:

```
Large document, very uniform text     → fewer semantic split points → fewer chunks
Small document, highly varied topics  → many semantic splits → more chunks than expected
```

So document size is a **strong proxy** for chunk count, not a guarantee.

---

### The Real Tradeoff Matrix

| Document Size | Chunk Count | Best `max_threads` | Why |
|---|---|---|---|
| Large (8k+ tokens) | Many (20+) | 1–2 | Cache stays warm per doc, high savings |
| Medium (3-8k tokens) | Medium (8-15) | 2–4 | Balanced |
| Small (<3k tokens) | Few (2-5) | 5–10 | Cache barely helps, parallelism wins |
| Mixed sizes | Mixed | 2–3 | Safe middle ground |

---

### The Cache Hit Timing Problem With Threads

This is the deeper issue your intuition is pointing at:

```
max_threads=1, doc with 20 chunks:

t=0:00  chunk1  → writes doc to cache
t=0:15  chunk2  → reads from cache ✅
t=0:30  chunk3  → reads from cache ✅
t=0:45  chunk4  → reads from cache ✅
... all 20 chunks hit cache → ~90% savings


max_threads=5, doc with 20 chunks:

t=0:00  chunk1, chunk2, chunk3, chunk4, chunk5 → ALL try to write cache simultaneously
        → 5x cache_creation charges instead of 1 ❌
t=0:01  chunk6-10 → some read cache ✅, some write again ❌ (race condition)
... unpredictable, savings much lower
```

With multiple threads, multiple chunks from the **same document** can start simultaneously before the cache is written, causing **multiple cache creation charges** for the same document.

---

### Practical Decision Rule

```
Calculate: chunks_per_document = total_chunks / total_documents

chunks_per_doc > 15  →  max_threads = 1 or 2
chunks_per_doc 8-15  →  max_threads = 2 to 4  
chunks_per_doc < 8   →  max_threads = 4 to 8
```

The fundamental rule: **threads and cache efficiency are inversely proportional**. You're always trading one for the other, and the size of your documents determines which side of that tradeoff is worth paying for.