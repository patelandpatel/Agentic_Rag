Sure! Let's walk through the entire pipeline with this query:

> **"How did Meta's Q3 2025 tax charge affect their earnings per share, and what does academic research say about the impact of one-time tax charges on stock prices?"**

This is a perfect example because it's complex enough to trigger every single node.

---

## Cell 8 — Entry Point

```python
USER_QUERY = "How did Meta's Q3 2025 tax charge affect their earnings per share, \
and what does academic research say about the impact of one-time tax charges on stock prices?"

init_state = AgenticRAGState(
    question=USER_QUERY,
    original_question=USER_QUERY,  # preserved forever — never overwritten
)
result = graph.invoke(init_state)
```

State at this point:
```
question          = "How did Meta's Q3 2025 tax charge..."
original_question = "How did Meta's Q3 2025 tax charge..."
sub_queries       = []
retrieved_docs    = []
attempts          = 0
```

---

## Node 1 — `query_planner` 🟡

The question has **two distinct parts** — one about Meta's financials, one about academic research. The LLM detects this and decomposes it.

```
INPUT:  "How did Meta's Q3 2025 tax charge affect their EPS, 
         and what does academic research say about one-time tax charges?"

LLM OUTPUT (JSON):
[
  "How did Meta's Q3 2025 one-time tax charge affect diluted EPS?",
  "What does academic research say about the stock price impact of one-time tax charges?"
]
```

Log:
```
[query_planner] Decomposing: 'How did Meta's Q3 2025 tax charge...'
[query_planner] 2 sub-queries: ['How did Meta's Q3 2025...', 'What does academic research...']
```

State update:
```
sub_queries = [
    "How did Meta's Q3 2025 one-time tax charge affect diluted EPS?",
    "What does academic research say about the stock price impact of one-time tax charges?"
]
```

---

## Node 2 — `faiss_retriever`

Takes the **first sub-query** and searches the local FAISS index (your PDF about Meta's Q3 earnings).

```
QUERY:  "How did Meta's Q3 2025 one-time tax charge affect diluted EPS?"

RETRIEVED DOCS (5 chunks from the earnings PDF):
  - Chunk 1: "...net income of $2.71 billion included a $15.93B non-cash tax charge..."
  - Chunk 2: "...diluted EPS reported at $1.05; excluding the charge, EPS was $7.25..."
  - Chunk 3: "...One Big Beautiful Bill Act implementation triggered the charge..."
  - Chunk 4: "...Q3 revenue reached $42.3 billion, up 19% year-over-year..."
  - Chunk 5: "...management noted the charge was non-recurring in nature..."
```

Log:
```
[faiss_retriever] Querying FAISS: 'How did Meta's Q3 2025 one-time tax charge...'
[faiss_retriever] ✅ Retrieved 5 doc(s)
```

State update:
```
retrieved_docs    = [5 chunks from earnings PDF]
retrieval_source  = "faiss"
```

---

## Node 3 — `iterative_retrieval_check` 🟢

Checks if the retrieved docs are actually relevant **before wasting tokens on generation**.

**Stage 1 — Fast cosine similarity check:**
```
Best similarity score = 0.12  (well below threshold of 0.30)
→ FAST PASS: docs are relevant, skip LLM call
```

Log:
```
[iterative_check] Checking 5 doc(s) for relevance
[iterative_check] ✅ Fast check passed (score=0.12)
```

State update:
```
docs_are_relevant = True
```

Since `docs_are_relevant = True`, the routing function `should_call_react` sends us **directly to `cot_answerer`**, skipping `react_router` entirely. The FAISS docs were good enough.

---

## Node 4 — `react_router` 🔴 (second iteration)

`react_router` is **skipped this time** because FAISS was relevant. But it WILL fire for the second sub-query about academic research. Let's fast-forward to that moment.

After `cot_answerer` finishes sub-query 1, the graph loops back for sub-query 2: *"What does academic research say about stock price impact of one-time tax charges?"*

Your local PDFs have **no academic papers** on this topic, so FAISS returns weak results, `iterative_retrieval_check` sets `docs_are_relevant = False`, and the router calls `react_router`.

```
INPUT TO LLM:
  Query:           "What does academic research say about stock price impact..."
  FAISS relevant:  False
  Previous source: faiss

LLM OUTPUT: "arxiv"
```

Log:
```
[react_router] Tool selected: 'arxiv'
[arxiv] 🔍 Searching arXiv: 'one-time tax charges stock price impact earnings'
[arxiv] ✅ Retrieved 3 paper(s)
```

State update:
```
retrieved_docs    = [3 arXiv abstracts] + [5 weak FAISS docs]
retrieval_source  = "arxiv"
```

---

## Node 5 — `cot_answerer` 🩷

Now runs Chain of Thought for **both sub-queries**. The key thing here is the `<scratchpad>` — the LLM must reason step by step before writing the answer.

**Sub-query 1** (using FAISS docs):
```
<scratchpad>
Step 1: The context mentions a $15.93B non-cash tax charge.
Step 2: Reported net income was $2.71B, reported EPS was $1.05.
Step 3: Excluding the charge, net income = $18.64B, EPS = $7.25.
Step 4: The difference is $6.20 per share, purely from one charge.
Step 5: The charge was triggered by the One Big Beautiful Bill Act.
</scratchpad>

Answer: Meta's Q3 2025 diluted EPS was severely depressed by a $15.93B 
one-time non-cash tax charge. Reported EPS came in at $1.05, but excluding 
this charge the adjusted EPS would have been $7.25 — a $6.20 difference 
attributable entirely to this single non-recurring item.
```

**Sub-query 2** (using arXiv docs):
```
<scratchpad>
Step 1: The arXiv papers discuss event studies around tax-related announcements.
Step 2: One paper finds markets partially price in non-recurring charges...
Step 3: Another finds stock prices recover within 2 quarters when charge is non-cash...
</scratchpad>

Answer: Academic research suggests markets distinguish between cash and 
non-cash one-time charges. Non-cash charges like Meta's tend to cause 
short-term price pressure but recover as investors focus on operating 
earnings...
```

Log:
```
[cot_answerer] Answering 2 sub-queries (attempt 1)
[cot_answerer] Sub-Q 1/2 answered
[cot_answerer] Sub-Q 2/2 answered
```

State update:
```
sub_answers = [
    "Q: How did Meta's tax charge affect EPS?\nA: Reported EPS $1.05...",
    "Q: What does academic research say?\nA: Markets distinguish cash vs non-cash..."
]
cot_reasoning = "<scratchpad content for both sub-queries>"
attempts      = 1
```

---

## Node 6 — `answer_synthesiser` 🔵

Takes both sub-answers and merges them into one coherent response, removing repetition and resolving any contradictions.

```
INPUT:
  original_question: "How did Meta's Q3 2025 tax charge affect their EPS..."
  sub_answers: [answer 1, answer 2]

OUTPUT:
  Meta's Q3 2025 reported diluted EPS of $1.05 was significantly impacted 
  by a $15.93B one-time non-cash tax charge stemming from the One Big 
  Beautiful Bill Act. Excluding this charge, adjusted EPS would have been 
  $7.25 — a $6.20 per-share impact.

  Academic research supports the view that markets treat non-cash one-time 
  charges differently from cash charges. Event studies suggest short-term 
  price pressure followed by recovery as investors focus on operating 
  metrics rather than GAAP figures distorted by non-recurring items.

  📚 Sources used: ARXIV
```

Log:
```
[synthesiser] Synthesising 2 sub-answer(s)
[synthesiser] ✅ Final answer synthesised
```

---

## Node 7 — `self_reflector` 🟠

Judges the final answer against the **original question** with the document context:

```
Does the answer address: 
  ✅ How the tax charge affected EPS? → Yes, $1.05 vs $7.25 explained
  ✅ What academic research says?     → Yes, arXiv papers cited

Reflection: YES
Explanation: The answer addresses both the financial impact on EPS 
and the academic perspective on one-time tax charges.
```

Log:
```
[reflector] Reflecting on answer (attempt 1)
[reflector] ✅ APPROVED
[router] Answer approved → END
```

Since `needs_revision = False`, `should_continue` returns `END` — the pipeline stops.

---

## What Would Trigger `query_rewriter`

If the reflection had said NO — for example, if the synthesis missed the academic research part — `should_continue` would return `"query_rewriter"`. The rewriter would generate a more specific query like:

```
"Meta Q3 2025 EPS impact of $15.93B One Big Beautiful Bill Act tax charge 
 AND academic studies on non-cash one-time charges stock price recovery"
```

Then the graph loops back to `query_planner` with this new question, re-decomposes it, and tries again — up to `MAX_ATTEMPTS = 3`.

---

## Final Log Summary

```
[query_planner]      2 sub-queries planned
[faiss_retriever]    5 docs retrieved
[iterative_check]    ✅ Fast check passed (score=0.12)
[router]             Docs relevant → cot_answerer  ← react_router SKIPPED
[cot_answerer]       Sub-Q 1 answered (FAISS context)
                     → iterative_check: NOT RELEVANT for sub-Q 2
[react_router]       Tool selected: arxiv           ← FAISS insufficient
[arxiv]              3 papers retrieved
[cot_answerer]       Sub-Q 2 answered (arXiv context)
[synthesiser]        Final answer synthesised
[reflector]          ✅ APPROVED
[router]             END — 1 attempt total
```

This example exercised every single node at least once, and showed both the happy path (FAISS sufficient → skip ReAct) and the fallback path (FAISS insufficient → ReAct → arXiv).