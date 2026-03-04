# Self-Reflection RAG Agent — Walkthrough

This notebook implements a **production-quality Self-Reflection RAG agent** using LangGraph. It is a cleaned and upgraded version of an earlier draft — the key improvements are structured `ChatPromptTemplate` prompts, dual-destination logging, proper error handling with `try/except`, and a configurable `MAX_ATTEMPTS` loop guard.

---

## The Core Idea

A self-reflection agent generates an answer, then evaluates that answer against the source documents before returning it. If the answer doesn't pass the evaluation, the query is rewritten and the pipeline runs again.

```
retrieve → generate answer → reflect (YES/NO)
                                  │
              ┌─── YES  ──────────┘  → END
              └─── NO   → rewrite query → retrieve again → (repeat)
```

This gives the agent one automatic self-correction pass before the user sees a result, which catches cases where the first retrieval pulled the wrong chunks.

---

## Architecture

```
START
  │
  ▼
retriever
  │
  ▼
responder
  │
  ▼
reflector
  │
  ├── revised=False  OR  attempts ≥ MAX_ATTEMPTS  ──► END
  │
  └── revised=True ──► query_rewriter ──► retriever (loop)
```

**`MAX_ATTEMPTS = 2`** — configured at the top of the notebook as a named constant, not a magic number buried in the routing function.

---

## Production Improvements Over the Draft Version

| Area | Before | After |
|---|---|---|
| Prompts | Raw f-strings | `ChatPromptTemplate.from_messages()` |
| Logging | `print()` statements | `logging` to file + console (`rag_agent.log`) |
| Error handling | None | `try/except` on every LLM call with fallback |
| Loop guard | Magic number inline | `MAX_ATTEMPTS = 2` config variable |
| `load_dotenv()` | Imported but never called | Called at top of config cell |
| Reflection judgment | Exact string match | Lowercase `"reflection: yes" in result` |

---

## State Schema

```python
class RAGReflectionState(BaseModel):
    original_question: str          # never mutated — used in reflector
    question:          str          # current query (may be rewritten)
    retrieved_docs:    List[Document] = []
    answer:            str = ""
    reflection:        str = ""     # full reflector output (verdict + explanation)
    revised:           bool = False # True = reflector rejected the answer
    attempts:          int = 0
    next_step:         str = ""
    context:           str = ""     # stored context for downstream use
```

`original_question` is immutable throughout the pipeline. The `reflector` node always evaluates the answer against `original_question`, not `question`. This ensures that even if the query gets rewritten across loop iterations, the final judgment is always against what the user actually asked.

---

## Document Store

```python
pdf_paths = [
    "./pdf/Earnings-Presentation-Q3-2025-Final.pdf",
    "./pdf/ICIBM 2025_submission_43_paper_v2.pdf",
]
```

Two PDFs are loaded with `PyPDFLoader`. Pages are split into 1000-character chunks with 100-character overlap and embedded using `text-embedding-3-small`. The retriever returns the top 5 most similar chunks (`k=5`).

The notebook includes defensive guards — it logs a warning if a file path doesn't exist, raises a `ValueError` if no documents are loaded, and raises again if splitting produces zero chunks (which happens with scanned/image PDFs).

---

## Prompts

All prompts use `ChatPromptTemplate.from_messages()` with a system/human turn structure.

**Answer prompt:**
```
system: "You are a helpful assistant. Use the provided context to 
         answer the user's question accurately."
human:  "Context: {context}\n\nQuestion: {question}"
```

**Reflection prompt:**
```
system: "You are evaluating an answer against the source documents. 
         Only use the provided context as the source of truth — do 
         not question whether the data is real or current.
         Respond strictly:
         Reflection: YES or NO
         Explanation: <your reasoning>"
human:  "Context: {context}\n\nQuestion: {question}\n\nAnswer: {answer}"
```

The instruction *"do not question whether the data is real or current"* is important for financial or academic PDFs — without it, GPT-4o may flag earnings figures as potentially outdated rather than evaluating whether the answer matches the document.

**Rewrite prompt:**
```
system: "You are refining a search query to retrieve better documents. 
         Given the original question and why the previous answer was 
         insufficient, rewrite the query to be more specific."
human:  "Original question: {question}\n\nWhy it failed: {reflection}
         \n\nRewritten query:"
```

---

## Node-by-Node Description

### Node 1: `retrieve_docs`

Invokes the FAISS retriever using `state.question` (which may be the original or a rewritten version on loop iterations). Stores the list of returned `Document` objects in `state.retrieved_docs`.

Logs the number of retrieved documents at INFO level. If retrieval fails, the exception propagates (no silent failure — this is intentional, as a retrieval error likely indicates a configuration problem worth surfacing).

---

### Node 2: `generate_answer` (responder)

Concatenates all retrieved doc page contents, runs the answer chain, and increments `state.attempts`:

```python
chain  = answer_prompt | llm | StrOutputParser()
answer = chain.invoke({"context": context, "question": state.question})
return state.model_copy(update={"answer": answer, "attempts": state.attempts + 1})
```

On failure the `except` block logs the error and sets `answer = "Error generating answer."` so the pipeline can continue rather than crash. The reflector will correctly flag this as needing revision.

---

### Node 3: `reflect_on_answer` (reflector)

Uses `state.original_question` (not the possibly-rewritten `state.question`) as the evaluation baseline. Parses the LLM's structured response:

```python
is_ok   = "reflection: yes" in result.lower()
revised = not is_ok
```

The full reflection text (verdict + explanation) is stored in `state.reflection`. This text is used by `query_rewriter` in the next step — it tells the rewriter *why* the answer failed, not just that it did.

On LLM failure: logs the error, sets `is_ok = True` (fail-safe: approve the answer rather than loop infinitely on an API error).

---

### Node 4: `query_rewriter`

Sends the original question and the reflection explanation to GPT-4o:

```python
chain.invoke({
    "question":   state.question,
    "reflection": state.reflection
})
```

The rewritten query is stored in `state.question`. Note this overwrites the working query, not `original_question`. On failure: logs the error and keeps `state.question` unchanged (falls back to the original, preventing an empty or broken rewrite from crashing the next retrieval).

---

### Routing: `should_continue`

```python
def should_continue(state) -> str:
    if not state.revised:         return END   # approved
    if state.attempts >= MAX_ATTEMPTS: return END   # loop guard
    return "query_rewriter"                   # retry
```

The routing result is used in a `add_conditional_edges` call with an explicit `path_map=["query_rewriter", END]` so LangGraph can render the full graph topology before execution.

---

## End-to-End Example

**Query:**
```
"What is the Meta Q3 earnings?"
```

---

**`retrieve_docs` (attempt 1):**

FAISS retrieves 5 chunks from `Earnings-Presentation-Q3-2025-Final.pdf`:
```
Chunk 1: "Total revenue for Q3 2025 was $40.6 billion, a 19% 
          increase year-over-year..."
Chunk 2: "Family Daily Active People (DAP) reached 3.29 billion 
          on average in September 2025..."
Chunk 3: "Operating income was $17.3 billion, representing an 
          operating margin of 42%..."
```

---

**`generate_answer` (attempt 1):**

```
Meta reported Q3 2025 revenue of $40.6 billion, up 19% year-over-year. 
Operating income reached $17.3 billion with a 42% operating margin. 
Daily Active People across the Family of Apps averaged 3.29 billion.
[attempts = 1]
```

---

**`reflect_on_answer` (attempt 1):**

```
Reflection: YES
Explanation: The answer correctly cites revenue ($40.6B), growth rate 
(19%), operating income ($17.3B), margin (42%), and DAP (3.29B) — all 
figures are present and consistent with the provided context.
[revised = False]
```

Graph routes to **END**.

---

**Final output:**
```python
result["answer"]     # "Meta reported Q3 2025 revenue of $40.6 billion..."
result["reflection"] # "Reflection: YES\nExplanation: ..."
result["attempts"]   # 1
```

---

### What a Failed Reflection Looks Like

Suppose the query was instead *"What drove Meta's revenue growth in Q3?"* and the retriever only pulled high-level summary pages without the segment breakdown:

```
Reflection: NO
Explanation: The answer states revenue grew 19% but does not 
explain what drove that growth. The context contains information 
about advertising revenue contribution by region but the answer 
did not reference it.
[revised = True]
```

`query_rewriter` would then produce:
```
"Meta Q3 2025 advertising revenue breakdown by region and growth drivers"
```

The loop returns to `retrieve_docs` with this more targeted query, pulling the regional breakdown chunks, and a more complete answer is generated on attempt 2.

---

## Key Design Decisions

**Why `ChatPromptTemplate` instead of f-strings?**
`ChatPromptTemplate` enforces the system/human message structure that GPT-4o expects for reliable instruction-following. Raw f-strings collapse everything into a single user turn, which degrades instruction adherence on structured-output prompts (like the `Reflection: YES or NO` format).

**Why does `reflector` use `original_question` but `query_rewriter` uses `question`?**
Reflection judges completeness against user intent (original). Rewriting needs to know where the query currently is to make a useful improvement — rewriting from scratch every time ignores context gained from the first attempt.

**Why is `MAX_ATTEMPTS = 2` the right default?**
One retry is almost always sufficient. If the rewritten query still doesn't retrieve better documents on attempt 2, the document set likely doesn't contain a better answer. Adding more attempts mostly increases latency and API cost without meaningfully improving output quality.

**Why log to both file and console?**
Console output lets you monitor progress during notebook execution. The file (`rag_agent.log`) gives you a permanent record to debug issues after the fact — especially useful when running the pipeline on larger document sets where a single run can take several minutes.
