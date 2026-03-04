# Iterative Retrieval RAG — Walkthrough

This notebook implements an **Iterative Retrieval + Self-Reflection RAG agent** using LangGraph. Unlike standard RAG which retrieves once and generates, this agent evaluates its own answer and loops back to retrieve again with a refined query if the answer falls short.

---

## The Core Idea

Standard RAG has a fixed pipeline:

```
retrieve → generate → done
```

The problem: if the first retrieval pulls vague or off-topic chunks, the generated answer will be poor — and the system has no way to know or recover.

Iterative RAG adds a feedback loop:

```
retrieve → generate → reflect on answer
                            │
               ┌── YES: answer is good → END
               └── NO:  refine query → retrieve again → (repeat)
```

The agent behaves like a researcher who reads a draft, decides it's not complete enough, and goes back to search with a more targeted query.

---

## Architecture

```
START
  │
  ▼
retrieve_docs
  │
  ▼
generate_answer
  │
  ▼
reflect_on_answer
  │
  ├── verified=True  OR  attempts ≥ 2  ──► END
  │
  └── verified=False ──► refine_query ──► retrieve_docs (loop)
```

**Loop guard:** `attempts >= 2` — prevents infinite cycles when the index simply doesn't contain an answer.

---

## State Schema

```python
class IterativeRAGState(BaseModel):
    question:         str           # original user question, never changed
    refined_question: str = ""      # rewritten query used in subsequent loops
    retrieved_docs:   List[Document] = []
    answer:           str = ""
    verified:         bool = False  # True = reflector approved the answer
    attempts:         int = 0       # incremented in generate_answer
```

The `refined_question` field is key — when the loop fires, `refine_query` writes a new query here. The `retrieve_docs` node checks this field first: if it's non-empty it uses the refined query, otherwise it falls back to the original `question`. This means the original question is preserved throughout.

---

## Document Store

```python
docs   = TextLoader("internal_docs.txt", encoding="utf-8").load()
chunks = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
           .split_documents(docs)
vectorstore = FAISS.from_documents(chunks, OpenAIEmbeddings())
retriever   = vectorstore.as_retriever()
```

A plain text file is loaded, split into 500-character chunks with 50-character overlap, and embedded into a FAISS index using OpenAI embeddings. The retriever uses default top-K similarity search.

---

## Node-by-Node Description

### Node 1: `retrieve_docs`

Checks `state.refined_question` — uses it if set, otherwise uses `state.question`. Invokes the FAISS retriever and stores results in `state.retrieved_docs`.

```python
query = state.refined_question or state.question
docs  = retriever.invoke(query)
```

On the first pass this uses the original question. On any retry loop it uses the LLM-refined query.

---

### Node 2: `generate_answer`

Concatenates all retrieved doc contents into a single context string, then calls GPT-4o with a straightforward prompt:

```
Use the following context to answer the question:
Context: {context}
Question: {state.question}
```

Increments `state.attempts` by 1 each time it runs.

---

### Node 3: `reflect_on_answer`

Sends the question and generated answer back to GPT-4o with a stricter evaluation prompt:

```
Evaluate whether the answer below is factually sufficient and complete.
Question: {question}
Answer: {answer}
Respond 'YES' if it's complete, otherwise 'NO' with feedback.
```

Parses the response: if `"yes"` appears in the lowercase output, `state.verified = True`. Otherwise `state.verified = False` and the feedback is available to the `refine_query` node.

---

### Node 4: `refine_query`

Only reached when `verified=False` and `attempts < 2`. Asks GPT-4o to rewrite the query:

```
The answer appears incomplete. Suggest a better version of the query 
that would help retrieve more relevant context.

Original Question: {question}
Current Answer: {answer}
```

The new query is stored in `state.refined_question` and the loop returns to `retrieve_docs`.

---

### Routing Logic (`should_continue`)

```python
lambda s: END if s.verified or s.attempts >= 2 else "refine"
```

Three exit conditions:
- Reflector approved the answer (`verified=True`) → END
- Maximum attempts reached (`attempts >= 2`) → END (returns best available answer)
- Otherwise → `refine_query` → loop back

---

## End-to-End Example

**Query:**
```
"agent loops and transformer-based systems?"
```

---

**Attempt 1 — `retrieve_docs`:**

Uses original query. Retrieves 4 chunks from `internal_docs.txt` about general agent architectures and transformer attention mechanisms.

---

**Attempt 1 — `generate_answer`:**

```
Agent loops are control structures used in LLM-based systems where the 
model iteratively refines its output or takes sequential actions. 
Transformer-based systems use self-attention to process sequences...
[attempts = 1]
```

---

**Attempt 1 — `reflect_on_answer`:**

```
NO — The answer explains agent loops and transformers separately 
but does not explain how they interact or why transformers are 
specifically used in agentic loop architectures.
[verified = False]
```

---

**`refine_query`:**

```
Rewritten query: "how transformer architecture enables agent loop 
reasoning and tool use in LLM systems"
```

---

**Attempt 2 — `retrieve_docs`:**

Uses the refined query. Retrieves 4 new chunks with more specific content about ReAct-style reasoning loops and attention in tool-calling agents.

---

**Attempt 2 — `generate_answer`:**

```
Transformer-based systems power agent loops through their self-attention 
mechanism, which allows the model to maintain context across multiple 
reasoning steps. In ReAct-style agents, the transformer processes the 
full history of actions and observations at each loop iteration, 
enabling coherent multi-step reasoning without external memory...
[attempts = 2]
```

---

**Attempt 2 — `reflect_on_answer`:**

```
YES — The answer explains the relationship between transformers and 
agent loops, references specific mechanisms (self-attention, context 
window), and covers the multi-step reasoning aspect.
[verified = True]
```

---

**Graph exits to END.**

**Final state:**
```python
result["answer"]    # the approved paragraph above
result["verified"]  # True
result["attempts"]  # 2
```

---

## Key Design Decisions

**Why preserve `original_question` separately from `refined_question`?**
The `generate_answer` node always answers `state.question` (the original). Only retrieval uses the refined version. This prevents the agent from gradually drifting away from user intent across loop iterations.

**Why cap at `attempts >= 2`?**
Two attempts is sufficient for most refinement cases. Beyond that, the index likely doesn't contain a better answer and further looping wastes tokens without improvement.

**Why check `"yes"` in lowercase output rather than exact match?**
GPT-4o occasionally wraps the verdict in additional text. A substring check is more robust than an exact string comparison.
