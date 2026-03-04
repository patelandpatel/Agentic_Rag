# Query Planning & Decomposition RAG — Walkthrough

This notebook implements a **Query Decomposition RAG agent** using LangGraph. Instead of sending the user's question directly to retrieval, a planner node first breaks it into 2–3 focused sub-questions, retrieves documents for each independently, then generates a unified final answer from all combined context.

---

## The Core Idea

Standard RAG sends the raw question to the retriever:

```
question → retrieve → generate → done
```

The problem: a question with multiple distinct parts (e.g. *"Explain agent loops AND video diffusion challenges"*) will retrieve chunks that partially address each topic but may miss important detail on either one.

Query decomposition solves this by treating each sub-topic as its own retrieval task:

```
question → plan (split into sub-questions) → retrieve per sub-question → generate unified answer
```

This is like assigning each part of a compound question to a separate research task before writing the final report.

---

## Architecture

```
START
  │
  ▼
planner          ← splits question into 2–3 sub-questions
  │
  ▼
retriever        ← runs FAISS search for EACH sub-question, merges all docs
  │
  ▼
responder        ← generates one final answer from all combined context
  │
  ▼
END
```

This is a **linear 3-node graph** — no loops, no branching. The complexity lives in the retriever node, which runs multiple queries rather than one.

---

## State Schema

```python
class RAGState(BaseModel):
    question:       str              # original user question
    sub_questions:  List[str] = []   # decomposed by planner node
    retrieved_docs: List[Document] = []  # merged docs from all sub-queries
    answer:         str = ""         # final synthesised answer
```

Each node returns a full new `RAGState` object rather than using `model_copy`. This is a simpler but slightly less memory-efficient pattern than the `model_copy(update=...)` approach — fine for a linear graph with no loop state management.

---

## Document Store

```python
urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2024-04-12-diffusion-video/",
]
docs   = [WebBaseLoader(url).load() for url in urls]  # two blog posts
chunks = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
           .split_documents(docs)
vectorstore = FAISS.from_documents(chunks, OpenAIEmbeddings())
retriever   = vectorstore.as_retriever()
```

Two Lilian Weng blog posts are loaded — one on LLM agents, one on diffusion video generation. These are split into 500-character chunks and embedded into FAISS. The example query is specifically designed to span both documents, which is where decomposition provides the most benefit.

---

## Node-by-Node Description

### Node 1: `planner`

Sends the original question to GPT-4o with a prompt instructing it to return 2–3 sub-questions as a numbered or bulleted list:

```
Break the following complex question into 2-3 sub-questions:
Question: {state.question}
Sub-questions:
```

The response is split on newlines and stripped of bullet/dash formatting characters. The result is stored in `state.sub_questions`.

**No JSON parsing is used here** — the output is treated as a raw text list. This is simpler but more fragile than JSON; an LLM that adds preamble text before the sub-questions could produce noise in the list.

---

### Node 2: `retriever`

Loops over every sub-question and calls the FAISS retriever for each:

```python
for sub in state.sub_questions:
    docs = retriever.invoke(sub)
    all_docs.extend(docs)
```

All results are merged into a single flat list without deduplication. If two sub-questions retrieve the same chunk, it appears twice in the context — something to be aware of for longer document sets.

---

### Node 3: `responder`

Concatenates all retrieved doc contents and sends the combined context + original question to GPT-4o:

```
Use the context below to answer the question.
Context: {all retrieved content}
Question: {original question}
```

Note that the original `state.question` is used here, not the individual sub-questions. The LLM synthesises a unified answer from the broader context pool.

---

## End-to-End Example

**Query:**
```
"Explain how agent loops work and what are the challenges in 
diffusion video generation?"
```

---

**Node 1 — `planner`:**

GPT-4o decomposes this into:
```
1. How do agent loops work in large language model systems?
2. What are the main challenges in diffusion-based video generation?
3. How does the iterative nature of agent loops relate to the 
   iterative denoising process in diffusion models?
```

`state.sub_questions` = list of 3 strings above.

---

**Node 2 — `retriever`:**

Runs 3 separate FAISS queries:

Sub-question 1 → retrieves chunks from the agent blog post:
```
Chunk A: "ReAct agent loops combine reasoning traces and actions. 
          At each step the model produces a Thought → Action → 
          Observation triple..."

Chunk B: "Planning in LLM agents can be decomposed into task 
          decomposition, reflection, and memory components..."
```

Sub-question 2 → retrieves chunks from the diffusion video blog post:
```
Chunk C: "Temporal consistency remains the primary challenge in 
          video diffusion — generating coherent motion across 
          frames requires the model to reason over long horizons..."

Chunk D: "Computational cost of video diffusion scales with the 
          number of frames. Generating 16 frames at 256x256 
          requires ~8x the FLOPs of a single image..."
```

Sub-question 3 → retrieves a mix from both posts:
```
Chunk E: "Both agent loops and diffusion processes are iterative 
          by nature — one refines actions, the other refines 
          pixel distributions..."
```

`state.retrieved_docs` = [Chunk A, B, C, D, E, ...] (merged flat list)

---

**Node 3 — `responder`:**

GPT-4o receives all chunks and the original question, then produces:

```
Agent loops in LLM-based systems are iterative control structures 
where the model cycles through Thought → Action → Observation steps 
(the ReAct pattern). At each iteration, the model uses its attention 
mechanism to process the full history of prior actions, enabling it 
to refine its behavior without external memory.

Diffusion video generation faces two primary challenges: temporal 
consistency and computational cost. Maintaining coherent motion 
across frames requires the model to reason over long time horizons — 
something current architectures struggle with. Additionally, generating 
multi-frame sequences requires roughly 8x the compute of single-image 
generation at equivalent resolution.

Interestingly, both processes are iterative in nature — agent loops 
refine actions across steps while diffusion models refine noisy pixel 
distributions across denoising steps. This structural similarity has 
motivated recent work on applying diffusion-style planning to 
agentic decision-making.
```

---

**Final state:**
```python
final_state["sub_questions"]  # ['How do agent loops work...', ...]
final_state["answer"]         # the synthesised paragraph above
```

---

## Key Design Decisions

**Why retrieve per sub-question rather than once with the full query?**
A combined query like *"agent loops AND video diffusion challenges"* creates a single embedding vector that sits between the two topics. Each sub-question embedding sits closer to its own relevant document cluster, pulling more precise chunks.

**Why merge all docs into one flat list for the final answer?**
The responder prompt uses the original compound question, so the LLM needs the full cross-topic context to write a unified answer. Keeping them in one pool lets GPT-4o draw connections across topics (as seen in the third paragraph of the example output).

**When does this pattern underperform?**
When sub-questions are too similar to each other, multiple queries retrieve the same chunks. When sub-questions are very long, the combined context can overflow the model's effective context window. For those cases, adding a deduplication step and per-sub-question answering (like the `cot_answerer` pattern in `research_Agent.ipynb`) is a better approach.
