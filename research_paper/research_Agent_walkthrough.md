# research_Agent.ipynb — Full Execution Walkthrough

This notebook builds a **multimodal agentic RAG system** that can answer questions about scientific research papers. It handles text, tables, and figures from PDFs, embeds everything into a FAISS vector store, and then queries it through an 8-node LangGraph agent with chain-of-thought reasoning and self-reflection.

---

## Architecture at a Glance

```
PDF Files
   │
   ├─ Step 1: Extract structured text blocks (PyMuPDF)
   ├─ Step 2: Extract tables → Markdown (pdfplumber)
   ├─ Step 3: Extract images/figures (PyMuPDF)
   ├─ Step 4: Describe figures via GPT-4o Vision → text
   ├─ Step 5: Hybrid chunking (recursive splitter, 800 chars, 100 overlap)
   ├─ Step 6: Assemble all chunks (text + table + figure)
   └─ Step 7: Embed → FAISS vector store
                            │
                            ▼
                     LangGraph Agent
          ┌──────────────────────────────────┐
          │  query_planner                   │ ← Block 1 (Yellow)
          │  faiss_retriever                 │ ← Retrieval
          │  iterative_check                 │ ← Block 4 (Green)
          │  react_router (if needed)        │ ← Block 3 (Red)
          │  cot_answerer                    │ ← Block 2 (Pink)
          │  answer_synthesiser              │ ← Block 5 (Blue)
          │  self_reflector                  │ ← Block 6 (Orange)
          │  query_rewriter (if needed)      │ ← Loop-back helper
          └──────────────────────────────────┘
```

---

## Phase 1: PDF Extraction

### Step 1 — Structured Text Extraction (`extract_text_with_structure`)

PyMuPDF (`fitz`) parses every page into text blocks. For each block it records:
- The raw text
- The maximum font size in the block
- The bounding box position on the page

It then classifies blocks as **headings** vs **body text** using three conditions:
- Font size ≥ 11pt
- Text is shorter than 80 characters
- Text matches a known scientific section name (e.g. "abstract", "results", "methods")

After classification, `assign_sections` walks through the blocks in order and stamps each one with the current section name — so every paragraph knows it belongs to "introduction" or "results" etc.

**Example output for a paper on m6A methylation:**
```
Block 1 → heading, font=14, section="introduction"
          text: "Introduction"

Block 2 → text, font=10, section="introduction"
          text: "N6-methyladenosine (m6A) is the most abundant internal
                 modification on eukaryotic mRNA..."

Block 3 → text, font=10, section="results"
          text: "METTL3 deletion in NK cells reduced AKT phosphorylation
                 by approximately 40% compared to wild-type controls..."
```

---

### Step 2 — Table Extraction (`extract_tables`)

pdfplumber scans each page for tables. For each table found:
1. The first row becomes column headers
2. Remaining rows are parsed into a Pandas DataFrame
3. The DataFrame is converted to **Markdown format** for human-readable storage
4. The first 50 words from that page are captured as `context` (for section inference)

**Example table chunk content:**
```
Table 1 on page 4.
Context: Flow cytometry analysis of NK cells from METTL3-conditional...

| Cell Type | %IFN-γ+ | %TNF-α+ | p-value |
|-----------|---------|---------|---------|
| WT NK     | 42.3    | 38.1    | —       |
| KO NK     | 18.7    | 15.2    | <0.001  |
```

---

### Step 3 — Image & Figure Extraction (`extract_images`)

PyMuPDF's `get_images()` enumerates every embedded image on each page. For each image:
- Images smaller than 100×100 pixels are skipped (logos, icons, decorative elements)
- Qualifying images are saved to `extracted_output/figures/`
- `find_caption` searches for text blocks immediately below the image bounding box (within 150px) and prioritises blocks starting with "Fig" or "Figure"

**Example:**
```
paper_page3_fig0.png  → 680×520px, saved to figures/
Caption found: "Figure 2. METTL3 deletion impairs NK cell cytokine production..."
```

---

### Step 4 — Figure Description via GPT-4o Vision (`describe_figure_with_openai`)

This is where the pipeline becomes **multimodal**. Each extracted image is base64-encoded and sent to GPT-4o with a structured scientific prompt asking for:

1. Figure type (flow cytometry, western blot, bar graph, etc.)
2. Biological entities shown (cell types, proteins, markers)
3. Quantitative values visible (percentages, p-values, MFI)
4. What the figure demonstrates scientifically
5. Differences between experimental groups

The returned text description is stored as a `DocumentChunk` with `content_type="figure"` — making the visual information searchable via text embedding.

**Example figure chunk content:**
```
Figure on page 3.
Caption: Figure 2. METTL3 deletion impairs NK cell cytokine production.

Description: This is a flow cytometry dot plot showing intracellular cytokine 
staining of NK cells. The x-axis shows IFN-γ expression and the y-axis shows 
TNF-α expression. Two groups are compared: wild-type (WT) NK cells showing 42.3% 
double-positive cells, and METTL3-knockout (KO) NK cells showing 18.7% 
double-positive cells. The p-value for this difference is <0.001, indicating 
a statistically significant reduction in cytokine production upon METTL3 deletion.
```

This description is what gets embedded — so a user asking *"what does the flow cytometry show about METTL3 knockout?"* can retrieve this chunk even though the original content was an image.

---

### Step 5 — Hybrid Chunking (`recursive_split` + `create_text_chunks`)

Text from each section is split using a **recursive character splitter** — it tries to break on paragraph boundaries (`\n\n`), then line breaks (`\n`), then sentences (`. `), then words (` `), and only splits mid-word as a last resort.

**Parameters:**
- `CHUNK_SIZE = 800` characters
- `CHUNK_OVERLAP = 100` characters

Each chunk gets 100 characters of the previous chunk prepended, preserving context across boundaries.

**Example:**
```
Section "results" text (1,800 chars total) → split into 3 chunks:

Chunk 0: "METTL3-deficient NK cells showed a marked reduction in 
          m6A modification levels... [800 chars]"

Chunk 1: "[last 100 chars of chunk 0]... AKT phosphorylation was 
          significantly decreased in KO cells... [800 chars]"

Chunk 2: "[last 100 chars of chunk 1]... These findings suggest that 
          m6A modification regulates NK cell metabolic reprogramming."
```

---

### Step 6 — Assemble All Chunks

The pipeline calls all three chunk creators and concatenates the results:

```
text_chunks    → N chunks  (one DocumentChunk per section/split)
figure_chunks  → M chunks  (one per figure, containing GPT-4o description)
table_chunks   → K chunks  (one per detected table, in Markdown)

all_chunks = figure_chunks + text_chunks + table_chunks
```

Each `DocumentChunk` dataclass carries:
```python
chunk_id       # UUID
source_file    # e.g. "mettl3_nk_paper.pdf"
page_number    # e.g. 4
content_type   # "text" | "table" | "figure"
section        # e.g. "results"
content        # the actual text to embed
image_path     # only for figures
metadata       # dict with extra info (dimensions, row count, etc.)
```

---

### Step 7 — FAISS Vector Store (`embed_chunks` + `build_faiss_index`)

All chunk contents are embedded using **`all-MiniLM-L6-v2`** from SentenceTransformers (384 dimensions, fast, good for scientific text). Embeddings are computed in batches of 32.

```
384-dim float32 vectors
      │
      ▼
faiss.IndexFlatL2  ← exact L2 distance search (no approximation)
      │
      ├── faiss_index.bin   (the binary vector index)
      └── metadata.pkl      (list of dicts — one per chunk, preserves all fields)
```

**Search at query time:** The query is embedded with the same model. FAISS returns the top-K nearest vectors by L2 distance. Lower distance = more similar. A threshold of `RELEVANCE_THRESHOLD = 0.70` is used — chunks with distance above this are considered irrelevant.

---

## Phase 2: LangGraph Agent

The agent is an 8-node LangGraph DAG with a self-reflection loop. The state object `AgenticRAGState` (Pydantic BaseModel) is passed through every node immutably — each node calls `state.model_copy(update={...})` to return a modified copy.

### State Schema

```python
class AgenticRAGState(BaseModel):
    original_question: str       # never changed — used for final reflection
    question: str                # current active query (may be rewritten)
    sub_queries: List[str]       # decomposed sub-questions
    retrieved_docs: List[Document]
    retrieval_source: str        # "faiss" | "arxiv" | "wikipedia"
    docs_are_relevant: bool
    cot_reasoning: str           # scratchpad from CoT node
    sub_answers: List[str]       # one answer per sub-query
    final_answer: str
    reflection: str              # YES/NO verdict + explanation
    needs_revision: bool         # True → loop back
    attempts: int                # loop guard
```

---

### Node-by-Node Description

#### Node 1: `query_planner` (Yellow)

**Purpose:** Decompose the user question into 1–4 focused sub-questions.

**Why:** A broad question retrieves vague chunks. Narrow sub-questions pull precise targeted passages.

**How:** Sends the question to GPT-4o with a prompt that says "return only a JSON array of sub-questions." Strips any markdown fences from the response, parses the JSON, and stores the result in `state.sub_queries`. Falls back to a single-item list if parsing fails.

---

#### Node 2: `faiss_retriever`

**Purpose:** Query the FAISS index for each sub-query and collect all matching chunks.

**How:** Runs `query_faiss_store` for each sub-query (top-4 per query). Results are deduplicated by `chunk_id` before being packed into `Document` objects. The metadata dict attached to each Document includes `content_type`, `image_path`, `distance`, etc.

---

#### Node 3: `iterative_retrieval_check` (Green)

**Purpose:** Decide if the retrieved docs are good enough before spending tokens on generation.

**Two-stage check:**
1. **Fast cosine check** — If the best L2 distance from FAISS is below `RELEVANCE_THRESHOLD = 0.70`, the docs pass immediately. No LLM call.
2. **LLM relevance judge** — If the fast check fails, the first 300 chars of up to 4 docs are shown to GPT-4o with a YES/NO prompt: *"Do these documents contain enough information to answer this query?"*

Sets `state.docs_are_relevant = True/False` which drives the next conditional edge.

---

#### Node 4: `react_router` (Red) — only called if docs are NOT relevant

**Purpose:** Select a different retrieval source when FAISS fails.

**Priority:**
1. `faiss` — already tried, but re-selectable if LLM thinks it can work
2. `arxiv` — for academic/research questions (fetches paper abstracts)
3. `wikipedia` — last resort for general background

The LLM is given the query, the `docs_are_relevant` flag, and the previously used source, then responds with exactly one word. If `arxiv` is chosen, `search_arxiv()` fetches up to 3 paper abstracts and prepends them to the existing docs. Wikipedia works the same way.

---

#### Node 5: `cot_answerer` (Pink)

**Purpose:** Answer each sub-query individually with explicit chain-of-thought reasoning.

**How:** For each sub-query, the full context (all retrieved docs concatenated) and the sub-question are passed to GPT-4o with a prompt that instructs it to:
1. Write step-by-step reasoning inside a `<scratchpad>` section
2. Then write the final answer after the closing `</scratchpad>` tag

The scratchpad content is stored in `state.cot_reasoning` for auditability. The final answer portion is stored in `state.sub_answers`.

---

#### Node 6: `answer_synthesiser` (Blue)

**Purpose:** Merge all sub-answers into one coherent, non-repetitive final answer.

**How:** All sub-answers are sent to GPT-4o with the original question. The prompt asks the model to synthesise them, remove redundancy, and flag any contradictions. The retrieval source is appended as a footnote (e.g. `📚 Sources used: FAISS`).

---

#### Node 7: `self_reflector` (Orange)

**Purpose:** Judge whether the final answer fully addresses the **original** question.

**How:** The original question (not any rewritten version), the final answer, and the retrieved context are passed to GPT-4o with a strict format prompt:
```
Reflection: YES or NO
Explanation: <one sentence>
```

If `YES` → `state.needs_revision = False` → graph exits to END.
If `NO` → `state.needs_revision = True` → graph routes to `query_rewriter`.

Loop is capped at `MAX_ATTEMPTS = 3`.

---

#### Node 8: `query_rewriter` — loop-back helper

**Purpose:** Rewrite `state.question` based on the reflection feedback so the next retrieval pass searches for something meaningfully different.

**How:** The original question and the reflection explanation ("why it failed") are sent to GPT-4o with a prompt asking for only the rewritten query, no explanation. `state.sub_queries` is reset to `[]` so `query_planner` will re-decompose from scratch on the next loop.

---

### Full Graph Wiring

```
START
  │
  ▼
query_planner ──────────────────────────────────────────────────────┐
  │                                                                  │ (loop-back after rewrite)
  ▼                                                                  │
faiss_retriever                                                      │
  │                                                                  │
  ▼                                                                  │
iterative_check                                                      │
  │                                                                  │
  ├─ docs_relevant=True  ──────────────────────────┐                 │
  │                                                │                 │
  └─ docs_relevant=False ──► react_router          │                 │
                                │                  │                 │
                                └──────────────────▼                 │
                                              cot_answerer           │
                                                  │                  │
                                                  ▼                  │
                                          answer_synthesiser         │
                                                  │                  │
                                                  ▼                  │
                                           self_reflector            │
                                                  │                  │
                                ┌─────────────────┴──────────────┐   │
                                │                                │   │
                         needs_revision=False           needs_revision=True
                                │                                │   │
                               END                       query_rewriter
                                                                 │   │
                                                                 └───┘
```

---

## End-to-End Example

**User query:**
```
Does m6A methylation in the coding region (CDS) increase or decrease translation 
efficiency, and how does this contrast with its effect in the 3' UTR?
```

---

**Node 1 — query_planner:**

GPT-4o decomposes this into:
```json
[
  "What is the effect of m6A methylation in the CDS on translation efficiency?",
  "What is the effect of m6A methylation in the 3' UTR on translation efficiency?",
  "How do the translation effects of m6A in CDS and 3' UTR differ from each other?"
]
```

---

**faiss_retriever:**

Runs 3 FAISS queries. Retrieves up to 12 chunks, deduplicated to ~8 unique chunks. Example retrieved chunks:
```
Chunk A (text, results, page 6, distance=0.31):
  "m6A modifications in the coding sequence increased ribosome occupancy
   by 23% in reporter assays, suggesting a positive effect on translation 
   elongation rate..."

Chunk B (figure, results, page 7, distance=0.38):
  "Figure 4. m6A site distribution across mRNA regions.
   Description: Bar graph showing relative translation efficiency (RTE) for 
   mRNA constructs with m6A placed in CDS vs 3' UTR. CDS-modified mRNAs show 
   RTE of 1.8x compared to unmodified controls. 3' UTR-modified mRNAs show 
   RTE of 0.6x..."

Chunk C (text, discussion, page 9, distance=0.45):
  "In contrast to the stimulatory effect observed in the CDS, m6A in the 
   3' UTR recruits YTHDF2 and promotes mRNA decay, effectively reducing 
   protein output..."
```

---

**iterative_retrieval_check:**

Best distance = 0.31 < 0.70 threshold → **Fast check passes immediately.** `docs_are_relevant = True`. No LLM call needed.

---

**cot_answerer:**

For sub-question 1, GPT-4o produces:
```
<scratchpad>
Step 1: Chunk A mentions "m6A in coding sequence increased ribosome occupancy 
        by 23%". Ribosome occupancy increase = faster translation.
Step 2: Chunk B's figure shows CDS-modified mRNA has RTE of 1.8x vs baseline.
Step 3: Both lines of evidence agree — CDS m6A is stimulatory.
</scratchpad>

Answer: m6A methylation in the CDS increases translation efficiency. 
Evidence shows a 23% increase in ribosome occupancy and a 1.8x increase 
in relative translation efficiency in CDS-modified mRNAs.
```

Similar answers are generated for sub-questions 2 and 3.

---

**answer_synthesiser:**

Merges the three sub-answers:
```
m6A methylation has opposing effects on translation depending on its 
position within the mRNA. When located in the coding sequence (CDS), 
m6A increases translation efficiency — reporter assays show a 1.8x 
increase in relative translation efficiency and a 23% boost in ribosome 
occupancy. This appears to accelerate elongation.

In contrast, m6A in the 3' UTR reduces translation output. Rather than 
promoting elongation, 3' UTR m6A recruits YTHDF2, which accelerates 
mRNA decay, leading to a 0.6x relative translation efficiency — a net 
inhibitory effect on protein production.

The key distinction is mechanistic: CDS m6A facilitates ribosome 
progression, while 3' UTR m6A triggers degradation of the transcript 
before translation can complete.

📚 Sources used: FAISS
```

---

**self_reflector:**

GPT-4o evaluates the answer against the original question:
```
Reflection: YES
Explanation: The answer addresses both the CDS effect (stimulatory, 1.8x RTE) 
and the 3' UTR effect (inhibitory, 0.6x RTE) and explains the mechanistic 
contrast between them.
```

`needs_revision = False` → graph exits to **END**.

---

**Final state:**
```python
result["sub_queries"]       # 3 decomposed questions
result["retrieval_source"]  # "faiss"
result["attempts"]          # 1
result["cot_reasoning"]     # scratchpad from all 3 sub-questions
result["final_answer"]      # the synthesised paragraph above
result["reflection"]        # "Reflection: YES ..."
```

---

## Key Design Decisions

**Why GPT-4o Vision for figures?**
Flow cytometry plots, western blots, and bar graphs carry quantitative data (percentages, p-values, MFI values) that cannot be retrieved from text alone. By converting them to textual descriptions, the same FAISS index can serve both text and visual questions.

**Why recursive chunking over fixed-size?**
Scientific text has natural semantic boundaries at paragraphs and sentences. Splitting at those boundaries keeps related concepts together, which improves retrieval precision.

**Why two-stage relevance check?**
The fast cosine check covers the majority of cases without spending an LLM call. The LLM judge only activates when scores are borderline, keeping API costs low.

**Why `original_question` is never mutated?**
The self-reflector always judges the final answer against what the user originally asked, not against any rewritten version. Without this, a rewritten query might pass reflection even if it drifted from user intent.

**Why `MAX_ATTEMPTS = 3`?**
Prevents infinite loops when retrieval genuinely cannot answer a question (e.g. it is outside the indexed papers). After 3 attempts, the best available answer is returned.
