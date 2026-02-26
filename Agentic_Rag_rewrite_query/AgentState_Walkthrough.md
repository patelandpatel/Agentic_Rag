# AgentState — Complete Walkthrough

## What is AgentState?

```python
class AgentState(TypedDict):
    messages:          Annotated[Sequence[BaseMessage], add_messages]
    sub_questions:     list[str]
    retrieved_docs:    dict[str, str]
    failed_questions:  list[str]
    rewrite_count:     int
```

AgentState is a **shared dictionary** that every node in the graph can read from and write to.
Think of it as a whiteboard — each node reads what's on it, does its work, and writes its results back.

---

## The 5 Fields Explained

### 1. `messages: Annotated[Sequence[BaseMessage], add_messages]`

The **full conversation tape**. Every message ever sent in this pipeline run lives here.

**The `add_messages` reducer** is the key detail:
- Without it → each node **replaces** the list (old messages lost)
- With it → each node **appends** to the list (all messages preserved)

```
# WRONG without add_messages (replaces)
messages = [HumanMessage("question")]
node returns {"messages": [AIMessage("answer")]}
messages = [AIMessage("answer")]           ← original question GONE ❌

# CORRECT with add_messages (appends)
messages = [HumanMessage("question")]
node returns {"messages": [AIMessage("answer")]}
messages = [HumanMessage("question"), AIMessage("answer")]  ← both kept ✅
```

**Who uses it:**
| Access pattern | Who uses it | Why |
|---|---|---|
| `messages[0].content` | decompose, generate, rewrite | Always the original question |
| `messages[-1]` | grade node | Always the most recent tool result |
| Full list | agent node | Sends full history to LLM |

---

### 2. `sub_questions: list[str]`

The **master list of sub-questions** for this pipeline run.

- Created by `decompose` node at the start
- Updated by `rewrite` node — failed questions replaced with rewritten versions
- Used by `agent` to know what to retrieve
- Used by `grade` to know what to grade
- Used by `generate` to structure the final context

**Key rule:** Successful sub-questions are **never removed** from this list. Only failed ones get replaced in-place by their rewritten versions.

---

### 3. `retrieved_docs: dict[str, str]`

A **mapping from sub-question → retrieved content**.

```python
{
    "What is LangGraph?": "[Chunk 1] Score: 0.82 | Source: ...\nLangGraph is...",
    "What is LangChain?": "[Chunk 1] Score: 0.79 | Source: ...\nLangChain is...",
    "How do they differ?": ""   # empty = failed
}
```

**Key rules:**
- Accumulates across ALL retrieval passes — successful results are never overwritten
- When a sub-question is rewritten, its content migrates to the new key
- `generate` reads this to build the context block for the final answer
- Empty string `""` = attempted but returned no relevant content

---

### 4. `failed_questions: list[str]`

The **list of sub-questions that failed grading** in the last pass.

- Written by `grade` node after each grading round
- Read by `agent` node — on rewrite passes only failed questions are re-processed
- Read by `route_after_grade` — decides whether to rewrite or generate
- Cleared by `rewrite` node after rewriting (will be repopulated by next grade pass)

**Key rule:** `agent` checks this first:
```python
pending = failed if failed else sub_questions
# First pass: failed=[] → process all sub_questions
# Rewrite pass: failed=["Q3"] → only process failed ones
```

---

### 5. `rewrite_count: int`

A **loop counter** that prevents infinite rewrite loops.

- Starts at `0` (set by `decompose`)
- Incremented by `rewrite` node each cycle
- Compared against `MAX_REWRITES = 3` in `route_after_grade`

**Routing logic:**
```
rewrite_count < 3  →  keep trying (go to rewrite)
rewrite_count = 3  →  ceiling hit
    some passed    →  generate with partial context
    none passed    →  END (avoid hallucination)
```

---

## Full Example Walkthrough

**Query:** `"What is LangGraph and how does it differ from LangChain?"`

---

### `run()` — Initial State Created

```python
initial_state = {
    "messages":         [HumanMessage("What is LangGraph and how does it differ from LangChain?")],
    "sub_questions":    [],
    "retrieved_docs":   {},
    "failed_questions": [],
    "rewrite_count":    0,
}
```

```
┌─────────────────────────────────────────────────────────────┐
│ STATE AFTER run() initialisation                            │
├─────────────────────────────────────────────────────────────┤
│ messages:          [HumanMessage("What is LangGraph...?")]  │
│ sub_questions:     []                                       │
│ retrieved_docs:    {}                                       │
│ failed_questions:  []                                       │
│ rewrite_count:     0                                        │
└─────────────────────────────────────────────────────────────┘
```

---

### STAGE 1 — `decompose` node

**Reads:** `messages[0].content` → the original question
**LLM splits it into 3 sub-questions**
**Writes:** `sub_questions`, initialises `retrieved_docs`, `failed_questions`, `rewrite_count`

```
┌─────────────────────────────────────────────────────────────────────────┐
│ STATE AFTER decompose                                                   │
├─────────────────────────────────────────────────────────────────────────┤
│ messages:          [HumanMessage("What is LangGraph...?")]              │
│                    ← unchanged, add_messages appended nothing new       │
│                                                                         │
│ sub_questions:     ["What is LangGraph?",                               │
│                     "What is LangChain?",                               │
│                     "How does LangGraph differ from LangChain?"]        │
│                    ← NEW — decompose wrote these                        │
│                                                                         │
│ retrieved_docs:    {}                                                   │
│                    ← initialised empty                                  │
│                                                                         │
│ failed_questions:  []                                                   │
│                    ← initialised empty                                  │
│                                                                         │
│ rewrite_count:     0                                                    │
│                    ← initialised to 0                                   │
└─────────────────────────────────────────────────────────────────────────┘
```

---

### STAGE 2 — `agent` node (Pass 1)

**Reads:** `failed_questions` (empty) → so processes all `sub_questions`
**For each sub-question:** sends `HumanMessage(sub_q)` to LLM → LLM returns AIMessage with tool_call
**Writes:** 3 new AIMessages appended to `messages`

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│ STATE AFTER agent (Pass 1)                                                       │
├──────────────────────────────────────────────────────────────────────────────────┤
│ messages:          [HumanMessage("What is LangGraph...?"),                       │
│                     AIMessage(tool_calls=[{name:"langgraph_docs",                │
│                                            args:{"query":"What is LangGraph?"}}]),│
│                     AIMessage(tool_calls=[{name:"langchain_docs",                │
│                                            args:{"query":"What is LangChain?"}}]),│
│                     AIMessage(tool_calls=[{name:"langgraph_docs",                │
│                                            args:{"query":"How does LangGraph..."}}])]│
│                    ← 3 new AIMessages appended by add_messages                  │
│                                                                                  │
│ sub_questions:     ["What is LangGraph?",                                        │
│                     "What is LangChain?",                                        │
│                     "How does LangGraph differ from LangChain?"]                 │
│                    ← unchanged                                                   │
│                                                                                  │
│ retrieved_docs:    {}           ← unchanged                                      │
│ failed_questions:  []           ← unchanged                                      │
│ rewrite_count:     0            ← unchanged                                      │
└──────────────────────────────────────────────────────────────────────────────────┘
```

---

### STAGE 3 — `retriever` (ToolNode, Pass 1)

**Reads:** The last AIMessages with tool_calls
**Executes:** Each tool call → runs similarity search on vectorstore
**Writes:** 3 ToolMessages appended to `messages`

Let's say:
- Q1 "What is LangGraph?" → tool returns **bad content** (score below threshold → empty string)
- Q2 "What is LangChain?" → tool returns **good content**
- Q3 "How do they differ?" → tool call **never made** (agent skipped it) → no ToolMessage

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│ STATE AFTER retriever (Pass 1)                                                   │
├──────────────────────────────────────────────────────────────────────────────────┤
│ messages:          [HumanMessage("What is LangGraph...?"),        ← [0]          │
│                     AIMessage(tool_calls=[langgraph_docs Q1]),    ← [1]          │
│                     AIMessage(tool_calls=[langchain_docs Q2]),    ← [2]          │
│                     AIMessage(tool_calls=[langgraph_docs Q3]),    ← [3]          │
│                     ToolMessage("No relevant docs above threshold"),← [4] Q1 bad │
│                     ToolMessage("[Chunk 1] Score: 0.81 | Source:  ← [5] Q2 good  │
│                                  LangChain is a framework..."),                  │
│                     ToolMessage("No relevant docs above threshold")]← [6] Q3 bad │
│                    ← 3 ToolMessages appended                                     │
│                                                                                  │
│ sub_questions:     ["What is LangGraph?", "What is LangChain?",                  │
│                     "How does LangGraph differ from LangChain?"]  ← unchanged   │
│ retrieved_docs:    {}           ← unchanged (grade node updates this)            │
│ failed_questions:  []           ← unchanged                                      │
│ rewrite_count:     0            ← unchanged                                      │
└──────────────────────────────────────────────────────────────────────────────────┘
```

---

### STAGE 4 — `grade` node (Pass 1)

**Reads:** ToolMessages by position → maps to pending sub_questions
**Calls:** grader LLM for each sub-question with content
**Writes:** `retrieved_docs` updated, `failed_questions` populated

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│ STATE AFTER grade (Pass 1)                                                       │
├──────────────────────────────────────────────────────────────────────────────────┤
│ messages:          [... same 7 messages ...]   ← unchanged (grade is a NODE,     │
│                                                   not an edge, but doesn't touch │
│                                                   messages)                      │
│                                                                                  │
│ sub_questions:     ["What is LangGraph?",                                        │
│                     "What is LangChain?",                                        │
│                     "How does LangGraph differ from LangChain?"]  ← unchanged   │
│                                                                                  │
│ retrieved_docs:    {                                                             │
│                      "What is LangGraph?":  "",         ← empty, no content     │
│                      "What is LangChain?":  "[Chunk 1] Score: 0.81...",← GOOD   │
│                      "How does LangGraph differ...": "", ← empty, no content    │
│                    }                                                             │
│                    ← NEW — grade wrote content for Q2, empty for Q1 and Q3     │
│                                                                                  │
│ failed_questions:  ["What is LangGraph?",                                        │
│                     "How does LangGraph differ from LangChain?"]                 │
│                    ← NEW — Q1 failed (empty), Q2 passed, Q3 failed (empty)      │
│                                                                                  │
│ rewrite_count:     0            ← unchanged                                      │
└──────────────────────────────────────────────────────────────────────────────────┘
```

**`route_after_grade` reads state:**
- `failed = ["Q1", "Q3"]` → not empty
- `rewrite_count = 0` < `MAX_REWRITES = 3`
- → Routes to **`rewrite`**
- Logs: `"Routing to rewrite (attempt 1/3) for 2 failed sub-question(s)"`

---

### STAGE 5 — `rewrite` node

**Reads:** `failed_questions` → rewrites each with LLM
**Writes:** `sub_questions` updated (failed replaced with rewritten), `failed_questions` cleared, `rewrite_count` incremented

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│ STATE AFTER rewrite                                                              │
├──────────────────────────────────────────────────────────────────────────────────┤
│ messages:          [... same 7 messages ...]   ← unchanged                       │
│                                                                                  │
│ sub_questions:     ["What is LangGraph, and what are its features or use cases?",│
│                     "What is LangChain?",             ← Q2 KEPT (it passed)      │
│                     "What are the primary differences between LangGraph          │
│                      and LangChain?"]                                            │
│                    ← Q1 and Q3 REPLACED with rewritten versions                 │
│                                                                                  │
│ retrieved_docs:    {                                                             │
│                      "What is LangGraph, and what are its features or use cases?": "",│
│                      ← key migrated from old Q1 to new Q1                       │
│                      "What is LangChain?": "[Chunk 1] Score: 0.81...",          │
│                      ← Q2 content PRESERVED under same key                      │
│                      "What are the primary differences...": "",                  │
│                      ← key migrated from old Q3 to new Q3                       │
│                    }                                                             │
│                                                                                  │
│ failed_questions:  []           ← CLEARED — will be repopulated by next grade   │
│                                                                                  │
│ rewrite_count:     1            ← INCREMENTED from 0 to 1                       │
└──────────────────────────────────────────────────────────────────────────────────┘
```

---

### STAGE 6 — `agent` node (Pass 2)

**Reads:** `failed_questions` is `[]` but wait — it was cleared by rewrite.
So it falls back to `sub_questions` and processes all 3.
But `agent` is smart: Q2 already has content in `retrieved_docs`, so ideally only Q1 and Q3 need new retrieval.

> **Note:** In this implementation `agent` processes all pending sub-questions again on the rewrite pass. The `retrieved_docs` for Q2 is simply overwritten with the same good content again — no harm done.

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│ STATE AFTER agent (Pass 2)                                                       │
├──────────────────────────────────────────────────────────────────────────────────┤
│ messages:          [HumanMessage("What is LangGraph...?"),        ← [0]          │
│                     AIMessage(tool_calls Q1 original),            ← [1]          │
│                     AIMessage(tool_calls Q2 original),            ← [2]          │
│                     AIMessage(tool_calls Q3 original),            ← [3]          │
│                     ToolMessage(Q1 bad),                          ← [4]          │
│                     ToolMessage(Q2 good),                         ← [5]          │
│                     ToolMessage(Q3 bad),                          ← [6]          │
│                     AIMessage(tool_calls new Q1 rewritten),       ← [7] NEW      │
│                     AIMessage(tool_calls Q2 again),               ← [8] NEW      │
│                     AIMessage(tool_calls new Q3 rewritten)]       ← [9] NEW      │
│                    ← 3 more AIMessages appended                                  │
│                                                                                  │
│ sub_questions:     ["What is LangGraph, and what are its features or use cases?",│
│                     "What is LangChain?",                                        │
│                     "What are the primary differences between LangGraph...?"]    │
│                    ← unchanged                                                   │
│ retrieved_docs:    {Q1: "", Q2: "[Chunk 1]...", Q3: ""}  ← unchanged            │
│ failed_questions:  []           ← unchanged (still cleared from rewrite)         │
│ rewrite_count:     1            ← unchanged                                      │
└──────────────────────────────────────────────────────────────────────────────────┘
```

---

### STAGE 7 — `retriever` (ToolNode, Pass 2)

This time the rewritten queries match the vectorstore content well.

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│ STATE AFTER retriever (Pass 2)                                                   │
├──────────────────────────────────────────────────────────────────────────────────┤
│ messages:          [... previous 10 messages ...,                                │
│                     ToolMessage("[Chunk 1] Score: 0.77 | LangGraph is a          │
│                                  low-level orchestration framework..."),  ← [10] │
│                     ToolMessage("[Chunk 1] Score: 0.81 | LangChain is..."), ← [11]│
│                     ToolMessage("[Chunk 1] Score: 0.74 | LangGraph differs       │
│                                  from LangChain in that...")]    ← [12]          │
│                    ← 3 more ToolMessages appended, total 13 messages now         │
│                                                                                  │
│ sub_questions:     [rewritten Q1, Q2, rewritten Q3]  ← unchanged                │
│ retrieved_docs:    {Q1: "", Q2: "[Chunk 1]...", Q3: ""}  ← unchanged            │
│ failed_questions:  []           ← unchanged                                      │
│ rewrite_count:     1            ← unchanged                                      │
└──────────────────────────────────────────────────────────────────────────────────┘
```

---

### STAGE 8 — `grade` node (Pass 2)

**This time all 3 pass grading.**

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│ STATE AFTER grade (Pass 2)                                                       │
├──────────────────────────────────────────────────────────────────────────────────┤
│ messages:          [... 13 messages unchanged ...]                               │
│                                                                                  │
│ sub_questions:     ["What is LangGraph, and what are its features or use cases?",│
│                     "What is LangChain?",                                        │
│                     "What are the primary differences between LangGraph...?"]    │
│                    ← unchanged                                                   │
│                                                                                  │
│ retrieved_docs:    {                                                             │
│                      "What is LangGraph, and what are its features or use cases?":│
│                        "[Chunk 1] Score: 0.77 | LangGraph is...",  ← UPDATED ✅ │
│                      "What is LangChain?":                                       │
│                        "[Chunk 1] Score: 0.81 | LangChain is...", ← preserved ✅│
│                      "What are the primary differences between LangGraph...?":   │
│                        "[Chunk 1] Score: 0.74 | LangGraph differs...", ← UPDATED✅│
│                    }                                                             │
│                    ← all 3 now have content                                      │
│                                                                                  │
│ failed_questions:  []           ← all passed, nothing failed                    │
│                                                                                  │
│ rewrite_count:     1            ← unchanged                                      │
└──────────────────────────────────────────────────────────────────────────────────┘
```

**`route_after_grade` reads state:**
- `failed = []` → empty!
- → Routes to **`generate`**
- Logs: `"All sub-questions passed → generate"`

---

### STAGE 9 — `generate` node

**Reads:**
- `messages[0].content` → original question `"What is LangGraph and how does it differ from LangChain?"`
- `retrieved_docs` → builds structured context block
- `sub_questions` → to iterate and structure context

**Writes:** Final answer string appended to `messages`

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│ STATE AFTER generate (FINAL)                                                     │
├──────────────────────────────────────────────────────────────────────────────────┤
│ messages:          [HumanMessage("What is LangGraph...?"),        ← [0]  always  │
│                     ... (12 intermediate messages) ...,                           │
│                     "LangGraph is a low-level orchestration        ← [-1] ANSWER │
│                      framework for building stateful agents.                     │
│                      LangChain builds on top of LangGraph..."]                   │
│                    ← final answer string appended as last message                │
│                                                                                  │
│ sub_questions:     ["What is LangGraph, and what are its features or use cases?",│
│                     "What is LangChain?",                                        │
│                     "What are the primary differences between LangGraph...?"]    │
│                    ← unchanged                                                   │
│                                                                                  │
│ retrieved_docs:    { Q1: "...", Q2: "...", Q3: "..." }  ← unchanged             │
│                                                                                  │
│ failed_questions:  []           ← unchanged                                      │
│ rewrite_count:     1            ← unchanged (only rewrote once)                  │
└──────────────────────────────────────────────────────────────────────────────────┘
```

**`run()` then extracts:** `messages[-1]` → the final answer string → returns it.

---

## Visual Summary of State Changes Per Stage

```
Stage            messages        sub_questions    retrieved_docs    failed_q    rewrite_count
─────────────────────────────────────────────────────────────────────────────────────────────
run() init       [HumanMsg]      []               {}                []          0
─────────────────────────────────────────────────────────────────────────────────────────────
decompose        unchanged       [Q1, Q2, Q3]     {}                []          0
─────────────────────────────────────────────────────────────────────────────────────────────
agent P1         +3 AIMsg        unchanged        unchanged         unchanged   unchanged
─────────────────────────────────────────────────────────────────────────────────────────────
retriever P1     +3 ToolMsg      unchanged        unchanged         unchanged   unchanged
─────────────────────────────────────────────────────────────────────────────────────────────
grade P1         unchanged       unchanged        {Q1:"",           [Q1,Q3]     unchanged
                                                   Q2:"good",
                                                   Q3:""}
─────────────────────────────────────────────────────────────────────────────────────────────
rewrite          unchanged       [Q1new,Q2,Q3new] keys migrated     []          1
─────────────────────────────────────────────────────────────────────────────────────────────
agent P2         +3 AIMsg        unchanged        unchanged         unchanged   unchanged
─────────────────────────────────────────────────────────────────────────────────────────────
retriever P2     +3 ToolMsg      unchanged        unchanged         unchanged   unchanged
─────────────────────────────────────────────────────────────────────────────────────────────
grade P2         unchanged       unchanged        {Q1:"good",       []          unchanged
                                                   Q2:"good",
                                                   Q3:"good"}
─────────────────────────────────────────────────────────────────────────────────────────────
generate         +final answer   unchanged        unchanged         unchanged   unchanged
─────────────────────────────────────────────────────────────────────────────────────────────
```

**Key insight:** `messages[0]` is **always the original question** because `add_messages` only appends — it never shifts or removes. `messages[-1]` is always the most recent thing added.

---

## How Multiple Queries Are Handled

```python
questions = [
    "What is LangGraph and how does it differ from LangChain?",
    "How do I build a chatbot with memory using LangChain?",
    "What are LangGraph workflows and how do I use map-reduce in it?",
]

for q in questions:
    answer = run(q)
```

Each call to `run()` creates a **completely fresh `initial_state`**:

```python
def run(question: str) -> str:
    initial_state = {
        "messages":         [HumanMessage(content=question)],  # brand new list
        "sub_questions":    [],
        "retrieved_docs":   {},
        "failed_questions": [],
        "rewrite_count":    0,
    }
    result = graph.invoke(initial_state)  # completely isolated execution
```

So each question runs in total isolation:

```
Query 1 run:   messages[0] = HumanMessage("What is LangGraph...?")
               → full pipeline → answer returned
               → state DISCARDED after graph.invoke() returns

Query 2 run:   messages[0] = HumanMessage("How do I build a chatbot...?")
               → brand new state, no memory of Query 1
               → full pipeline → answer returned

Query 3 run:   messages[0] = HumanMessage("What are LangGraph workflows...?")
               → brand new state, no memory of Query 1 or 2
               → full pipeline → answer returned
```

**`messages[0]` is always the current question** because each `run()` starts a fresh list with only that question as its first element.

### What if you wanted shared memory across queries?

You would need to persist state between runs — either by passing the previous `result["messages"]` into the next `initial_state`, or by using LangGraph's built-in checkpointing with a `MemorySaver`. The current implementation does not do this — each query is independent by design.
