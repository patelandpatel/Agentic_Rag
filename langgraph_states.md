# LangGraph State Types — Complete Guide

LangGraph uses **state** to pass data between nodes in a graph. The state is shared across all nodes and gets updated as the graph executes. There are several types of state available depending on your use case.

---

## Overview

| State Type | Complexity | Best For |
|---|---|---|
| `TypedDict` (Custom State) | High control | RAG, complex pipelines |
| `MessagesState` | Minimal setup | Chatbots, tool-calling agents |
| `Pydantic` State | Validation needed | Production apps, strict typing |
| `dataclass` State | Python-native | Cleaner syntax preference |
| Multi-Schema State | Privacy/performance | Input/output separation |
| Private State | Internal data hiding | Multi-node pipelines |

---

## 1. Custom State with `TypedDict` (Most Common)

The most flexible option. You define every field your graph needs.

```python
from typing import TypedDict, List, Annotated
from langgraph.graph import StateGraph

class MyState(TypedDict):
    question: str
    retrieved_docs: List[str]
    response: str
    attempt_count: int

graph = StateGraph(MyState)
```

### Key Points
- You control exactly what data flows through the graph
- Each node can read and update any field
- No automatic message accumulation — you manage fields manually
- Best for **RAG pipelines**, **multi-step workflows**, **structured data processing**

### Example Node
```python
def retriever_node(state: MyState) -> MyState:
    docs = retrieve(state["question"])
    return {"retrieved_docs": docs}
```

---

## 2. `MessagesState` (Built-in Shortcut)

A pre-built state that ships with LangGraph. It only tracks a list of messages, with an automatic **append reducer** so messages accumulate rather than overwrite.

```python
from langgraph.graph import StateGraph, MessagesState

graph = StateGraph(MessagesState)
```

### Under the Hood
```python
from typing import Annotated
from langgraph.graph.message import add_messages
from langchain_core.messages import AnyMessage

class MessagesState(TypedDict):
    messages: Annotated[List[AnyMessage], add_messages]
```

### How Messages Accumulate
```
messages = [
  HumanMessage("What is the capital of France?"),
  AIMessage(tool_calls=[{"name": "search", ...}]),   # AI calls a tool
  ToolMessage("Paris is the capital of France"),      # Tool result
  AIMessage("The capital of France is Paris.")        # Final answer
]
```

### Key Points
- Zero boilerplate — just plug and play
- `add_messages` reducer **appends** new messages instead of replacing the list
- All tool calls, tool results, AI responses live in one unified list
- Best for **conversational agents**, **ReAct agents**, **tool-calling agents**

---

## 3. Custom State with `add_messages` Reducer

You can combine a custom state with the `add_messages` reducer — getting the best of both worlds: custom fields **plus** auto-accumulating messages.

```python
from typing import TypedDict, List, Annotated
from langgraph.graph.message import add_messages
from langchain_core.messages import AnyMessage

class MyCustomState(TypedDict):
    messages: Annotated[List[AnyMessage], add_messages]  # auto-accumulates
    retrieved_docs: List[str]                             # custom field
    user_id: str                                          # custom field
    response: str                                         # custom field

graph = StateGraph(MyCustomState)
```

### Key Points
- `messages` auto-accumulates like `MessagesState`
- Other fields behave like regular custom state (overwrite on update)
- Best when you need **both** conversation history **and** structured data

---

## 4. Pydantic State (With Validation)

Use Pydantic's `BaseModel` instead of `TypedDict` when you need **runtime validation** of state data.

```python
from pydantic import BaseModel, Field
from typing import List, Optional
from langgraph.graph import StateGraph

class MyState(BaseModel):
    question: str
    retrieved_docs: List[str] = Field(default_factory=list)
    response: Optional[str] = None
    confidence_score: float = Field(ge=0.0, le=1.0, default=0.0)

graph = StateGraph(MyState)
```

### Key Points
- Validates field types at **runtime** — throws errors if wrong type is passed
- Supports default values, field constraints (`ge`, `le`, `min_length`, etc.)
- Slightly more overhead than `TypedDict`
- Best for **production apps** where data integrity matters

---

## 5. Dataclass State

Use Python's `@dataclass` decorator for a cleaner, more Pythonic syntax.

```python
from dataclasses import dataclass, field
from typing import List, Optional
from langgraph.graph import StateGraph

@dataclass
class MyState:
    question: str = ""
    retrieved_docs: List[str] = field(default_factory=list)
    response: Optional[str] = None
    attempt_count: int = 0

graph = StateGraph(MyState)
```

### Key Points
- Cleaner syntax than `TypedDict` — supports default values natively
- Can add methods to the dataclass if needed
- No runtime validation (unlike Pydantic)
- Best when you prefer **Python-native** syntax over dict-like access

---

## 6. Multi-Schema State (Input/Output Schemas)

You can define **separate schemas** for what goes **into** the graph vs what comes **out**. This is useful for hiding internal state from the caller.

```python
from typing import TypedDict, List
from langgraph.graph import StateGraph

# Full internal state (used inside the graph)
class InternalState(TypedDict):
    question: str
    retrieved_docs: List[str]
    intermediate_steps: List[str]   # internal — not exposed
    response: str

# What the user sends in
class InputSchema(TypedDict):
    question: str

# What the graph returns to the user
class OutputSchema(TypedDict):
    response: str

graph = StateGraph(InternalState, input=InputSchema, output=OutputSchema)
```

### Key Points
- The caller only sees `InputSchema` and `OutputSchema`
- Internal fields like `retrieved_docs`, `intermediate_steps` are hidden
- Great for **clean APIs**, **multi-agent systems**, **privacy**

---

## 7. Private State Between Nodes

Sometimes a node needs to pass temporary data to the **next node only**, without polluting the global state. LangGraph supports this with `PrivateState`.

```python
from typing import TypedDict
from langgraph.graph import StateGraph

class PublicState(TypedDict):
    question: str
    response: str

class PrivateState(TypedDict):
    internal_score: float   # Only visible to specific nodes

def node_a(state: PublicState) -> PrivateState:
    # Computes something internally
    return {"internal_score": 0.87}

def node_b(state: PrivateState) -> PublicState:
    # Uses the private data from node_a
    if state["internal_score"] > 0.5:
        return {"response": "High confidence answer"}
    return {"response": "Low confidence answer"}
```

### Key Points
- Temporary data that doesn't need to persist in global state
- Keeps your main state clean
- Useful for **scoring**, **intermediate computations**, **routing signals**

---

## Reducers — How State Gets Updated

By default, when a node returns a value, it **overwrites** the existing field. **Reducers** change this behavior.

### Default (Overwrite)
```python
class State(TypedDict):
    response: str   # new value replaces old value
```

### `add_messages` Reducer (Append)
```python
from langgraph.graph.message import add_messages

class State(TypedDict):
    messages: Annotated[List[AnyMessage], add_messages]  # appends new messages
```

### Custom Reducer
You can write your own reducer for any field:

```python
def merge_lists(existing: List, new: List) -> List:
    return existing + new   # combine old and new lists

class State(TypedDict):
    retrieved_docs: Annotated[List[str], merge_lists]  # accumulates docs
```

---

## Choosing the Right State

```
Are you building a chatbot or tool-calling agent?
  └── YES → Use MessagesState

Do you need messages + extra fields?
  └── YES → Custom TypedDict with add_messages reducer

Do you need runtime validation?
  └── YES → Pydantic BaseModel

Do you want to hide internal state from callers?
  └── YES → Multi-Schema State (input/output schemas)

Do you have temporary data between specific nodes?
  └── YES → Private State

Everything else?
  └── Custom TypedDict (most flexible)
```

---

## Quick Comparison

```python
# Option 1: MessagesState — simplest
graph = StateGraph(MessagesState)

# Option 2: Custom TypedDict — most flexible
class State(TypedDict):
    question: str
    docs: List[str]
    answer: str
graph = StateGraph(State)

# Option 3: Best of both — messages + custom fields
class State(TypedDict):
    messages: Annotated[List[AnyMessage], add_messages]
    docs: List[str]
graph = StateGraph(State)

# Option 4: Pydantic — with validation
class State(BaseModel):
    question: str
    answer: str = ""
graph = StateGraph(State)
```

---

## Summary

- **`MessagesState`** → Fastest to get started, auto-manages message history
- **Custom `TypedDict`** → Most common in production, full control
- **Pydantic** → When data integrity and validation matter
- **Multi-Schema** → Clean public APIs, hide internal complexity
- **Reducers** → Control *how* state fields get updated (overwrite vs append vs custom)

The most powerful pattern is combining a **custom TypedDict** with the **`add_messages` reducer** on the messages field — giving you full control over your data while still auto-managing conversation history.
