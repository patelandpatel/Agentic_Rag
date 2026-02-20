# Agentic RAG: Advanced Retrieval-Augmented Generation

This repository contains a collection of implementations for **Agentic RAG** systems. Unlike standard RAG pipelines, these agents use reasoning loops to improve retrieval accuracy, rewrite sub-optimal queries, and perform self-reflection on generated answers.

## 🚀 Features

* **Query Rewriting:** An agentic layer that optimizes user queries for better vector database retrieval.
* **Self-Reflection/Correction:** The system evaluates if the retrieved context is relevant and if the final answer contains hallucinations.
* **Multi-Source Retrieval:** Supports PDF processing (via PyPDF) and Web Search/Scraping integrations.
* **Vector Infrastructure:** Optimized FAISS index management for high-speed similarity search.

## 📂 Project Structure

The repository is organized into specific implementation patterns:

| Folder / Notebook | Description |
| :--- | :--- |
| `Basic_Agentic_Rag/` | The core foundation of the RAG pipeline. |
| `Agentic_Rag_rewrite_query/` | Implementation focused on the Query Transformation agent. |
| `ReAct_Agentic_Reag/` | Using the ReAct (Reason + Act) framework for complex multi-step tool use. |
| `Self_check_answer_ipynb/` | Logic for the self-reflection and grading loop. |
| `query_split_Agentic_Rag/` | Agents that split complex questions into smaller, searchable sub-queries. |

## 🛠️ Installation

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/patelandpatel/Agentic_Rag.git](https://github.com/patelandpatel/Agentic_Rag.git)
   cd Agentic_Rag

```

2. **Create a virtual environment:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

```


3. **Install dependencies:**
```bash
pip install -r requirements.txt

```


4. **Set up Environment Variables:**
Create a `.env` file in the root directory:
```env
OPENAI_API_KEY=your_api_key_here
TAVILY_API_KEY=your_tavily_key_here  # Optional for web search

```



## 🖥️ Usage

Open any of the notebooks in the subdirectories to explore specific agentic behaviors. For a complete end-to-end flow, start with:
`Basic_Agentic_Rag/Agentic_Rag.ipynb`

## 🛡️ License

Distributed under the MIT License. See `LICENSE` for more information.
s
