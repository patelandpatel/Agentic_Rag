# Agentic RAG — Rewrite Query Agent

## Overview

This project implements a production-level Agentic Retrieval-Augmented Generation (RAG) system using LangGraph. The agent answers user questions by retrieving relevant information from local FAISS vector stores, falling back to Wikipedia when local knowledge is insufficient, and rewriting the query when retrieved documents are not relevant. A rewrite ceiling prevents infinite loops.

---

## Architecture

The system is composed of four major layers that work together in a directed graph.

The first layer is the vectorstore layer, which loads and persists FAISS indexes built from web documentation. The second is the tool layer, which exposes those vectorstores and Wikipedia as callable tools for the agent. The third is the node layer, which defines the logic executed at each step of the graph. The fourth is the graph layer, which wires the nodes together with conditional routing.

---

## Vectorstore Strategy

Each knowledge source gets its own isolated FAISS index stored on disk. On every run, the system checks whether the index already exists. If it does, it loads it directly from disk without making any embedding API calls, which saves both cost and time. If it does not exist, it fetches the URLs, splits the documents into chunks, embeds them, and saves the index for future runs.

Two separate indexes are maintained — one for LangGraph documentation and one for LangChain documentation — so that retrieval stays scoped to the correct knowledge domain.

---

## Embedding Configuration

| Setting | Value | Reason |
|---|---|---|
| Model | text-embedding-3-small | Cheaper and faster than ada-002, with comparable quality |
| Chunk size (embedding) | 200 | Batches API calls to avoid rate limits |
| Chunk size (text) | 1000 | Balances context richness with retrieval precision |
| Chunk overlap | 100 | Prevents information loss at chunk boundaries |
| add_start_index | True | Stores character offset in metadata for traceability |

---

## Retrieval Strategy

Each retriever tool uses a two-stage retrieval pipeline inside a closure so that each tool instance operates independently with its own vectorstore, threshold, and tuning parameters.

In the first stage, MMR (Maximal Marginal Relevance) retrieval is used to select a diverse set of candidate documents. MMR balances relevance with diversity, preventing the same near-duplicate chunks from all being returned for the same query. It draws from a larger candidate pool before selecting the final results.

In the second stage, relevance scores are fetched for the same query using similarity search. Since MMR does not return scores, this second call provides the numeric scores needed to filter out low-quality chunks. Any chunk whose score falls below the configured threshold is dropped before the results are returned to the agent.

---

## MMR Parameters

| Parameter | Default | Description |
|---|---|---|
| k | 5 | Number of final chunks returned to the agent |
| fetch_k | 20 | Candidate pool MMR selects from before applying diversity |
| lambda_mult | 0.6 | Balance between relevance and diversity. 0 = max diversity, 1 = max relevance |
| threshold | 0.30 — 0.35 | Minimum relevance score a chunk must have to be returned |

---

## Tools

Three tools are available to the agent. The agent decides autonomously which tool to call based on the query.

| Tool | Source | Purpose |
|---|---|---|
| retriever_vector_db_blog | Local FAISS index | Answers questions about LangGraph |
| retriever_vector_langchain_blog | Local FAISS index | Answers questions about LangChain |
| wikipedia | Wikipedia API | Fallback when local indexes return no relevant results |

---

## Graph Nodes

| Node | Role |
|---|---|
| agent | Invokes the LLM with bound tools. Decides whether to retrieve or finish |
| retriever | Executes whichever tool the agent selected |
| grade_documents | Scores the retrieved document for relevance. Routes to generate, rewrite, or END |
| rewrite | Rewrites the original question to improve the next retrieval attempt |
| generate | Produces a concise final answer from the retrieved context |

---

## Graph Routing

The agent node routes conditionally based on whether the LLM decided to call a tool. If a tool was called, execution moves to the retriever. If no tool was called, the graph ends.

After retrieval, the grade_documents function inspects the retrieved content and routes in one of three directions. If the document is relevant, execution moves to generate. If the document is not relevant and rewrites remain, execution moves to rewrite and then back to agent for another retrieval attempt. If the document is not relevant and the rewrite ceiling has been reached, the graph ends without generating an answer to prevent infinite loops.

---

## Rewrite Loop Guard

The original query rewrite mechanism had no exit condition, meaning the agent could loop indefinitely if retrieval kept failing. The rewrite_count field in AgentState tracks how many rewrites have occurred. When this count reaches MAX_REWRITES (default 3), grade_documents returns END instead of rewrite, breaking the loop.

If the desired behavior is to always produce some answer even after exhausted rewrites, grade_documents can be changed to return generate instead of END at the ceiling, which forces the agent to answer with whatever context it has.

---

## State Schema

| Field | Type | Description |
|---|---|---|
| messages | Sequence[BaseMessage] | Full conversation history, merged automatically by add_messages |
| rewrite_count | int | Number of query rewrites performed so far in this run |

---

## Module-Level Optimizations

Several objects that were previously rebuilt on every node invocation are now constructed once at module load time. The agent model is bound to tools once. The relevance grading chain is constructed once. The RAG generation chain is constructed once. The rewrite chain is constructed once. This avoids repeated object construction on every graph step and makes each node function a thin wrapper around a pre-built chain.

---

## Known Root Cause of Empty Retrieval

The output logs show that the vectorstore consistently returns chunks with negative relevance scores and content of only the word "Redirecting...". This means the URLs were fetched but the pages returned HTTP redirects instead of actual documentation content. The scraped documents contain no meaningful text, so no chunk can clear the relevance threshold. The fix is to resolve the correct final URLs before passing them to the loader, or to use a loader that follows redirects automatically.
