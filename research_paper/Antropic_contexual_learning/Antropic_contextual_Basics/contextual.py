import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import json
import anthropic
import chromadb
import voyageai
from chromadb.config import Settings
from tqdm import tqdm

# ── Constants ─────────────────────────────────────────────────────────────────
MODEL_NAME = "claude-sonnet-4-5"

DOCUMENT_CONTEXT_PROMPT = """
<document>
{doc_content}
</document>
"""

CHUNK_CONTEXT_PROMPT = """
Here is the chunk we want to situate within the whole document
<chunk>
{chunk_content}
</chunk>

Please give a short succinct context to situate this chunk within the overall 
document for the purposes of improving search retrieval of the chunk.
Answer only with the succinct context and nothing else.
"""


# =============================================================================
# ContextualChromaDB
# =============================================================================

class ContextualChromaDB:
    """
    A contextual vector database backed by ChromaDB (persistent, HNSW-indexed).

    For each chunk, Claude generates a short situating context which is
    prepended to the chunk before embedding. Both the original content and
    the generated context are stored as separate metadata fields so you can
    inspect what Claude produced for every chunk.

    Usage:
        db = ContextualChromaDB(
            name               = "rag_research",
            voyage_api_key     = "...",
            anthropic_api_key  = "...",
        )
        db.load_data(dataset, parallel_threads=1)
        results = db.search("how does DPR work?", k=20)
    """

    def __init__(
        self,
        name:              str,
        voyage_api_key:    str | None = None,
        anthropic_api_key: str | None = None,
        persist_dir:       str        = "./data/chroma",
    ):
        # ── API clients ───────────────────────────────────────────────────────
        if voyage_api_key is None:
            voyage_api_key = os.getenv("VOYAGE_API_KEY")
        if anthropic_api_key is None:
            anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")

        self.voyage_client    = voyageai.Client(api_key=voyage_api_key)
        self.anthropic_client = anthropic.Anthropic(api_key=anthropic_api_key)
        self.name             = name

        # ── Query cache (avoids re-embedding identical queries) ───────────────
        '''
        # ── Query cache (avoids re-embedding identical queries) ───────────────
        import redis, json as _json
        self.query_cache = redis.Redis(host="localhost", port=6379, db=0, decode_responses=True)
        
        '''
        self.query_cache: dict[str, list[float]] = {}

        # ── Token usage tracking (thread-safe) ───────────────────────────────
        self.token_counts = {
            "input":            0,
            "output":           0,
            "cache_read":       0,
            "cache_creation":   0,
        }
        self.token_lock = threading.Lock()

        # ── ChromaDB persistent client with advanced HNSW config ─────────────
        self.chroma_client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True,
            ),
        )

        self.collection = self.chroma_client.get_or_create_collection(
            name=name,
            metadata={
                # cosine similarity — best for text embeddings
                # options: "cosine" | "l2" | "ip"
                "hnsw:space": "cosine",

                # number of neighbours each node connects to in the HNSW graph
                # higher → better recall, more RAM, slower build
                # range: 2–100 | default: 16
                "hnsw:M": 32,

                # search width during index construction
                # higher → better index quality, slower ingestion
                # range: 100–2000 | default: 100
                "hnsw:construction_ef": 200,

                # search width at query time
                # higher → more accurate results, slower query
                # range: 10–500 | default: 10
                "hnsw:search_ef": 100,

                # elements buffered in RAM before flushing to disk
                "hnsw:batch_size":      1000,
                "hnsw:sync_threshold":  2000,
            },
        )

    # =========================================================================
    # situate_context
    # =========================================================================

    def situate_context(self, doc: str, chunk: str) -> tuple[str, Any]:
        """
        Ask Claude to generate a short situating context for one chunk.

        The full document is marked with cache_control so Anthropic caches it
        server-side. Every chunk after the first in the same document reads the
        document from cache at ~10% of normal token cost.

        Returns:
            (contextualized_text, usage)
        """
        response = self.anthropic_client.messages.create(
            model=MODEL_NAME,
            max_tokens=1000,
            temperature=0.0,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            # ── full document (cached after first chunk) ──────
                            "type": "text",
                            "text": DOCUMENT_CONTEXT_PROMPT.format(doc_content=doc),
                            "cache_control": {"type": "ephemeral"},
                        },
                        {
                            # ── the specific chunk to situate ─────────────────
                            "type": "text",
                            "text": CHUNK_CONTEXT_PROMPT.format(chunk_content=chunk),
                        },
                    ],
                }
            ],
            extra_headers={"anthropic-beta": "prompt-caching-2024-07-31"},
        )
        return response.content[0].text, response.usage

    # =========================================================================
    # load_data
    # =========================================================================

    def load_data(
        self,
        dataset:          list[dict[str, Any]],
        parallel_threads: int = 1,
    ) -> None:
        """
        Process every chunk in the dataset:
          1. Generate situating context via Claude (with prompt caching)
          2. Embed  original + context  with Voyage AI
          3. Store vectors + metadata in ChromaDB

        If the collection already contains data, loading is skipped entirely
        to avoid redundant Claude/Voyage API calls.

        Args:
            dataset:          Output of DocumentProcessor.run()
            parallel_threads: Worker threads for Claude calls.
                              1  = sequential (best cache hit rate)
                              >1 = parallel   (faster wall-clock, lower cache hits)
        """
        # ── Guard: skip if already loaded ────────────────────────────────────
        if self.collection.count() > 0:
            print(
                f"Collection '{self.name}' already contains "
                f"{self.collection.count()} chunks. Skipping load."
            )
            return

        total_chunks = sum(len(doc["chunks"]) for doc in dataset)
        print(f"Processing {total_chunks} chunks with {parallel_threads} thread(s).")

        # ── Nested worker: one call per chunk ─────────────────────────────────
        def process_chunk(doc: dict, chunk: dict) -> dict:
            """
            Situate one chunk within its source document.

            Returns a dict with:
              - text_to_embed:  what gets sent to Voyage
              - metadata:       stored in ChromaDB (includes both content fields)
              - chunk_id:       ChromaDB document ID
            """
            contextualized_text, usage = self.situate_context(
                doc["content"], chunk["content"]
            )

            # ── update shared token counters thread-safely ────────────────
            with self.token_lock:
                self.token_counts["input"]          += usage.input_tokens
                self.token_counts["output"]         += usage.output_tokens
                self.token_counts["cache_read"]     += usage.cache_read_input_tokens
                self.token_counts["cache_creation"] += usage.cache_creation_input_tokens

            return {
                # what gets embedded — original chunk + claude context
                "text_to_embed": (
                    f"{chunk['content']}\n\n{contextualized_text}"
                ),

                # stored in ChromaDB — both fields kept separate for inspection
                "metadata": {
                    "doc_id":                 doc["doc_id"],
                    "original_uuid":          doc["original_uuid"],
                    "chunk_id":               chunk["chunk_id"],
                    "original_index":         chunk["original_index"],
                    # ── inspectable content fields ────────────────────────
                    "original_content":       chunk["content"],
                    "contextualized_content": contextualized_text,
                },

                # unique ID for ChromaDB
                "chunk_id": chunk["chunk_id"],
            }

        # ── Submit all chunks to thread pool ──────────────────────────────────
        texts_to_embed: list[str]             = []
        metadatas:      list[dict[str, Any]]  = []
        ids:            list[str]             = []

        print(f"Processing {total_chunks} chunks with {parallel_threads} threads")

        with ThreadPoolExecutor(max_workers=parallel_threads) as executor:
            futures = [
                executor.submit(process_chunk, doc, chunk)
                for doc in dataset
                for chunk in doc["chunks"]
            ]

            for future in tqdm(
                as_completed(futures),
                total=total_chunks,
                desc="Processing chunks",
            ):
                result = future.result()
                texts_to_embed.append(result["text_to_embed"])
                metadatas.append(result["metadata"])
                ids.append(result["chunk_id"])

        # ── Embed + store in ChromaDB ─────────────────────────────────────────
        self._embed_and_store(texts_to_embed, metadatas, ids)

        # ── Token usage report ────────────────────────────────────────────────
        total_tokens = (
            self.token_counts["input"]
            + self.token_counts["cache_read"]
            + self.token_counts["cache_creation"]
        )
        savings_pct = (
            (self.token_counts["cache_read"] / total_tokens) * 100
            if total_tokens > 0
            else 0.0
        )

        print(f"\nContextual ChromaDB loaded. Total chunks: {len(texts_to_embed)}")
        print(f"  Input tokens (full price):      {self.token_counts['input']:,}")
        print(f"  Output tokens:                  {self.token_counts['output']:,}")
        print(f"  Cache creation tokens:          {self.token_counts['cache_creation']:,}")
        print(f"  Cache read tokens (90% off):    {self.token_counts['cache_read']:,}")
        print(f"  Cache savings:                  {savings_pct:.1f}% of input read from cache")

    # =========================================================================
    # _embed_and_store
    # =========================================================================

    def _embed_and_store(
        self,
        texts:     list[str],
        metadatas: list[dict[str, Any]],
        ids:       list[str],
    ) -> None:
        """
        Embed all texts in batches of 128 via Voyage AI, then add everything
        to the ChromaDB collection in a single call.

        Args:
            texts:     enriched strings (original + context) to embed
            metadatas: parallel list of metadata dicts (same order as texts)
            ids:       parallel list of chunk IDs (same order as texts)
        """
        batch_size = 128

        # ── Voyage embedding in batches ───────────────────────────────────────
        print(f"Embedding {len(texts)} chunks with Voyage AI...")

        raw_batches = [
            self.voyage_client.embed(
                texts[i : i + batch_size], model="voyage-4"
            ).embeddings
            for i in tqdm(
                range(0, len(texts), batch_size),
                desc="Embedding batches",
            )
        ]

        # flatten list-of-batches → flat list of vectors
        all_embeddings = [
            embedding
            for batch in raw_batches
            for embedding in batch
        ]

        # ── Store in ChromaDB ─────────────────────────────────────────────────
        # ChromaDB add() is called once with all data — it handles
        # batching internally and persists to disk automatically
        self.collection.add(
            ids        = ids,
            embeddings = all_embeddings,
            documents  = texts,        # the enriched text (original + context)
            metadatas  = metadatas,    # includes original_content + contextualized_content
        )

        print(f"Stored {len(ids)} chunks in ChromaDB collection '{self.name}'.")

    # =========================================================================
    # search
    # =========================================================================

    def search(
        self,
        query:          str,
        k:              int       = 20,
        doc_id_filter:  str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Retrieve the top-k most similar chunks for a query.

        Args:
            query:         The search string
            k:             Number of results to return
            doc_id_filter: Optional — restrict search to one document.
                           Pass a doc_id when debugging against ground truth
                           to isolate whether the problem is doc-level retrieval
                           or chunk-level ranking.

        Returns:
            List of dicts, each containing:
              - metadata  (with original_content + contextualized_content)
              - document  (the enriched text that was embedded)
              - distance  (lower = more similar for cosine)
              - similarity (1 - distance, higher = more similar)
        """
        if not self.collection.count():
            raise ValueError("Collection is empty. Run load_data() first.")


        '''
        _cached = self.query_cache.get(f"query_cache:{query}")
        if _cached:
            query_embedding = _json.loads(_cached)
        else:
            query_embedding = self.voyage_client.embed(
                [query], model="voyage-4"
            ).embeddings[0]
            self.query_cache.set(f"query_cache:{query}", _json.dumps(query_embedding), ex=604800)  # 7-day TTL
        '''
        # ── Query cache: skip Voyage call if seen before ──────────────────────
        if query in self.query_cache:
            query_embedding = self.query_cache[query]
        else:
            query_embedding = self.voyage_client.embed(
                [query], model="voyage-4"
            ).embeddings[0]
            self.query_cache[query] = query_embedding

        # ── Optional metadata pre-filter ─────────────────────────────────────
        # Filtering happens INSIDE ChromaDB before similarity search —
        # much faster than fetching all results and filtering in Python.
        #
        # Use cases:
        #   Production:  doc_id_filter=None  → search entire collection
        #   Debugging:   doc_id_filter="paper1" → compare against ground truth
        where_filter = (
            {"doc_id": {"$eq": doc_id_filter}}
            if doc_id_filter is not None
            else None
        )

        # ── ChromaDB similarity search ────────────────────────────────────────
        raw = self.collection.query(
            query_embeddings = [query_embedding],
            n_results        = k,
            where            = where_filter,
            include          = [
                "metadatas",   # doc_id, chunk_id, original_content, contextualized_content
                "documents",   # enriched text that was embedded
                "distances",   # chromadb cosine distance (0 = identical, 2 = opposite)
            ],
        )

        # ── Format results ────────────────────────────────────────────────────
        results = []
        for i in range(len(raw["ids"][0])):
            distance = raw["distances"][0][i]
            results.append({
                "metadata":   raw["metadatas"][0][i],
                "document":   raw["documents"][0][i],
                "distance":   distance,
                # convert to intuitive 0→1 similarity score
                # ChromaDB cosine distance = 1 - cosine_similarity
                "similarity": round(1 - distance, 4),
            })

        return results


# =============================================================================
# Convenience: evaluate_db compatible wrapper
# =============================================================================

def retrieve_contextual_chroma(
    query: str,
    db:    ContextualChromaDB,
    k:     int = 20,
) -> list[dict[str, Any]]:
    """
    Drop-in replacement for retrieve_base() in evaluate_retrieval().
    Wraps ContextualChromaDB.search() to match the expected signature.
    """
    return db.search(query, k=k)





import json
import os
from typing import Any

from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
from tqdm import tqdm


class ElasticsearchBM25:
    def __init__(self, index_name: str = "contextual_bm25_index"):
        self.es_client = Elasticsearch("http://localhost:9200")
        self.index_name = index_name
        self.create_index()

    def create_index(self):
        index_settings = {
            "settings": {
                "analysis": {"analyzer": {"default": {"type": "english"}}},
                "similarity": {"default": {"type": "BM25"}},
                "index.queries.cache.enabled": False,
            },
            "mappings": {
                "properties": {
                    "content": {"type": "text", "analyzer": "english"},
                    "contextualized_content": {"type": "text", "analyzer": "english"},
                    "doc_id": {"type": "keyword", "index": False},
                    "chunk_id": {"type": "keyword", "index": False},
                    "original_index": {"type": "integer", "index": False},
                }
            },
        }

        if not self.es_client.indices.exists(index=self.index_name):
            self.es_client.indices.create(
                index=self.index_name,
                settings=index_settings["settings"],
                mappings=index_settings["mappings"],
            )
            print(f"Created index: {self.index_name}")

    def index_documents(self, documents: list[dict[str, Any]]):
        actions = [
            {
                "_index": self.index_name,
                "_source": {
                    "content": doc["original_content"],
                    "contextualized_content": doc["contextualized_content"],
                    "doc_id": doc["doc_id"],
                    "chunk_id": doc["chunk_id"],
                    "original_index": doc["original_index"],
                },
            }
            for doc in documents
        ]
        success, _ = bulk(self.es_client, actions)
        self.es_client.indices.refresh(index=self.index_name)
        return success

    def search(self, query: str, k: int = 20) -> list[dict[str, Any]]:
        self.es_client.indices.refresh(index=self.index_name)

        response = self.es_client.search(
            index=self.index_name,
            query={
                "multi_match": {
                    "query": query,
                    "fields": ["content", "contextualized_content"],
                }
            },
            size=k,
        )

        return [
            {
                "doc_id": hit["_source"]["doc_id"],
                "original_index": hit["_source"]["original_index"],
                "content": hit["_source"]["content"],
                "contextualized_content": hit["_source"]["contextualized_content"],
                "score": hit["_score"],
            }
            for hit in response["hits"]["hits"]
        ]


# =============================================================================
# ONLY THIS FUNCTION CHANGED — everything above is identical to original
# =============================================================================

def create_elasticsearch_bm25_index(db):
    """
    Build an ElasticsearchBM25 index from a ContextualChromaDB instance.

    The old version accessed db.metadata directly (pickle-based flat list).
    ContextualChromaDB stores everything inside ChromaDB, so we pull it out
    via collection.get() and reshape it into the same flat list format that
    index_documents() expects.
    """
    es_bm25 = ElasticsearchBM25()

    # ── Pull all stored data out of ChromaDB ──────────────────────────────────
    # collection.get() returns parallel lists:
    #   ids       → ["chunk_0", "chunk_1", ...]
    #   metadatas → [{"doc_id": ..., "original_content": ..., ...}, ...]
    raw = db.collection.get(include=["metadatas"])

    # ── Reshape into flat list of dicts that index_documents() expects ────────
    # Each dict needs: original_content, contextualized_content,
    #                  doc_id, chunk_id, original_index
    # All of these are already stored as metadata fields in ChromaDB —
    # we just zip ids + metadatas together into one list of dicts.
    documents = [
        {
            "original_content":       meta["original_content"],
            "contextualized_content": meta["contextualized_content"],
            "doc_id":                 meta["doc_id"],
            "chunk_id":               chunk_id,
            "original_index":         meta["original_index"],
        }
        for chunk_id, meta in zip(raw["ids"], raw["metadatas"])
    ]

    print(f"Extracted {len(documents)} chunks from ChromaDB for Elasticsearch indexing.")

    es_bm25.index_documents(documents)
    return es_bm25


# =============================================================================
# retrive with reranking + semantic contextual chunking
# =============================================================================


def retrieve_with_rerank(
    query: str,
    db: ContextualChromaDB,
    k: int = 20,
    cohere_api_key: str | None = None,
) -> list[dict[str, Any]]:
    """
    Two-stage retrieval:
      Stage 1 — broad semantic search: retrieve k*10 candidates from ChromaDB
      Stage 2 — Cohere reranker: score all candidates, return top-k

    Output format is identical to db.search() so retrieve_advanced()
    needs zero changes when swapping db.search() for this function.
    """
    if cohere_api_key is None:
        cohere_api_key = os.getenv("COHERE_API_KEY")

    co = cohere.Client(cohere_api_key)

    # ── Stage 1: Over-retrieve (cast a wide net) ──────────────────────────────
    candidates = db.search(query, k=k * 10)

    # ── Prepare documents for Cohere (original + context combined) ────────────
    documents = [
        f"{res['metadata']['original_content']}\n\nContext: {res['metadata']['contextualized_content']}"
        for res in candidates
    ]

    # ── Stage 2: Rerank ───────────────────────────────────────────────────────
    rerank_response = co.rerank(
        model="rerank-english-v3.0",
        query=query,
        documents=documents,
        top_n=k,
    )

    # ── Map reranked results back to original db.search() format ─────────────
    reranked_results = []
    for r in rerank_response.results:
        original = candidates[r.index]   # map back using Cohere's index
        reranked_results.append({
            "metadata":   original["metadata"],
            "document":   original["document"],
            "distance":   original["distance"],
            # replace similarity with Cohere's relevance score (0→1)
            "similarity": round(r.relevance_score, 4),
        })

    return reranked_results



# =============================================================================
# retrieve_advanced — unchanged from original
# =============================================================================

def retrieve_advanced(
    query: str,
    db,
    es_bm25: ElasticsearchBM25,
    k: int,
    semantic_weight: float = 0.8,
    bm25_weight: float = 0.2,
):
    num_chunks_to_recall = 75  # 150

    # Semantic search
    semantic_results = db.search(query, k=num_chunks_to_recall)
    ranked_chunk_ids = [
        (result["metadata"]["doc_id"], result["metadata"]["original_index"])
        for result in semantic_results
    ]

    # BM25 search using Elasticsearch
    bm25_results = es_bm25.search(query, k=num_chunks_to_recall)
    ranked_bm25_chunk_ids = [
        (result["doc_id"], result["original_index"]) for result in bm25_results
    ]

    # Combine results
    chunk_ids = list(set(ranked_chunk_ids + ranked_bm25_chunk_ids))
    chunk_id_to_score = {}

    # Initial scoring with weights
    for chunk_id in chunk_ids:
        score = 0
        if chunk_id in ranked_chunk_ids:
            index = ranked_chunk_ids.index(chunk_id)
            score += semantic_weight * (1 / (index + 1))
        if chunk_id in ranked_bm25_chunk_ids:
            index = ranked_bm25_chunk_ids.index(chunk_id)
            score += bm25_weight * (1 / (index + 1))
        chunk_id_to_score[chunk_id] = score

    # Sort chunk IDs by their scores in descending order
    sorted_chunk_ids = sorted(
        chunk_id_to_score.keys(), key=lambda x: (chunk_id_to_score[x], x[0], x[1]), reverse=True
    )

    # Assign new scores based on sorted order
    for index, chunk_id in enumerate(sorted_chunk_ids):
        chunk_id_to_score[chunk_id] = 1 / (index + 1)

    # ── Pull full metadata from ChromaDB for final results ────────────────────
    # Old code used db.metadata (in-memory list).
    # We fetch from ChromaDB collection instead.
    raw_all = db.collection.get(include=["metadatas"])
    all_metadata = [
        {**meta, "chunk_id": cid}
        for cid, meta in zip(raw_all["ids"], raw_all["metadatas"])
    ]

    # Prepare final results
    final_results = []
    semantic_count = 0
    bm25_count = 0

    for chunk_id in sorted_chunk_ids[:k]:
        '''
            we are looking :
            1. document(doc_id)
            2. original_index 
            if both matches then store all metadata to chunk_metadata by (all_chunks ---> chunk) using ✅ nex() [lookup]
        '''
        chunk_metadata = next(
            chunk
            for chunk in all_metadata
            if chunk["doc_id"] == chunk_id[0] and chunk["original_index"] == chunk_id[1]
        )
        is_from_semantic = chunk_id in ranked_chunk_ids
        is_from_bm25 = chunk_id in ranked_bm25_chunk_ids
        final_results.append(
            {
                "chunk": chunk_metadata,
                "score": chunk_id_to_score[chunk_id],
                "from_semantic": is_from_semantic,
                "from_bm25": is_from_bm25,
            }
        )

        if is_from_semantic and not is_from_bm25:
            semantic_count += 1
        elif is_from_bm25 and not is_from_semantic:
            bm25_count += 1
        else:
            semantic_count += 0.5
            bm25_count += 0.5

    return final_results, semantic_count, bm25_count

# =============================================================================
# Main — quick smoke test
# =============================================================================

if __name__ == "__main__":


    # ── Load dataset produced by DocumentProcessor ────────────────────────────
    with open("../data/codebase_chunks.json", encoding="utf-8") as f:
        dataset = json.load(f)

    # ── Build contextual ChromaDB ─────────────────────────────────────────────
    db = ContextualChromaDB(
        name        = "rag_research",
        persist_dir = "../data/chroma",
    )

    db.load_data(dataset, parallel_threads=5)

    queries = [
        # Q1 — Simple definition: single concept, expects a direct factual answer
        "What is Retrieval-Augmented Generation?",

        # Q2 — Mechanism: how a component works, expects a process-level chunk
        "How does the retriever component select relevant documents in a RAG pipeline?",

        # Q3 — Comparison: two distinct approaches contrasted across the paper
        "What are the differences between sparse retrieval methods like BM25 "
        "and dense retrieval methods like DPR in the context of RAG?",

        # Q4 — Critical/analytical: requires understanding trade-offs discussed
        # across multiple sections — tests whether contextual chunking surfaces
        # the right cross-cutting content
        "What are the key limitations and open research challenges of current "
        "RAG systems when applied to knowledge-intensive tasks, and what "
        "architectural improvements does the survey propose to address them?",
    ]

    
    for i, query in enumerate(queries, start=1):
        print(f"\n{'=' * 60}")
        print(f"  Q{i}: {query}")
        print(f"{'=' * 60}")

        # ── Quick search smoke test ───────────────────────────────────────────────
        results = db.search(query, k=5)

        print("\n── Top 5 results ──────────────────────────────────────────────")
        for i, r in enumerate(results, 1):
            print(f"\n[{i}] similarity={r['similarity']:.4f}  chunk={r['metadata']['chunk_id']}")
            print(f"     original:      {r['metadata']['original_content']}...")
            print(f"     contextualized:{r['metadata']['contextualized_content']}...")

        # ── Debug: search within a specific document ──────────────────────────────
        print("\n── Filtered search (doc_id='paper1') ──────────────────────────")
        filtered = db.search(
            "how does DPR handle out-of-domain retrieval?",
            k=5,
            doc_id_filter="paper1",
        )
        for i, r in enumerate(filtered, 1):
            print(f"[{i}] similarity={r['similarity']:.4f}  chunk={r['metadata']['chunk_id']}")


        # Step 1 — your ChromaDB should already be loaded (you did this earlier)
    db = ContextualChromaDB(
        name        = "rag_research",
        persist_dir = "../data/chroma",
    )

    # Step 2 — create the Elasticsearch index from your ChromaDB data
    es_bm25 = create_elasticsearch_bm25_index(db)

    # Step 3 — now call retrieve_advanced
    results, semantic_count, bm25_count = retrieve_advanced(
        query          = "What are the key limitations and open research challenges of current "
            "RAG systems when applied to knowledge-intensive tasks, and what "
            "architectural improvements does the survey propose to address them?",
        db             = db,
        es_bm25        = es_bm25,
        k              = 5,
        semantic_weight = 0.8,
        bm25_weight    = 0.2,
    )

    # Step 4 — print results
    print(f"Semantic contributed: {semantic_count}")
    print(f"BM25 contributed:     {bm25_count}")

    for i, r in enumerate(results, 1):
        print(f"\n[{i}] score={r['score']:.4f}  from_semantic={r['from_semantic']}  from_bm25={r['from_bm25']}")
        print(f"     {r['chunk']['original_content']}")