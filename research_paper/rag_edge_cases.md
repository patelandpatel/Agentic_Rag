# Edge Cases to Consider When Building a RAG Pipeline

This document covers every function in the contextual retrieval pipeline and the real-world edge cases that can silently break your code or produce bad results. No code — just the reasoning and decisions behind each safeguard.

---

## 1. PDF Text Extraction (`extract_text_from_pdf`)

### Scanned / Image-Based PDFs
The most common silent failure in any document pipeline. If a PDF was created by scanning physical pages, the file contains images of text, not actual text characters. `PyMuPDF` (fitz) will return empty strings for every page, and the rest of your pipeline will run on nothing — producing zero chunks with no error message. Always check if the returned page list is empty and raise an explicit error pointing the user toward OCR tools.

### Partially Scanned PDFs
Some PDFs are a mix — the first 10 pages are real text, the last 20 are scanned appendices. Your extractor will succeed and return something, but you silently lose a large portion of the document. It is worth logging which page numbers returned empty text so the user knows the extraction was incomplete.

### Password-Protected PDFs
Attempting to open an encrypted PDF without providing the password raises a library exception. If your pipeline processes folders of PDFs automatically, one locked file will crash the whole batch. Wrap per-file processing in a try/except at the folder level.

### Corrupt or Malformed PDF Files
PDFs with broken internal structure can cause the library to hang, throw unexpected errors, or return garbage text. Always catch exceptions at the file level rather than letting one bad file kill a multi-file run.

### Special Characters and Encoding Issues
PDFs that were converted from other formats (Word, LaTeX) sometimes contain unusual Unicode characters, ligatures (ﬁ, ﬂ), or encoding artifacts. These can corrupt your sentence splitting and produce tokens that confuse the embedding model. It is worth normalising unicode before further processing.

---

## 2. Sentence Splitting (`split_into_sentences`)

### Abbreviations Triggering False Splits
Simple punctuation-based splitting breaks on abbreviations like "Dr.", "Fig.", "et al.", "U.S.", "vs." — treating them as sentence endings. Academic and legal documents are full of these. The resulting fragments are semantically meaningless and produce noisy embeddings. Consider using a proper sentence tokeniser (spaCy, NLTK punkt) for production use.

### Empty or Whitespace-Only Strings
After splitting, some elements may be empty strings or contain only whitespace. Passing these to an embedding model produces near-zero vectors. In cosine similarity, a near-zero vector will compute as `NaN` or behave unpredictably. Always filter out any string that contains no alphanumeric characters.

### Symbol-Only Lines
Academic papers frequently contain page headers, footers, figure captions like "— — —", table borders made of dashes, or LaTeX artifacts. These produce meaningless embeddings and add noise to your chunk boundary detection. Filter any string that contains no real words.

### Very Long "Sentences"
Some documents contain run-on text with no punctuation — entire paragraphs as one sentence, or code blocks, or citation lists. A single such string can exceed the embedding API's token limit. You need to handle truncation or secondary splitting on these.

### Line Breaks vs. Sentence Breaks
PDFs often have line breaks mid-sentence due to column layouts or hyphenation. Raw `fitz` text extraction will include these as separate fragments that split_into_sentences treats as multiple short sentences. This creates many tiny, context-free chunks. Pre-processing to join hyphenated words and rejoin mid-sentence line breaks significantly improves quality.

---

## 3. OpenAI Embeddings (`get_embeddings`)

### Input Count Limit Per Request
OpenAI's embeddings endpoint accepts a maximum of 2048 inputs per API call. Sending all sentences from a long document in a single call will return a `400 BadRequestError` with a confusing message about `$.input`. Always batch your inputs, with 500 per batch being a safe conservative limit.

### Per-Input Token Limit
Each individual string sent to the embeddings API has a hard limit of 8191 tokens for `text-embedding-3-small`. Academic papers can produce sentences far exceeding this — long footnotes, citation blocks, equation-heavy paragraphs. A single oversized input in a batch fails the entire batch. Truncate any input that exceeds the limit before sending.

### Newlines Degrading Embedding Quality
OpenAI's own documentation recommends replacing newline characters with spaces before embedding. Newlines in the middle of text can affect how the model tokenises the input and reduce embedding quality. Always clean inputs before sending.

### Rate Limits
If you are embedding thousands of sentences, you will eventually hit OpenAI's rate limits (requests per minute or tokens per minute). Without retry logic, the call raises an exception and you lose all progress. Implement exponential backoff retries.

### Cost Awareness
Each API call has a cost. For large document collections, the total token count across all sentences can be significant. It is worth estimating cost upfront: count total characters, divide by 4 for rough token estimate, multiply by the model's per-token price.

---

## 4. Cosine Similarity (`cosine_similarity`)

### Zero Vectors Producing NaN
If either input vector is all zeros (or near-zero), the denominator in the cosine formula becomes zero, and the result is `NaN` or raises a division error. Zero vectors can come from: empty strings that slipped through filtering, strings of pure symbols, or rare degenerate embedding outputs. An explicit guard that returns `0.0` when the denominator is near-zero is essential. Returning `0.0` is the safest choice — it forces a chunk boundary, which is better than propagating `NaN` into similarity comparisons.

### Identical Vectors Producing Exactly 1.0
Not a bug, but worth noting: if two consecutive sentences are identical (e.g. a repeated header), similarity will be exactly 1.0 and they will always be merged into the same chunk regardless of threshold. This is usually the correct behaviour.

### Threshold Sensitivity
A similarity threshold of 0.75 is a reasonable starting point but is highly domain-dependent. Dense academic text has high sentence-to-sentence similarity throughout; a threshold of 0.75 may produce very few chunk breaks and create huge chunks. Conversational or narrative text varies more; 0.75 may break too aggressively. Always tune the threshold against your specific document type.

---

## 5. Semantic Chunking (`semantic_chunk_pdf`)

### Minimum Chunk Size Dropping Real Content
A `min_chunk_sentences` guard of 2 or higher will silently drop any buffer that ends with fewer sentences — for example, the last section of a document, a short conclusion paragraph, or a single-sentence abstract. This content disappears without any warning. Setting the minimum to 1 or removing the minimum entirely and always flushing is safer.

### The Last Chunk Always Being Lost
A common bug: the loop flushes a chunk whenever a boundary is detected, but the final sentences after the last boundary detection never trigger a flush because there is no "next sentence" to compare against. The tail of every document is silently dropped. Always force-flush whatever remains in the buffer after the loop ends.

### Chunk Size vs. Retrieval Quality
The blog recommends "a few hundred tokens" per chunk. Chunks that are too small (under ~50 tokens) lack enough context to be semantically meaningful on their own. Chunks that are too large (over ~400 tokens) contain multiple topics and dilute retrieval precision — the embedding averages over too many concepts. The sweet spot for academic papers is roughly 150–256 tokens.

### Chunk Overlap
The current implementation has no overlap between chunks. A sentence at a chunk boundary belongs entirely to one chunk or the other. If that sentence is critical context for the topic that follows, the next chunk starts without it. Adding a small overlap (1–2 sentences repeated at the start of the next chunk) can improve retrieval for boundary-spanning topics. This is a known trade-off between storage efficiency and retrieval quality.

---

## 6. Claude Contextual Enrichment (`add_contextual_retrieval`)

### Whole Document Exceeding Claude's Context Window
Passing the full text of a 200-page paper to Claude for every single chunk will exceed the model's context window. Claude Haiku supports up to 200k tokens, but very large documents can still hit this limit when combined with the prompt template. Truncate the whole document to a safe character limit before using it as context. You will lose some context for the final chunks, but this is better than a hard failure.

### Rate Limits on Large Documents
Calling Claude in a tight loop for every chunk — potentially hundreds of calls — will trigger API rate limits. Without retry logic, the call fails mid-pipeline and you lose all the enrichment work done so far. Exponential backoff (wait 1s, then 2s, then 4s on subsequent retries) handles transient rate limit errors gracefully.

### No Progress Visibility
Enriching 200 chunks sequentially with no feedback looks identical to a hung process. A simple progress counter ("Chunk 47/200 enriched") lets you confirm the pipeline is running and estimate time remaining.

### Failed Enrichment Handling
If all retries fail for a chunk (e.g. persistent rate limiting, network error), the code should not crash. Falling back to the raw chunk text without context is far better than losing the chunk entirely or stopping the pipeline. Log the failure clearly so you can re-run enrichment for that chunk later.

### Context Window Allocation
The prompt template itself uses tokens. For very large documents, even after truncation, the prompt (document + chunk + instructions) may push against limits. The chunk text itself also consumes context. Make sure your truncation limit accounts for the prompt overhead and the chunk size, not just the document text alone.

### Prompt Faithfulness
The exact wording of the contextualisation prompt matters. The Anthropic blog's prompt is specific: it asks for a "short succinct context" and says "answer only with the succinct context and nothing else." Deviating from this (e.g. asking for a summary, or not adding the "nothing else" constraint) will produce longer, less focused context that adds noise rather than signal to retrieval.

---

## 7. Folder Processing (`process_pdf_folder`)

### One Bad File Crashing the Whole Batch
If you are processing a folder of 50 PDFs and the 23rd is scanned or corrupt, an unhandled exception will crash the entire run and you lose all remaining files. Wrap per-file logic in a try/except that logs the failure and continues to the next file.

### Empty Folder
If the folder path is wrong or contains no PDFs, the glob returns an empty list and the rest of the code runs on nothing — possibly printing misleading "0 documents processed" output with no error. An explicit check for an empty file list with a clear error message saves debugging time.

### Memory Accumulation
Processing many large PDFs sequentially without clearing results can accumulate significant memory. For very large collections, consider processing and saving results to disk per-document rather than holding all chunks in memory until the end.

### Output Naming Collisions
If two PDFs in the folder have the same stem (e.g. `paper.pdf` in two subfolders, or two files with the same name), the results dictionary will silently overwrite the first with the second. Using the full path as the key instead of just the file stem prevents this.

---

## General Principles

**Fail loudly, not silently.** Silent failures (returning empty lists, skipping content without logging) are the hardest bugs to diagnose. Raise exceptions or log warnings whenever something unexpected happens.

**Separate concerns.** Batching is a transport concern (how you talk to the API). Chunking is a content concern (how you split your document). Contextualisation is a semantic concern (how you enrich each chunk). Keeping these cleanly separated makes each step independently testable and replaceable.

**Always tune on your data.** Similarity thresholds, chunk sizes, batch sizes, and truncation limits are not universal constants. They are starting points that should be validated against your specific document type, language, and retrieval task.

**Cost is a concern at scale.** Every embedding call, every Claude API call, has a cost. For large document collections, estimate cost before running. Use prompt caching (as the blog describes) to reduce the cost of passing the whole document to Claude for every chunk.
