# KernelMind Retrieval Engine

KernelMind is a retrieval and reasoning system built to analyze large codebases and answer technical questions based on actual source code. The focus is on structured indexing, multi-step retrieval, and grounded answer generation.

Everything runs **fully self-hosted** right now.
Ollama handles the models locally, the vector database runs on my machine, and yes — it has definitely pushed my CPU/GPU harder than they deserve. I’m considering adding optional cloud support later so this setup is easier to run and doesn't heat the room.


## Features
### 1. **Multi-Stage Retrieval**

KernelMind layers several retrieval strategies to improve precision:
   - Embedding search (Qwen embeddings + Chroma)
   - BM25 keyword scoring
   - Type and domain-based boosting
   - Symbol-aware ranking
   - Call-chain expansion
   - Cross-encoder reranking with MiniLM
   - Chunk summarization before answer generation

The goal is to reduce noise and surface the most relevant code, even in large or complex repositories.

### 2. Structured Chunking

Repositories are split into meaningful units:
   - Functions
   - Classes
   - Methods
   - Logical blocks
   - Imports
   - Comments and docstrings

This improves retrieval quality compared to indexing entire files.

### 3. Batch-Safe Vector Insertion

I discovered pretty quickly that Chromadb has a limit on batch size, so my indexing pipeline uses safe batches automatically: 
It adds vectors in safe batches. Chroma cannot handle > ~5461 items per batch.

### 4. Supported Languages

KernelMind currently parses:
   - Python
   - JavaScript
   - TypeScript
   - JSON
   - YAML

Additional languages can be added by extending the chunking layer.

---
### Multi-Stage Retrieval (Detailed Overview)

KernelMind doesn't depend on a single retrieval signal. Codebases contain structure, naming conventions, and interconnected logic, so the system combines several techniques to narrow down the most relevant pieces of code.

Here is the full retrieval pass.

---

### 1. Embedding Search (Qwen + Chroma)

The query is embedded using Qwen embeddings.
Chroma returns its top-K vector matches.

This step handles semantic similarity well — for example, questions about logic, behavior, or “what happens when X is called.”
However, embeddings alone aren’t enough for precise identifiers or short names.

---

### 2. BM25 Keyword Retrieval

BM25 runs in parallel with embeddings.

It is particularly effective at:

- exact function or variable names
- file or module names
- short tokens
- error messages
- keywords

BM25 results are scored and kept for later merging.

---

### 3. Type and Domain-Based Boosts

Some chunks are more useful than others depending on the query.

KernelMind applies boosts based on:

- code type (functions, classes, logic blocks over comments)
- file roles (e.g., auth, middleware, configuration)
- symbol matches extracted from the query

If the query references a specific function or class, chunks related to that symbol get additional weight.

---

### 4. Call-Chain Expansion

When a symbol is detected — for example, a function name — KernelMind expands the context automatically:

- callers
- callees
- adjacent helper methods
- parent or child classes
- related modules

This helps surface connected logic that embeddings or keywords might miss.

---

### 5. Combined Scoring

All signals are merged into a single relevance score:

- embedding similarity
- BM25 score
- type/domain boosts
- symbol matches
- call-chain relevance

This produces a ranked list of candidate chunks.
But to improve precision, one more step is used.

---

### 6. Cross-Encoder Reranking (MiniLM-L-6-v2)

The top candidates are passed through a cross-encoder.
Unlike embeddings, the cross-encoder reads the query and chunk together and evaluates their relevance directly.

This model resolves many of the mistakes that embedding search and BM25 might introduce, and usually gives the cleanest final ordering.

---

### 7. Dedupe, Cleanup, and Ordering

After reranking:

- near-duplicate chunks are removed
- large unhelpful sections are filtered
- chunks from the same file are grouped
- the final list is sorted by combined relevance

This prepares the context for summarization.

---

### 8. Summarization Layer

Each selected chunk is summarized before being sent to the LLM.
The summaries keep the important logic while reducing noise.

This helps prevent hallucinations and keeps the answer grounded in the retrieved evidence.

---

### 9. Answer Generation (Qwen 2.5 Code via Ollama)

Finally, the summarized chunks, the raw chunks, and the original query are given to a Qwen Code model running via Ollama.

The model produces an answer based strictly on the retrieved context.

---
### Model Choices

Several models were tested during development:

- Gemma2 9B — capable but slow for long contexts
- DeepSeek Code Lite — decent reasoning, not ideal for iterative retrieval
- Qwen 2.5 Code — current choice  
  Strong balance of reasoning speed and code understanding

Embeddings use Qwen embeddings.
Reranking uses MiniLM-L-6-v2.

---

### Current Capabilities

- Works across Python, JS, TS, JSON, YAML
- Handles full repo analysis
- Understands symbols and code structure
- Expands context using call-chains
- Generates grounded, evidence-based answers
- Fully offline, self-hosted retrieval + reasoning
- Stable on very large repositories via safe batching

---

### Planned Improvements (Feel Free to add PRs!)

- Faster BM25 implementation
- Better noise filtering
- Stronger deduplication
- Cached reranker results
- Region-aware chunking for JS/TS
- Improved YAML/JSON parsing
- Optional cloud mode so local hardware isn’t overloaded

---

### Future Cloud Mode

Right now everything runs locally.
A cloud option is planned to make it easier to try the system without setting up local models or burning CPU/GPU cycles.
