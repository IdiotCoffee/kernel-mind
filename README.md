
# KernelMind

KernelMind is a local, code-aware retrieval and reasoning engine for navigating and understanding large codebases.  
It works entirely from the command line, uses a hybrid retrieval pipeline (BM25 + embeddings + type-based boosts), and synthesizes final answers using a locally-hosted Qwen 2.5 Coder 14B model.

Think of it as a fast, CLI-first tool that can ingest a GitHub repo, index the code, and answer deep questions about how that code works.

---

## Features

### 🔍 Code-Aware Retrieval  
KernelMind performs multi-stage retrieval over the indexed repository:

1. **BM25 keyword scoring** (via `rank_bm25 == BM25Okapi`)
2. **Embedding search** using a locally-hosted ChromaDB instance
3. **Type-Based Boosting** to push more “meaningful” chunks up the ranking (functions > methods > classes > imports > files)

```python
TYPE_BOOST = {
    "function": 0.20,
    "method":   0.18,
    "class":    0.10,
    "import":   0.02,
    "file":     0.00,
    None:       0.00,
}
```

4. **Cross-Encoder Reranking** using  
   `BAAI/bge-reranker-base`  
   This reorders the final candidate list to return the most semantically relevant code chunks.

The hybrid scoring pipeline ensures that results are not only keyword-relevant, but structurally meaningful.

---

## 🧠 Local LLM — Qwen 2.5 Coder 14B  
KernelMind uses **Qwen 2.5 Coder 14B**, running locally via Ollama, to synthesize final answers.

### *Will Qwen 14B handle raw code chunks?*  
Yes — but with some nuance:

- 14B models can easily handle 4K–8K tokens of mixed natural language + code.  
- For huge chunks (hundreds of lines), you may need to either  
  - reduce chunk size  
  - or rely on *summarized* chunks rather than raw code.  

As of now, KernelMind sticks to structural chunking (file / class / method / import).  
Direct inclusion of **raw method bodies** is planned but must be tested for context-window stability.

---

## 📁 Ingestion & Parsing

### Current behavior (v0.1)
KernelMind currently parses **Python files only**.  
The ingestion pipeline:

1. `kernelmind ingest <repo_url>`
2. Repository is cloned locally
3. Files are filtered using:
   ```python
   EXCLUDE_DIRS = {
       "node_modules", "dist", "build", "__pycache__", ".git",
       ".idea", ".vscode", ".venv", ".env"
   }

   INCLUDE_DIRS = {
       ".py", ".js", ".ts", ".java", ".go",
       ".json", ".yaml", ".yml", ".toml", ".md"
   }
   ```
4. Only Python files are parsed, and the following chunks are extracted:
   - File-level summary
   - Class names + docstrings
   - Method names + parents  
   - Function names + docstrings
   - Imports

*(Raw method bodies: optional and experimental — may push Qwen context window too far. This is being tested.)*

5. Structural metadata is embedded & persisted in **local ChromaDB**.

---

## 📐 Retrieval Flow (Full Pipeline)

Here’s how `kernelmind answer` actually works internally:

1. User runs:
   ```
   kernelmind answer "How does request throttling work?" --repo my_repo
   ```

2. Query is embedded and compared against:
   - BM25 keyword scored chunks  
   - Vector similarity chunks  
   - Type-boosted scores  
   - Cross-encoder reranked final set  

3. Top-K chunks are selected.

4. Qwen 2.5 Coder 14B receives:
   - the user question  
   - the top-ranked chunks  
   - a synthesizer prompt  

5. Qwen produces a structured, explanation-style answer.  
   Chunks are NOT shown unless the user requests them (`kernelmind search --show`).

---

## 🛠️ CLI Usage

KernelMind is a CLI-first tool with short aliases:

| Command | Alias | Description |
|--------|--------|-------------|
| `kernelmind` | `km` | base command |
| `kernelmind ingest` | `km i` | Clone + index a repo |
| `kernelmind search` | `km s` | Run query + show retrieved chunks |
| `kernelmind answer` | `km a` | Run query + synthesize final answer |

### Ingest a repo
```
km ingest https://github.com/someproject/somerepo
```

### Search (show raw chunks)
```
km search "how is authentication implemented?" --repo somerepo --show
```

### Full synthesized answer
```
km answer "how does the caching layer work?" --repo somerepo
```

---

## ⚙️ Requirements

KernelMind depends on:

- Python 3.10+
- Local ChromaDB instance (`chromadb==1.3.5`)
- Local LLM backend (Qwen 2.5 Coder 14B via Ollama)
- BM25 (`rank-bm25`)
- Cross-encoder reranker (`BAAI/bge-reranker-base`)

A full dependency list lives in `requirements.txt`.  
Some dependencies (e.g. Kubernetes clients, Uvicorn, HTTP frameworks) may be removed in future versions — these appear from earlier experiments and are not essential for v0.1.

Cleanup will happen before v0.2.

---

## 🚧 Roadmap

### Near-term (v0.2)
- JS/TS parser  
- Class/method signature extraction for Java & Go  
- Actual method-body chunking  
- Documentation chunking (README, Markdown, comments)  
- Improved token-aware chunk sizing for LLM stability  
- Configurable reranker weights

### Longer-term
- AST-aware cross-file call-graph generation  
- GitHub App integration for indexing private repos  
- Pluggable vector DB backends  
- Web UI (optional)

---

## 📝 Limitations (Honest List)

KernelMind v0.1 is functional but early:

- Only Python AST is parsed deeply.  
- JS/TS/Go/Java are recognized but not parsed.  
- Qwen 14B sometimes drops low-level code details if chunks are large.  
- Large repos (>20k LOC) may require memory tuning for embedding load.  
- No multi-file semantic stitching yet.  
- No context window optimization beyond naïve chunking.

These are intentional trade-offs for a small, local-first v0.1.

---

## 📦 Version
`v0.1.0` — first public release.

---

## 🤝 Contributing
Contributions for more language parsers, better chunking strategies, and UI tools are welcome.

---

## 📣 Final Notes

KernelMind is not just “another RAG script.”  
It is a carefully engineered, multi-stage code retrieval and reasoning pipeline optimized for **developer comprehension**, **local execution**, and **transparent retrieval logic**.

If you're working with large or unfamiliar codebases, this tool aims to make deep code questions answerable in seconds — right from your terminal.

