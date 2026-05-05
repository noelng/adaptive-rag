# Adaptive RAG

An agentic Retrieval-Augmented Generation (RAG) system built with [LangGraph](https://github.com/langchain-ai/langgraph), [LangChain](https://github.com/langchain-ai/langchain), **Gemma 3** as the LLM, and **ChromaDB** as the vector store. The system dynamically routes questions between a local knowledge base and web search, grades retrieved documents for relevance, and validates generated answers for hallucinations — retrying automatically up to a configurable limit.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Setup](#setup)
- [Usage](#usage)
- [API Reference](#api-reference)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)

---

## Overview

Adaptive RAG Chatbot improves on naive RAG by adding a multi-step reasoning loop around retrieval and generation:

1. **Route** — Decides whether the question is best answered from the local vector store or a live web search.
2. **Retrieve** — Fetches relevant documents from ChromaDB.
3. **Grade documents** — Filters out irrelevant documents; triggers web search as a fallback.
4. **Generate** — Produces an answer with Gemma 3.
5. **Grade generation** — Checks for hallucinations and answer quality. Retries up to `MAX_RETRIES` times before exiting gracefully.

---

## Features

- Adaptive routing between vector store and web search
- Document relevance grading before generation
- Hallucination detection with configurable retry limit (no infinite loops)
- Structured outputs via Pydantic for all graders
- ChromaDB for local, persistent vector storage
- Gemma 3 as the local LLM backbone
- Clean LangGraph state machine — easy to extend

---

## Architecture

```
User Question
      │
      ▼
 Route Question
  ┌───┴────┐
  │        │
Retrieve  Web Search ◄─────────────────────┐
  │        │                               │
  ▼        │                               │
Grade    ──┘                               │
Documents                                  │
  │                                        │
  ├── not relevant ──► Web Search          │
  │                                        │
  └── relevant ──► increment_retry         │
                         │                 │
                         ▼                 │
                      Generate             │
                         │                 │
                         ▼                 │
               Grade Generation            │
                 ┌────┬──────┐             │
                 │    │      │             │
               useful │  not supported    │
                 │  not  │ (hallucination) │
                 ▼ useful└───────────────►─┘
                END   │
                      ▼
                  Web Search
                      │
                      ▼
               increment_retry
                      │
                      ▼
                   Generate ...
                      │
               (max retries hit)
                      ▼
                     END
```

### Graph nodes

| Node | Description |
|---|---|
| `route_question` | Entry point — routes to vectorstore or web search |
| `retrieve` | Fetches documents from ChromaDB |
| `grade_documents` | Filters documents by relevance to the question |
| `web_search` | Runs a live web search as fallback |
| `increment_retry` | Bumps `retry_count` in state before every generation |
| `generate` | Calls Gemma 3 to produce an answer |
| `grade_generation` | Checks hallucination and answer quality; enforces retry limit |

---

## Project Structure

```
adaptive_rag/
├── src/                    # Source code
│   ├── workflow/          # Core workflow logic
│   │   ├── chains/       # LLM processing chains
│   │   │   ├── answer_grader.py
│   │   │   ├── generation.py
│   │   │   ├── hallucination_grader.py
│   │   │   ├── retrieval_grader.py
│   │   │   └── router.py
│   │   ├── nodes/        # Workflow nodes
│   │   │   ├── generate.py
│   │   │   ├── grade_documents.py
│   │   │   ├── retrieve.py
│   │   │   └── web_search.py
│   │   ├── consts.py     # Node constants
│   │   ├── graph.py      # Main workflow orchestration
│   │   └── state.py      # State management
│   ├── cli/              # Command line interface
│   │   └── main.py       # Interactive CLI
│   └── models/           # Model configurations
│       └── model.py      # LLM and embedding models
├── data/                 # Data processing
│   └── ingestion.py      # Document ingestion and vector store
├── tests/                # Test files
│   └── test_chains.py    # Chain testing suite
├── .env                  # Environment variables
├── .gitignore
├── main.py              # Application entry point
├── README.md
└── requirements.txt
```

---

## Setup

### Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com/) running locally with Gemma 3 pulled
- A [Tavily](https://tavily.com/) API key for web search (free tier available)

### 1. Clone the repository

```bash
git clone https://github.com/your-username/adaptive-rag-chatbot.git
cd adaptive-rag-chatbot
```

### 2. Create and activate a virtual environment

```bash
python -m venv adaptive_rag
source adaptive_rag/bin/activate        # Windows: .venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Pull Gemma 3 via Ollama

```bash
ollama pull gemma3
```

### 5. Configure environment variables

Create a `.env` file in the project root:

```env
GOOGLE_API_KEY=your-api-key-here

NEO4J_URI=neo4j+s://4bd54668.databases.neo4j.io
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your-password-here

TAVILY_API_KEY=your-api-key-here

LANGCHAIN_API_KEY=your-api-key-here
LANGCHAIN_TRACING_V2=true
LANGCHAIN_ENDPOINT=https://api.smith.langchain.com 
LANGCHAIN_PROJECT=agentic-rag

USER_AGENT=adaptive-rag/0.1
```

### 6. Ingest your documents into ChromaDB

```bash
python data/ingest.py --source ./docs
```

---

## Usage

### Run as a script

```python
from src.workflow.graph import app

result = app.invoke({
    "question": "What is the capital of France?",
    "retry_count": 0,
})

print(result["generation"])
```

### Run interactively

```bash
python main.py
```

Then type your question at the prompt:

```
> What is retrieval-augmented generation?
---ROUTE QUESTION---
---RETRIEVE---
---CHECK DOCUMENT RELEVANCY COMPARED TO THE QUESTION---
---GRADE: RELEVANT---
---GENERATE---
---CHECK HALLUCINATIONS---
---DECISION: GENERATION IS GROUNDED---
---DECISION: GENERATION IS USEFUL---

Answer: Retrieval-augmented generation (RAG) is ...
```

### Stream output token by token

```python
from src.workflow.graph import app

for chunk in app.stream({
    "question": "Explain adaptive RAG",
    "retry_count": 0,
}):
    for node, output in chunk.items():
        print(f"[{node}]", output.get("generation", ""))
```

---

## API Reference

### `GraphState`

The shared state passed between all nodes.

```python
class GraphState(TypedDict):
    question:     str        # The user's input question
    generation:   str        # The LLM's generated answer
    web_search:   bool       # Whether web search is needed
    documents:    List[str]  # Retrieved/filtered documents
    retry_count:  int        # Current loop iteration count
```

Always initialise `retry_count` to `0` when invoking the graph:

```python
app.invoke({"question": "...", "retry_count": 0})
```

---

### `answer_grader`

Grades whether the LLM's generation actually addresses the user's question.

```python
from src.workflow.chains.answer_grader import answer_grader

result = answer_grader.invoke({
    "question": "What is RAG?",
    "generation": "RAG stands for Retrieval-Augmented Generation ...",
})
print(result.binary_score)  # True / False
```

| Input field | Type | Description |
|---|---|---|
| `question` | `str` | The original user question |
| `generation` | `str` | The LLM-generated answer to evaluate |

| Output field | Type | Description |
|---|---|---|
| `binary_score` | `bool` | `True` if the answer resolves the question |

---

### `hallucination_grader`

Grades whether the generation is grounded in the retrieved documents.

```python
from src.workflow.chains.hallucination_grader import hallucination_grader

docs_text = "\n\n".join(d.page_content for d in documents)

result = hallucination_grader.invoke({
    "documents": docs_text,
    "generation": "The Eiffel Tower is in Paris ...",
})
print(result.binary_score)  # True / False
```

| Input field | Type | Description |
|---|---|---|
| `documents` | `str` | Concatenated plain-text content of retrieved documents |
| `generation` | `str` | The LLM-generated answer to evaluate |

| Output field | Type | Description |
|---|---|---|
| `binary_score` | `bool` | `True` if the answer is grounded in the documents |

> **Important:** Always pass `documents` as a plain concatenated string, not as a list of `Document` objects. Passing raw objects causes the grader to always return `False`.

---

### `retrieval_grader`

Grades whether a single retrieved document is relevant to the question.

```python
from src.workflow.chains.retrieval_grader import retrieval_grader

result = retrieval_grader.invoke({
    "question": "What is RAG?",
    "document": "RAG combines retrieval with generation ...",
})
print(result.binary_score)  # True / False
```

| Input field | Type | Description |
|---|---|---|
| `question` | `str` | The user question |
| `document` | `str` | The `page_content` of a single retrieved document |

| Output field | Type | Description |
|---|---|---|
| `binary_score` | `bool` | `True` if the document is relevant |

---

### `question_router`

Routes the question to either the vector store or web search.

```python
from src.workflow.chains.router import question_router

result = question_router.invoke({"question": "Latest news on LLMs?"})
print(result.datasource)  # "websearch" or "vectorstore"
```

| Input field | Type | Description |
|---|---|---|
| `question` | `str` | The user question |

| Output field | Type | Description |
|---|---|---|
| `datasource` | `str` | `"websearch"` or `"vectorstore"` |

---

### `app.invoke()`

Run the full adaptive RAG pipeline synchronously.

```python
result = app.invoke({
    "question": str,      # required
    "retry_count": int,   # required — always pass 0
})
```

Returns the final `GraphState` dict. Access the answer via `result["generation"]`.

---

### `app.stream()`

Run the pipeline and receive node-by-node output chunks.

```python
for chunk in app.stream({"question": "...", "retry_count": 0}):
    for node_name, node_output in chunk.items():
        print(node_name, node_output)
```

---

## Configuration

| Variable | Default | Description |
|---|---|---|
| `MAX_RETRIES` | `3` | Max generate → grade loop iterations before giving up |
| `CHROMA_PERSIST_DIR` | `./chroma_db` | ChromaDB persistence directory |
| `CHROMA_COLLECTION_NAME` | `rag_documents` | ChromaDB collection name |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `TAVILY_API_KEY` | — | Tavily API key for web search |

---

## Troubleshooting

### Generation loops more than `MAX_RETRIES` times

Make sure `retry_count` is initialised to `0` in your `app.invoke()` call, and that all paths through the `grade_generation` node route through `increment_retry` before looping back to `generate` — including the `"not supported"` (hallucination) path.

### `docs_text` is always empty

`documents` is an empty list — everything was filtered out by `grade_documents`. Add logging inside `grade_documents` to check:
- `INCOMING DOCS COUNT` — if `0`, the retrieval node returned nothing from ChromaDB.
- `FILTERED DOCS COUNT` — if `0` but incoming was non-zero, the retrieval grader is rejecting all documents. Check the `binary_score` type (`bool` vs `str`) in your `GradeDocuments` Pydantic model.

### Hallucination grader always returns `False`

The two most common causes:
1. You are passing `Document` objects instead of plain text to the grader. Always concatenate with `"\n\n".join(d.page_content for d in documents)`.
2. The system prompt is too strict. Use a lenient prompt that accepts paraphrasing and inference, not just verbatim quoting.

### Ollama / Gemma 3 not responding

Verify Ollama is running and the model is available:

```bash
ollama list          # should show gemma3
ollama run gemma3    # quick smoke test
```

### ChromaDB returns no results

Check that your ingestion script ran successfully and the collection name in `.env` matches the one used during ingestion.

---

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature-name`
3. Commit your changes with clear messages: `git commit -m "feat: add streaming support"`
4. Push to your fork: `git push origin feature/your-feature-name`
5. Open a Pull Request against `main`

Please make sure your code passes linting before submitting:

```bash
pip install ruff
ruff check .
```

---

## License

MIT License. See [LICENSE](LICENSE) for details.
