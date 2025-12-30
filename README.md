# Finance.AI - RAG Agent with ReAct

RAG (Retrieval-Augmented Generation) system specialized in finance and economics, featuring a ReAct agent for iterative reasoning and action.

## 🎯 Features

- **ReactAgent Framework**: Generic, reusable agent framework with explicit reasoning
- **RAG Agent V3**: Specialized implementation for finance using the new framework
- **ReAct Loop**: Explicit "Observe-Reason-Act" loop with up to 3 iterations
- **Smart Validation**: Analysis step after each tool call to verify if the query was fully answered
- **Financial Tools**: Semantic search, real-time stock prices (Yahoo Finance), and comparison
- **Local PDF Processing**: Endpoint to process PDFs locally using **Docling**, preserving layout and semantics
- **Decoupled Architecture**: Clean separation between generic agent logic and domain-specific tools

## 🏗️ Architecture

```
┌─────────────────┐
│   User Query    │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────┐
│     Generic ReactAgent Loop     │
│  ┌──────────────────────────┐   │
│  │ 1. Tool Selection (LLM)  │   │
│  │ 2. Tool Execution        │   │
│  │ 3. Explicit Reasoning    │←──┼── Uses _analyze_progress
│  │    (Critic Step)         │   │   to decide next move:
│  │                          │   │   - Continue (Finish)
│  │                          │   │   - Retry (Refine Query)
│  │                          │   │   - Switch Tool
│  └──────────────────────────┘   │
└────────┬────────────────────────┘
         │
         ▼
┌─────────────────┐
│ Final Synthesis │ ← Combines all observations
└─────────────────┘
```

## 🚀 Installation

### 1. Clone the repository
```bash
git clone https://github.com/fredstrey/react_agent.git
```

### 2. Create virtual environment
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure environment variables
Create `.env` file:
```env
OPENROUTER_API_KEY=your_key_here
```

### 5. Start Qdrant (Docker)
```bash
docker run -p 6333:6333 qdrant/qdrant
```

## 📦 Project Structure

```
Finance.AI/
├── ReactAgent/                # GENERIC FRAMEWORK
│   ├── agent.py               # Core ReAct Logic
│   ├── decorators.py          # @tool decorator
│   ├── executor.py            # Tool Executor
│   ├── registry.py            # Tool Registry
│   └── context.py             # Execution Context
├── agents/
│   ├── rag_agent_v3.py        # Finance Agent (implements ReactAgent)
│   └── ...
├── tools/
│   ├── rag_tools_v3.py        # Financial Tools (@tool decorated)
│   └── ...
├── api/
│   └── api.py                 # FastAPI (Async with run_in_threadpool)
└── examples/
    ├── rag_v3_demo.py         # Main Demo
    └── ...
```

## 🛠️ Available Tools

### 1. `search_documents`
Semantic search in financial documents
```python
search_documents(query="What is the Selic rate?")
```

### 2. `get_stock_price`
Get price of ONE stock
```python
get_stock_price(ticker="AAPL")
```

### 3. `compare_stocks`
Compare MULTIPLE stocks
```python
compare_stocks(tickers=["AAPL", "MSFT", "GOOGL"])
```

### 4. `redirect`
Indicates that question is out of scope

## 🎮 Usage

### Start API
```bash
python api/api.py
```

### Make request
```bash
curl -X POST http://localhost:8000/stream \
  -H "Content-Type: application/json" \
  -d '{"message": "What is the price of AAPL and who defines the Selic rate?"}'
```
**Or try the dubious vibecoded html frontend :D**

![alt text](image.png)

### Process PDF (Locally with Docling)
```bash
curl -X POST http://localhost:8000/process_pdf \
  -H "Content-Type: application/json" \
  -d '{"pdf_path": "C:/path/to/doc.pdf", "max_tokens": 500}'
```

### Add documents
```bash
python examples/add_finance_docs.py
```

## 🧠 ReAct Agent

The ReAct Agent implements a reasoning and action loop:

### Possible Decisions
- **CONTINUE**: Sufficient information
- **RETRY_WITH_REFINEMENT**: Refine query and try again
- **CALL_DIFFERENT_TOOL**: Call different tool
- **INSUFFICIENT_DATA**: Insufficient data after 3 iterations

### Execution Example
```
Query: "Price of AAPL and who defines Selic?"

Iteration 1: get_stock_price("AAPL") → $273.76
ReAct: Missing answer about Selic → CALL_DIFFERENT_TOOL

Iteration 2: search_documents("Who defines Selic?") → COPOM
ReAct: Both parts answered → CONTINUE

Response: "AAPL: $273.76. COPOM defines the Selic rate."
```

## ⚙️ Configuration

### LLM Models
Configured in `agents/rag_agent_v3.py`:
```python
RAGAgentV3(
    model="xiaomi/mimo-v2-flash:free",
    max_iterations=3  # ReAct iterations
)
```

### Qdrant
```python
EmbeddingManager(
    embedding_model="qwen3-embedding:0.6b",
    qdrant_url="http://localhost:6333",
    collection_name="rag_api"
)
```

## 📊 Implemented Features

✅ ReAct loop with 3 iterations  
✅ Sequential tool execution (semaphore)  
✅ Multi-part query detection  
✅ Automatic query refinement  
✅ Context accumulation between iterations  
✅ Intelligent response synthesis  
✅ Domain validation (finance/economics)  

## 🐛 Troubleshooting

### Qdrant won't connect
```bash
# Check if container is running
docker ps

# Start Qdrant
docker run -p 6333:6333 qdrant/qdrant
```

### Invalid API Key
Check `.env` file and configure `OPENROUTER_API_KEY`

### Empty responses
Run `python examples/add_finance_docs.py` to add documents
