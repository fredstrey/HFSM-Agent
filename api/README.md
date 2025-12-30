# API RAG Agent - Guia de Uso

## Visão Geral

API FastAPI que utiliza o RAG Agent com FunctionGemma para tool calling, qwen3-embedding para embeddings e gemma3:1b para geração de respostas.

## Iniciar o Servidor

```bash
python api/api.py
```

O servidor estará disponível em: `http://localhost:8000`

## Endpoints

### 1. GET `/` - Informações da API

Retorna informações sobre a API e modelos utilizados.

**Exemplo:**
```bash
curl http://localhost:8000/
```

**Resposta:**
```json
{
  "message": "RAG Agent API",
  "version": "2.0.0",
  "models": {
    "tool_caller": "gemma3:1b",
    "embeddings": "qwen3-embedding:0.6b",
    "response_generator": "gemma3:1b"
  },
  "endpoints": {
    "/stream": "POST - Chat com streaming",
    "/chat": "POST - Chat sem streaming",
    "/health": "GET - Health check",
    "/documents": "POST - Adicionar documentos"
  }
}
```

---

### 2. GET `/health` - Health Check

Verifica o status de todos os componentes.

**Exemplo:**
```bash
curl http://localhost:8000/health
```

**Resposta:**
```json
{
  "status": "healthy",
  "components": {
    "functiongemma": true,
    "qwen3": true,
    "qdrant": true
  },
  "collection": {
    "name": "rag_api",
    "documents": 0
  }
}
```

---

### 3. POST `/chat` - Chat sem Streaming

Envia uma mensagem e recebe resposta completa.

**Request Body:**
```json
{
  "message": "O que é RAG?",
  "conversation_id": "optional-uuid"
}
```

**Exemplo PowerShell:**
```powershell
$body = @{
    message = "O que é RAG?"
} | ConvertTo-Json

Invoke-WebRequest -Uri http://localhost:8000/chat `
    -Method POST `
    -ContentType "application/json" `
    -Body $body
```

**Exemplo Python:**
```python
import requests

response = requests.post(
    "http://localhost:8000/chat",
    json={"message": "O que é RAG?"}
)

print(response.json())
```

**Resposta:**
```json
{
  "answer": "RAG (Retrieval-Augmented Generation) é uma técnica...",
  "sources_used": ["doc1.txt", "doc2.txt"],
  "confidence": "high",
  "conversation_id": "uuid-here"
}
```

---

### 4. POST `/stream` - Chat com Streaming

Envia uma mensagem e recebe resposta em chunks via Server-Sent Events.

**Request Body:**
```json
{
  "message": "Explique embeddings",
  "conversation_id": "optional-uuid"
}
```

**Exemplo Python:**
```python
import requests
import json

response = requests.post(
    "http://localhost:8000/stream",
    json={"message": "Explique embeddings"},
    stream=True
)

for line in response.iter_lines():
    if line:
        data = json.loads(line)
        print(f"Type: {data['type']}, Content: {data['content']}")
```

**Chunks de Resposta:**

1. **Start:**
```json
{
  "type": "start",
  "content": "",
  "metadata": {"conversation_id": "uuid"}
}
```

2. **Thinking:**
```json
{
  "type": "thinking",
  "content": "Analisando sua pergunta com FunctionGemma...",
  "metadata": {}
}
```

3. **Response:**
```json
{
  "type": "response",
  "content": "Embeddings são...",
  "metadata": {
    "sources": ["doc1.txt"],
    "confidence": "high"
  }
}
```

4. **Done:**
```json
{
  "type": "done",
  "content": "",
  "metadata": {
    "conversation_id": "uuid",
    "sources_used": ["doc1.txt"],
    "confidence": "high"
  }
}
```

---

### 5. POST `/documents` - Adicionar Documentos

Adiciona documentos à base de conhecimento do RAG.

**Request Body:**
```json
{
  "documents": [
    "Python é uma linguagem de programação...",
    "Machine Learning é um subcampo da IA..."
  ],
  "metadatas": [
    {"source": "python_docs.txt", "topic": "programming"},
    {"source": "ml_guide.txt", "topic": "ai"}
  ]
}
```

**Exemplo PowerShell:**
```powershell
$body = @{
    documents = @(
        "Python é uma linguagem de programação...",
        "Machine Learning é um subcampo da IA..."
    )
    metadatas = @(
        @{source = "python_docs.txt"; topic = "programming"},
        @{source = "ml_guide.txt"; topic = "ai"}
    )
} | ConvertTo-Json -Depth 3

Invoke-WebRequest -Uri http://localhost:8000/documents `
    -Method POST `
    -ContentType "application/json" `
    -Body $body
```

**Exemplo Python:**
```python
import requests

documents = [
    "Python é uma linguagem de programação...",
    "Machine Learning é um subcampo da IA..."
]

metadatas = [
    {"source": "python_docs.txt", "topic": "programming"},
    {"source": "ml_guide.txt", "topic": "ai"}
]

response = requests.post(
    "http://localhost:8000/documents",
    json={"documents": documents, "metadatas": metadatas}
)

print(response.json())
```

**Resposta:**
```json
{
  "status": "success",
  "documents_added": 2,
  "total_documents": 2
}
```

---

## Modelos Utilizados

- **Tool Caller**: `gemma3:1b` - Decide quando buscar documentos
- **Embeddings**: `qwen3-embedding:0.6b` - Gera embeddings (1024 dimensões)
- **Response Generator**: `gemma3:1b` - Gera respostas finais

## Requisitos

1. **Ollama** com os modelos instalados:
   ```bash
   ollama pull gemma3:1b
   ollama pull qwen3-embedding:0.6b
   ollama pull gemma3:1b
   ```

2. **Qdrant** rodando:
   ```bash
   docker run -p 6333:6333 qdrant/qdrant
   ```

3. **Dependências Python**:
   ```bash
   pip install fastapi uvicorn sse-starlette
   ```

## Arquitetura

```
User Request
    ↓
FastAPI Endpoint
    ↓
RAG Agent
    ├── FunctionGemma (Tool Calling)
    ├── qwen3-embedding (Embeddings)
    ├── Qdrant (Vector DB)
    └── gemma3:1b (Response Generation)
    ↓
Response
```

## Tratamento de Erros

Todos os endpoints retornam erros HTTP apropriados:

- **500**: Erro interno do servidor
- **422**: Validação de dados falhou

**Exemplo de erro:**
```json
{
  "detail": "Mensagem de erro aqui"
}
```
