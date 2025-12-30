# Fred.AI - RAG Agent com ReAct

Sistema de RAG (Retrieval-Augmented Generation) especializado em finanças e economia, com agente ReAct para raciocínio e ação iterativa.

## 🎯 Características

- **RAG Agent V2**: Busca semântica em documentos financeiros
- **ReAct Agent**: Loop de raciocínio e ação com até 3 iterações
- **Ferramentas Financeiras**: Preços de ações, comparação, busca em documentos
- **Validação Inteligente**: Verifica se respostas são relevantes ao domínio
- **Síntese de Respostas**: Combina múltiplas iterações sem redundância

## 🏗️ Arquitetura

```
┌─────────────────┐
│   User Query    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Context Agent  │ ← Extrai intenção
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────┐
│      ReAct Loop (max 3x)        │
│  ┌──────────────────────────┐   │
│  │ 1. Tool Calling Agent    │   │
│  │ 2. Execute 1 Tool        │   │
│  │ 3. ReAct Analysis        │   │
│  │ 4. Decide: Continue/Retry│   │
│  └──────────────────────────┘   │
└────────┬────────────────────────┘
         │
         ▼
┌─────────────────┐
│ Response Synth  │ ← Combina respostas
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│Validation Agent │ ← Valida domínio
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Final Response │
└─────────────────┘
```

## 🚀 Instalação

### 1. Clone o repositório
```bash
git clone <repo-url>
cd Fred.AI
```

### 2. Crie ambiente virtual
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate  # Windows
```

### 3. Instale dependências
```bash
pip install -r requirements.txt
```

### 4. Configure variáveis de ambiente
Crie arquivo `.env`:
```env
OPENROUTER_API_KEY=your_key_here
```

### 5. Inicie Qdrant (Docker)
```bash
docker run -p 6333:6333 qdrant/qdrant
```

## 📦 Estrutura do Projeto

```
Fred.AI/
├── agents/
│   ├── context_agent.py      # Extração de intenção
│   ├── rag_agent_v2.py        # RAG Agent principal
│   ├── react_agent.py         # ReAct: Reasoning + Acting
│   └── validation_agent.py    # Validação de domínio
├── api/
│   └── api.py                 # FastAPI endpoints
├── core/
│   ├── tool_calling_agent.py  # Base para tool calling
│   ├── execution_context.py   # Contexto de execução
│   ├── registry.py            # Registro de ferramentas
│   └── executor.py            # Executor de ferramentas
├── embedding_manager/
│   └── embedding_manager.py   # Gerenciador de embeddings
├── providers/
│   ├── openrouter.py          # Provider OpenRouter
│   └── openrouter_function_caller.py
├── tools/
│   └── rag_tools.py           # Ferramentas RAG
└── examples/
    ├── add_finance_docs.py    # Adicionar documentos
    └── test_react_agent.py    # Testes do ReAct
```

## 🛠️ Ferramentas Disponíveis

### 1. `search_documents`
Busca semântica em documentos financeiros
```python
search_documents(query="O que é taxa Selic?")
```

### 2. `get_stock_price`
Obtém preço de UMA ação
```python
get_stock_price(ticker="AAPL")
```

### 3. `compare_stocks`
Compara MÚLTIPLAS ações
```python
compare_stocks(tickers=["AAPL", "MSFT", "GOOGL"])
```

### 4. `redirect`
Indica que pergunta está fora do escopo

## 🎮 Uso

### Iniciar API
```bash
python api/api.py
```

### Fazer requisição
```bash
curl -X POST http://localhost:8000/stream \
  -H "Content-Type: application/json" \
  -d '{"message": "Qual o preço da AAPL e quem define a taxa Selic?"}'
```

### Adicionar documentos
```bash
python examples/add_finance_docs.py
```

## 🧠 ReAct Agent

O ReAct Agent implementa um loop de raciocínio e ação:

### Decisões Possíveis
- **CONTINUE**: Informação suficiente
- **RETRY_WITH_REFINEMENT**: Refinar query e tentar novamente
- **CALL_DIFFERENT_TOOL**: Chamar ferramenta diferente
- **INSUFFICIENT_DATA**: Dados insuficientes após 3 iterações

### Exemplo de Execução
```
Query: "Preço da AAPL e quem define Selic?"

Iteração 1: get_stock_price("AAPL") → $273.76
ReAct: Falta responder sobre Selic → CALL_DIFFERENT_TOOL

Iteração 2: search_documents("Quem define Selic?") → COPOM
ReAct: Ambas partes respondidas → CONTINUE

Resposta: "AAPL: $273.76. COPOM define a taxa Selic."
```

## ⚙️ Configuração

### Modelos LLM
Configurados em `agents/rag_agent_v2.py`:
```python
RAGAgentV2(
    tool_caller_model="xiaomi/mimo-v2-flash:free",
    response_model="xiaomi/mimo-v2-flash:free",
    context_model="xiaomi/mimo-v2-flash:free",
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

## 📊 Recursos Implementados

✅ Loop ReAct com 3 iterações  
✅ Execução sequencial de ferramentas (semáforo)  
✅ Detecção de queries múltiplas  
✅ Refinamento automático de queries  
✅ Acumulação de contexto entre iterações  
✅ Síntese inteligente de respostas  
✅ Validação de domínio (finanças/economia)  

## 🐛 Troubleshooting

### Qdrant não conecta
```bash
# Verificar se container está rodando
docker ps

# Iniciar Qdrant
docker run -p 6333:6333 qdrant/qdrant
```

### API Key inválida
Verifique arquivo `.env` e configure `OPENROUTER_API_KEY`

### Respostas vazias
Execute `python examples/add_finance_docs.py` para adicionar documentos

## 📝 Licença

MIT License

## 👥 Contribuindo

Pull requests são bem-vindos! Para mudanças maiores, abra uma issue primeiro.
