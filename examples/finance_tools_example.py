"""
Exemplo de uso das tools financeiras com yfinance
"""
from tools.finance_tools import get_stock_price, compare_stocks
from core.registry import ToolRegistry
from core.executor import ToolExecutor
from providers.ollama import OllamaProvider
from examples.function_agent import FunctionAgent


def main():
    """Função principal"""
    
    print("=" * 70)
    print("📈 Exemplo - Tools Financeiras com yfinance")
    print("=" * 70)
    
    # ==========================================
    # 1. TESTAR TOOLS DIRETAMENTE
    # ==========================================
    
    print("\n🔍 Testando get_stock_price...")
    result = get_stock_price("AAPL", "1mo")
    print(f"\nResultado:")
    print(f"  Empresa: {result.get('company_name')}")
    print(f"  Preço atual: ${result.get('current_price')}")
    print(f"  Variação (1 mês): {result.get('change_percent')}%")
    print(f"  Resumo: {result.get('summary')}")
    
    print("\n" + "-" * 70)
    print("\n📊 Testando compare_stocks...")
    comparison = compare_stocks("AAPL,GOOGL,MSFT", "3mo")
    print(f"\nComparação (3 meses):")
    for stock in comparison.get('stocks', []):
        print(f"  {stock['ticker']}: {stock['change_percent']}%")
    print(f"\nResumo: {comparison.get('summary')}")
    
    # ==========================================
    # 2. USAR COM AGENTE
    # ==========================================
    
    print("\n" + "=" * 70)
    print("🤖 Usando tools com Agente")
    print("=" * 70)
    
    # Provider Ollama
    provider = OllamaProvider(
        model="gemma3:1b",
        temperature=0.3
    )
    
    # Registry e Executor
    registry = ToolRegistry()
    executor = ToolExecutor(registry)
    
    # Agente
    agent = FunctionAgent(
        provider=provider,
        executor=executor,
        max_iterations=3
    )
    
    # Testa queries
    queries = [
        "Qual é o preço atual da ação da Apple?",
        "Compare o desempenho das ações AAPL, GOOGL e MSFT nos últimos 6 meses"
    ]
    
    for query in queries:
        print(f"\n📝 Query: {query}")
        print("-" * 70)
        
        response = agent.run(query)
        print(f"Resposta: {response}")
        print()
    
    print("=" * 70)
    print("✅ Exemplo concluído!")
    print("=" * 70)


if __name__ == "__main__":
    main()
