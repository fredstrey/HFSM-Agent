"""
Poetry Agent Example
====================
Exemplo de um agente especializado em poesia que:
1. Começa direto no estado de resposta (AnswerState)
2. Tem uma persona criativa definida no system_instruction
3. Não gasta tokens com análise de intenção ou routing
"""

import sys
import asyncio
import os
from pathlib import Path

# Configuração de path para importação
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("⚠️ dotenv not found. Ensure .env is loaded or install python-dotenv.")

from finitestatemachineAgent import Agent

async def main():
    print("🎭 Inicializando Agente Poeta...")
    
    # Verifica key ou usa dummy para permitir inicialização (falhará na execução se dummy)
    api_key = os.getenv("OPENROUTER_API_KEY")
    
    # 1. Criação do Agente Especializado
    poet = Agent(
        llm_provider="openrouter",
        model="xiaomi/mimo-v2-flash:free",
        api_key=api_key,
        system_instruction=(
            "Você é um poeta virtuoso e criativo. "
            "Escreva poesias em português, com rimas ricas e métrica agradável. "
            "Seja profundo mas conciso."
        ),
        # 🔥 Configuração de Workflow Agent:
        initial_state="AnswerState",   # Pula routing/tools -> vai direto gerar texto
        enable_intent_analysis=False,  # Desabilita análise de intenção (desnecessária)
        enable_parallel_planning=False # Desabilita planejamento (desnecessário)
    )
    
    # 2. Definição do Tema
    tema = "O Bug que virou Feature no código da vida"
    prompt = f"Escreva um poema curto sobre: {tema}"
    
    print(f"\n✍️ Tema: '{tema}'")
    print("\n--- INÍCIO DO POEMA ---\n")
    
    # 3. Geração via Streaming (para efeito visual)
    try:
        async for token in poet.stream(prompt):
            print(token, end="", flush=True)
    except Exception as e:
        print(f"\n❌ Erro na geração: {e}")
        if "API_KEY" in str(e):
            print("💡 Dica: Verifique sua OPENROUTER_API_KEY")

    print("\n\n--- FIM DO POEMA ---")

if __name__ == "__main__":
    asyncio.run(main())
