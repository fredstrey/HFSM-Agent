import sys
import os
from pydantic import BaseModel, Field
from typing import List
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add the project root to sys.path to allow importing ReactAgent
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ReactAgent import ReactAgent, tool, AgentResponse

# 1. Define custom tools using the @tool decorator
@tool()
def get_weather(city: str) -> str:
    """Returns the current weather for a given city."""
    # Mock data
    weather_data = {
        "São Paulo": "Ensolarado, 28°C",
        "Curitiba": "Nublado, 18°C",
        "Rio de Janeiro": "Muito quente, 35°C"
    }
    return weather_data.get(city, f"Não tenho informações sobre o tempo em {city}.")

@tool()
def calculate_area(shape: str, dimensions: List[float]) -> float:
    """Calculates the area of a shape. shape can be 'circle' (dims: [radius]) or 'rectangle' (dims: [width, height])."""
    if shape == "circle":
        return 3.14 * (dimensions[0] ** 2)
    elif shape == "rectangle":
        return dimensions[0] * dimensions[1]
    return 0.0

# 2. Define a custom response model (optional)
class TravelRecommendation(BaseModel):
    destination: str
    weather: str
    activity_suggestion: str
    reasoning: str

# 3. Instantiate and run the agent
def main():
    print("=== Demo: Agente ReAct Genérico ===\n")
    
    # Simple query using a tool
    agent = ReactAgent(
        model="xiaomi/mimo-v2-flash:free",
        system_prompt="Você é um assistente de viagens. Use as ferramentas para informar o tempo.",
        tools=[get_weather]
    )
    
    query = "Como está o tempo em São Paulo hoje?"
    response, _ = agent.run(query)
    print(f"\nResposta do Agente: {response.answer}")
    
    print("\n" + "="*50 + "\n")
    
    # Query with structured output
    agent_structured = ReactAgent(
        model="xiaomi/mimo-v2-flash:free",
        system_prompt="Você recomenda viagens baseado no clima. Se estiver sol, sugira praia. Se nublado, sugira museu.",
        tools=[get_weather],
        response_model=TravelRecommendation
    )
    
    query_2 = "O que eu devo fazer em Curitiba considerando o clima de lá?"
    response_2, _ = agent_structured.run(query_2)
    
    if isinstance(response_2, TravelRecommendation):
        print(f"Destino: {response_2.destination}")
        print(f"Clima: {response_2.weather}")
        print(f"Sugestão: {response_2.activity_suggestion}")
        print(f"Raciocínio: {response_2.reasoning}")
    else:
        print(f"Resposta: {response_2.answer}")

if __name__ == "__main__":
    main()
