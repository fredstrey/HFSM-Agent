"""
Agent Response Model
"""
from pydantic import BaseModel, Field


class AgentResponse(BaseModel):
    """Resposta padronizada de um agent"""
    answer: str = Field(..., description="Resposta final do agent")
