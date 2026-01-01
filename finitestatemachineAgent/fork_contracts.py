"""
Fork Contracts - Domain Models
================================

Domain-agnostic contract models for deterministic parallel execution.
Forks return structured contracts instead of summaries.

EPISTEMIC ENHANCEMENT:
- Evidence types (retrieved, general_knowledge, inferred)
- Uncertainty tracking (with reasons)
- Confidence calibration by evidence type
"""

from pydantic import BaseModel, Field
from typing import Any, List, Dict, Optional
from enum import Enum


class EvidenceType(str, Enum):
    """Type of evidence supporting a claim."""
    RETRIEVED = "retrieved"  # From RAG/tools/documents
    GENERAL_KNOWLEDGE = "general_knowledge"  # From LLM training
    INFERRED = "inferred"  # Logical deduction


class Evidence(BaseModel):
    """
    Explicit evidence with epistemic type.
    
    Example:
        Evidence(type=EvidenceType.RETRIEVED, source="tool:search_web")
        Evidence(type=EvidenceType.GENERAL_KNOWLEDGE, source="llm_knowledge")
    """
    type: EvidenceType = Field(..., description="Type of evidence")
    source: str = Field(..., description="Source identifier (e.g., 'tool:search_web')")


class UncertainTopic(BaseModel):
    """
    Topic with explicit uncertainty and reason.
    
    Differentiates absence of evidence from absence of knowledge.
    
    Example:
        UncertainTopic(
            topic="ACC spread calculation",
            reason="not_found_in_retrieval"
        )
    """
    topic: str = Field(..., description="Topic that is uncertain")
    reason: str = Field(
        ..., 
        description="Why uncertain (e.g., 'not_found_in_retrieval', 'insufficient_context')"
    )


class Claim(BaseModel):
    """
    A verifiable claim extracted from research.
    
    EPISTEMIC ENHANCEMENT:
    - Evidence is now typed (retrieved, general_knowledge, inferred)
    - Confidence can be calibrated by evidence type
    
    Example:
        Claim(
            key="leasing.supervisor",
            value="BCB",
            evidence=[
                Evidence(type=EvidenceType.RETRIEVED, source="tool:search_web")
            ],
            confidence=0.95
        )
    """
    key: str = Field(
        ..., 
        description="Unique claim identifier (e.g., 'topic.subtopic')"
    )
    value: Any = Field(
        ..., 
        description="Claim value (can be string, number, bool, etc.)"
    )
    evidence: List[Evidence] = Field(
        default_factory=list, 
        description="Evidence supporting this claim (with types)"
    )
    confidence: float = Field(
        default=1.0, 
        ge=0.0, 
        le=1.0, 
        description="Confidence score (0.0 to 1.0), calibrated by evidence type"
    )


class ForkResult(BaseModel):
    """
    Structured contract returned by each fork.
    
    This is the output of ForkContractState, replacing the old text-based summary.
    
    EPISTEMIC ENHANCEMENT:
    - uncertain_topics replaces omissions (with explicit reasons)
    - Evidence is typed for epistemic reasoning
    """
    branch_id: str = Field(..., description="Fork identifier")
    
    claims: List[Claim] = Field(
        default_factory=list, 
        description="Verifiable claims extracted from research"
    )
    
    coverage: List[str] = Field(
        default_factory=list, 
        description="Topics/questions that were successfully covered"
    )
    
    uncertain_topics: List[UncertainTopic] = Field(
        default_factory=list, 
        description="Topics with explicit uncertainty and reasons"
    )
    
    confidence: float = Field(
        default=1.0, 
        ge=0.0, 
        le=1.0, 
        description="Overall confidence in the fork's results"
    )


class MergedContract(BaseModel):
    """
    Result of deterministic contract merge.
    
    This is the output of MergeState._contract_merge(), replacing semantic merge.
    
    EPISTEMIC ENHANCEMENT:
    - Preserves evidence plurality (variants)
    - Tracks uncertainty with reasons
    - Uncertainty never invalidates claims
    """
    resolved: Dict[str, Any] = Field(
        default_factory=dict, 
        description="Claims where all forks agree (with evidence variants)"
    )
    
    conflicts: Dict[str, List[Any]] = Field(
        default_factory=dict, 
        description="Claims where forks disagree (multiple values)"
    )
    
    coverage: List[str] = Field(
        default_factory=list, 
        description="All topics covered by at least one fork"
    )
    
    uncertain_topics: List[UncertainTopic] = Field(
        default_factory=list, 
        description="Topics with explicit uncertainty across forks"
    )
    
    total_forks: int = Field(
        default=0, 
        description="Number of forks that contributed"
    )
