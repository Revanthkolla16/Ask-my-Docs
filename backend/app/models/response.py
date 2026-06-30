from pydantic import BaseModel, Field
from typing import List, Optional, Literal


class Citation(BaseModel):
    """A single source citation linked to a chunk."""
    chunk_id: str = Field(..., description="ID of the chunk being cited")
    source: str = Field(..., description="Source document name/path")
    page_num: Optional[int] = Field(default=None, description="Page number within the source")
    section: Optional[str] = Field(default=None, description="Section heading within the source")
    snippet: str = Field(..., description="Short excerpt from the cited chunk")


class HallucinationFlag(BaseModel):
    """Hallucination detection result for a single claim in the answer."""
    claim: str = Field(..., description="The individual claim extracted from the answer")
    label: Literal["entailment", "neutral", "contradiction"] = Field(
        ..., description="NLI label for this claim vs its cited context"
    )
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score from the NLI model")
    cited_chunk_id: Optional[str] = Field(
        default=None, description="The chunk ID this claim was verified against"
    )
    flagged: bool = Field(
        ..., description="True if this claim is considered a potential hallucination"
    )


class QueryResponse(BaseModel):
    """Full response returned by POST /query."""
    answer: str = Field(..., description="The generated answer grounded in context")
    citations: List[Citation] = Field(
        default_factory=list,
        description="List of source citations supporting the answer",
    )
    hallucination_flags: List[HallucinationFlag] = Field(
        default_factory=list,
        description="Per-claim hallucination detection results (empty if detection was skipped)",
    )
    confidence: Optional[float] = Field(
        default=None,
        description="Overall confidence score (fraction of claims that are entailed)",
    )
    latency_ms: float = Field(..., description="Total pipeline latency in milliseconds")


class IngestResponse(BaseModel):
    """Response returned by POST /ingest."""
    document_id: str = Field(..., description="Unique identifier assigned to the ingested document")
    chunks_created: int = Field(..., description="Number of chunks created from the document")
    chunking_strategy: str = Field(..., description="Chunking strategy that was used")
    message: str = Field(default="Document ingested successfully")
