from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


# ============================================================================
# Workflow Models
# ============================================================================

class WorkflowParameter(BaseModel):
    """
    Represents a single parameter in a workflow.
    
    Attributes:
        name: Parameter identifier
        required: Whether parameter must be provided
        value: Current/submitted value
        default: Fallback value if not provided
    """
    name: str = Field(..., description="Parameter name")
    required: bool = Field(..., description="Whether the parameter is required")
    value: Optional[str] = Field(None, description="The actual value for this parameter")
    default: Optional[str] = Field(None, description="Default value if parameter is optional")

    class Config:
        json_schema_extra = {
            "example": {
                "name": "analysis_type",
                "required": True,
                "value": "financial",
                "default": None
            }
        }


class WorkflowObject(BaseModel):
    """
    Defines a complete workflow with parameters.
    
    Attributes:
        name: Workflow function identifier
        display_name: Human-readable workflow name
        parameters: List of parameter configurations
    """
    name: str = Field(..., description="Workflow function name")
    display_name: str = Field(..., description="Human-readable workflow name")
    parameters: List[WorkflowParameter] = Field(
        ..., 
        description="List of parameters with their values"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "name": "analyze_stock",
                "display_name": "Stock Analysis",
                "parameters": [
                    {
                        "name": "symbol",
                        "required": True,
                        "value": "AAPL",
                        "default": None
                    }
                ]
            }
        }


# ============================================================================
# Main Request Models
# ============================================================================

class UserAskRequest(BaseModel):
    """
    Main API request model for user queries.
    
    This is the primary request format for the /api/ask endpoint.
    
    Attributes:
        task: User's question or request
        reply_id: Document ID for response generation
        symbol: Stock ticker symbol (required for financial queries)
        symbol_name: Human-readable symbol name
        is_web_research: Enable web search capability
        is_workflow: Whether this uses a workflow
        workflow_object: Workflow configuration if applicable
        chat_id: Session identifier for conversation tracking
        report: Whether to return full report or summary
        deep_search: Enable deep search mode
        analysis_required: Whether quantitative analysis is needed
    """
    task: str = Field(
        ..., 
        description="The task or question the user is asking",
        min_length=1,
        max_length=5000
    )
    reply_id: str = Field(
        ..., 
        description="The id for the document to which we need to generate response"
    )
    symbol: str = Field(
        ..., 
        description="The stock symbol to fetch data for",
        min_length=1,
        max_length=10
    )
    symbol_name: str = Field(
        ..., 
        description="The stock symbol name to fetch data for"
    )
    is_web_research: bool = Field(
        ..., 
        description="Whether the user wants to have web research option enabled"
    )
    is_workflow: Optional[bool] = Field(
        False,
        description="Whether this is a workflow request"
    )
    workflow_object: Optional[WorkflowObject] = Field(
        None,
        description="Workflow object containing function name and parameter values"
    )
    chat_id: Optional[str] = Field(
        None,
        description="The chat ID to identify which conversation this message belongs to"
    )
    report: Optional[bool] = Field(
        False,
        description="Whether the user wants the full report or just the answer"
    )
    deep_search: Optional[bool] = Field(
        False,
        description="Whether the user wants to have deep search option enabled"
    )
    analysis_required: Optional[bool] = Field(
        False,
        description="Whether the user wants to have analysis required option enabled"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "task": "What is the current stock price of Apple?",
                "reply_id": "doc_12345",
                "symbol": "AAPL",
                "symbol_name": "Apple Inc.",
                "is_web_research": True,
                "is_workflow": False,
                "workflow_object": None,
                "chat_id": "chat_67890",
                "report": False,
                "deep_search": False,
                "analysis_required": True
            }
        }


# ============================================================================
# Configuration Models
# ============================================================================

class ExperimentConfig(BaseModel):
    """
    Configuration for experiments and evaluations.
    
    Attributes:
        splits: Data splits to use
        dataset_name: Dataset identifier
        experiment_prefix: Experiment name prefix
        max_concurrency: Parallel execution limit
        longterm: Whether to use long-term memory (mem0)
    """
    splits: Optional[List[str]] = Field(
        ["memory_saver_split"],
        description="List of splits to use for the experiment"
    )
    dataset_name: str = Field(
        "Fintech Dataset",
        description="Name of the dataset to use"
    )
    experiment_prefix: str = Field(
        ...,
        description="Prefix for the experiment",
        min_length=1,
        max_length=100
    )
    max_concurrency: int = Field(
        2,
        description="Maximum number of concurrent processes",
        ge=1,
        le=100
    )
    longterm: Optional[bool] = Field(
        True,
        description="Whether to use long-term memory (mem0). "
                    "If False, skips mem0 search and storage, uses only existing memory system."
    )

    class Config:
        json_schema_extra = {
            "example": {
                "splits": ["memory_saver_split", "test_split"],
                "dataset_name": "Fintech Dataset",
                "experiment_prefix": "exp_v1",
                "max_concurrency": 4,
                "longterm": True
            }
        }


# ============================================================================
# Memory Management Models
# ============================================================================

class DeleteMemoryRequest(BaseModel):
    """
    Request to delete a memory entry.
    
    Attributes:
        memory_id: Identifier of memory to delete
    """
    memory_id: str = Field(..., description="The ID of the memory to delete")

    class Config:
        json_schema_extra = {
            "example": {
                "memory_id": "mem_abc123"
            }
        }


class GetMemoriesRequest(BaseModel):
    """
    Request to retrieve memories.
    
    Attributes:
        limit: Maximum number of memories to return
    """
    limit: Optional[int] = Field(
        100,
        description="Maximum number of memories to retrieve (default: 100, max: 500)",
        ge=1,
        le=500
    )

    class Config:
        json_schema_extra = {
            "example": {
                "limit": 50
            }
        }


# ============================================================================
# Feedback Models
# ============================================================================

class FeedbackRequest(BaseModel):
    """
    User feedback submission for a message or conversation.
    
    This feedback is:
    1. Recorded in LangSmith for analytics
    2. Stored in MongoDB for historical tracking
    3. Used for model improvement and evaluation
    
    Attributes:
        trace_id: LangSmith trace identifier
        key: Feedback category (e.g., 'user_feedback', 'accuracy')
        comment: User's feedback text
        score: Numeric feedback score
        messageid: MongoDB message document ID
    """
    trace_id: str = Field(
        ..., 
        description="The trace ID to identify the specific conversation or run",
        min_length=1
    )
    key: str = Field(
        'user_feedback',
        description="The feedback key/category",
        min_length=1,
        max_length=50
    )
    comment: str = Field(
        ..., 
        description="User's feedback comment",
        min_length=1,
        max_length=5000
    )
    score: float = Field(
        ..., 
        description="Feedback score (typically 0-1 or 1-5 scale)",
        ge=0.0,
        le=5.0
    )
    messageid: str = Field(
        ..., 
        description="The message ID in MongoDB to update with feedback",
        min_length=1
    )

    class Config:
        json_schema_extra = {
            "example": {
                "trace_id": "trace_12345",
                "key": "user_feedback",
                "comment": "The analysis was very helpful and accurate",
                "score": 4.5,
                "messageid": "msg_67890"
            }
        }


# ============================================================================
# Response Models
# ============================================================================

class FeedbackResponse(BaseModel):
    """Response after feedback submission."""
    message: str
    message_id: str
    trace_id: str
    user_id: str
    feedback: Dict[str, Any]

    class Config:
        json_schema_extra = {
            "example": {
                "message": "Feedback submitted successfully",
                "message_id": "msg_67890",
                "trace_id": "trace_12345",
                "user_id": "user_123",
                "feedback": {
                    "key": "user_feedback",
                    "score": 4.5,
                    "comment": "Very helpful"
                }
            }
        }


class ErrorResponse(BaseModel):
    """Standard error response."""
    detail: str
    error_code: Optional[str] = None
    timestamp: Optional[str] = None

    class Config:
        json_schema_extra = {
            "example": {
                "detail": "Invalid request parameters",
                "error_code": "INVALID_REQUEST",
                "timestamp": "2024-04-19T12:00:00Z"
            }
        }
