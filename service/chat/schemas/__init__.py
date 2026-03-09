"""Chat schemas package"""

from service.chat.schemas.trace import (
    Trace,
    TraceSummary,
)

from service.chat.schemas.events import (
    BaseEvent,
    TraceStartEvent,
    TraceProgressEvent,
    TraceCompleteEvent,
    TraceErrorEvent,
    TraceCancelledEvent,
    StepStartEvent,
    StepUpdateEvent,
    StepProgressEvent,
    StepCompleteEvent,
    StepFailedEvent,
    StepCancelledEvent,
    ArtifactCreatedEvent,
    ArtifactUpdatedEvent,
    WebSocketMessage,
    Event,
)

from service.chat.schemas.steps import (
    BaseStep,
    UserInputProcessingStep,
    IntentRecognitionStep,
    HistoryCompressionStep,
    RetrievalStep,
    ToolCallStep,
    LLMCallStep,
    ResponseGenerationStep,
    CustomStep,
)

from service.chat.schemas.artifacts import (
    BaseArtifact,
    TextArtifact,
    JSONArtifact,
    ImageArtifact,
    FileArtifact,
    RetrievalResultChunk,
    RetrievalResultsArtifact,
    IntentArtifact,
    ToolCallArtifact,
    ToolResultArtifact,
    LLMOutputArtifact,
    UsageStatsArtifact,
    ErrorArtifact,
    Artifact,
)

__all__ = [
    # Trace
    "Trace",
    "TraceSummary",
    # Events
    "BaseEvent",
    "TraceStartEvent",
    "TraceProgressEvent",
    "TraceCompleteEvent",
    "TraceErrorEvent",
    "TraceCancelledEvent",
    "StepStartEvent",
    "StepUpdateEvent",
    "StepProgressEvent",
    "StepCompleteEvent",
    "StepFailedEvent",
    "StepCancelledEvent",
    "ArtifactCreatedEvent",
    "ArtifactUpdatedEvent",
    "WebSocketMessage",
    "Event",
    # Steps
    "BaseStep",
    "UserInputProcessingStep",
    "IntentRecognitionStep",
    "HistoryCompressionStep",
    "RetrievalStep",
    "ToolCallStep",
    "LLMCallStep",
    "ResponseGenerationStep",
    "CustomStep",
    # Artifacts
    "BaseArtifact",
    "TextArtifact",
    "JSONArtifact",
    "ImageArtifact",
    "FileArtifact",
    "RetrievalResultChunk",
    "RetrievalResultsArtifact",
    "IntentArtifact",
    "ToolCallArtifact",
    "ToolResultArtifact",
    "LLMOutputArtifact",
    "UsageStatsArtifact",
    "ErrorArtifact",
    "Artifact",
]
