from pydantic import BaseModel, Field


class SegmentResult(BaseModel):
    start: float = Field(description="Start time in seconds")
    end: float = Field(description="End time in seconds")
    text: str = Field(description="Transcribed text")
    avg_logprob: float = Field(description="Average log probability of the transcription")


class PostProcessedSegment(SegmentResult):
    low_confidence: bool = Field(description="Indicates if segment fell below confidence threshold")


class TranscriptionResponse(BaseModel):
    raw_results: list[SegmentResult]
    post_processed_results: list[PostProcessedSegment]


class ErrorResponse(BaseModel):
    detail: str
