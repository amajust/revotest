from pydantic import BaseModel


class SegmentResult(BaseModel):
    start: float
    end: float
    text: str
    avg_logprob: float


class PostProcessedSegment(SegmentResult):
    low_confidence: bool


class TranscriptionResponse(BaseModel):
    raw_results: list[SegmentResult]
    post_processed_results: list[PostProcessedSegment]


class ErrorResponse(BaseModel):
    detail: str
