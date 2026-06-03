from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    model_size: str = "small"
    device: str = "cpu"
    compute_type: str = "float32"

    vad_aggressiveness: int = 2
    frame_duration_ms: int = 30
    sample_rate: int = 16000
    silence_threshold_s: float = 0.5

    confidence_threshold: float = -0.6

    max_workers: int = 2
    max_file_size_mb: int = 50
    max_segment_duration_s: float = 30.0

    model_cache_dir: str = "/tmp/whisper-models"


settings = Settings()
