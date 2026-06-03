import os


class Settings:
    model_size = os.getenv("MODEL_SIZE", "small")
    device = os.getenv("DEVICE", "cpu")
    compute_type = os.getenv("COMPUTE_TYPE", "float32")

    vad_aggressiveness = int(os.getenv("VAD_AGRESSIVENESS", "2"))
    frame_duration_ms = int(os.getenv("FRAME_DURATION_MS", "30"))
    sample_rate = int(os.getenv("SAMPLE_RATE", "16000"))
    silence_threshold_s = float(os.getenv("SILENCE_THRESHOLD_S", "0.5"))

    confidence_threshold = float(os.getenv("CONFIDENCE_THRESHOLD", "-0.6"))

    max_workers = int(os.getenv("MAX_WORKERS", "2"))
    max_file_size_mb = int(os.getenv("MAX_FILE_SIZE_MB", "50"))

    model_cache_dir = os.getenv("MODEL_CACHE_DIR", "/tmp/whisper-models")


settings = Settings()
