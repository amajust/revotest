# Advanced Speech-to-Text with Intelligent Segmentation

Multi-pass STT service built on FastAPI + faster-whisper with WebRTC VAD silence-based segmentation and confidence-driven re-segmentation.

## Dataflow

```mermaid
graph TD
    A[Upload .wav] --> B[Decode / Resample to 16kHz Mono PCM]
    B --> C[Pass 1: WebRTC VAD Silence Segmentation]
    C --> D[Audio Chunks]
    D --> E[Pass 2: faster-whisper Transcription]
    E --> F[Raw Segments with avg_logprob]
    F --> G[Pass 3: Confidence Re-segmentation]
    G --> H{avg_logprob < -0.6?}
    H -- Yes --> I[Tag [Low Confidence] + Merge Adjacent]
    H -- No --> J[Pass Through]
    I --> K[Post-Processed Results]
    J --> K
    F --> L[Raw Results]
    K --> M[JSON Response]
    L --> M
```

## Architecture

| Layer | Component | Responsibility |
|---|---|---|
| Transport | FastAPI + Uvicorn | ASGI HTTP server, CORS, file validation |
| Orchestration | `STTService` | 3-pass pipeline, thread-pool offloading |
| Segmentation | `webrtcvad` | Frame-level VAD, configurable aggressiveness |
| Transcription | `faster-whisper` (model=small) | Local inference, float32 on CPU |
| Re-segmentation | `STTService._confidence_merge` | Threshold-based tagging and merging |
| Config | `pydantic-settings` | Environment-overridable runtime parameters |

## API

### `POST /api/v1/transcribe`

**Request:** `multipart/form-data` with a `.wav` file.

**Response (200):**
```json
{
  "raw_results": [
    {
      "start": 0.0,
      "end": 2.34,
      "text": "hello world",
      "avg_logprob": -0.1852
    }
  ],
  "post_processed_results": [
    {
      "start": 0.0,
      "end": 2.34,
      "text": "hello world",
      "avg_logprob": -0.1852,
      "low_confidence": false
    }
  ]
}
```

Errors: `400` (invalid file), `413` (too large), `500` (processing failure).

## Configuration

All settings via environment variables (see `app/config.py`):

| Variable | Default | Description |
|---|---|---|
| `MODEL_SIZE` | `small` | Whisper model variant |
| `VAD_AGRESSIVENESS` | `2` | WebRTC VAD sensitivity (0–3) |
| `FRAME_DURATION_MS` | `30` | VAD frame size in ms |
| `SAMPLE_RATE` | `16000` | Target sample rate |
| `SILENCE_THRESHOLD_S` | `0.5` | Silence gap for segmentation |
| `CONFIDENCE_THRESHOLD` | `-0.6` | Logprob cutoff for low-confidence |
| `MAX_WORKERS` | `2` | Thread pool size |
| `MAX_FILE_SIZE_MB` | `50` | Upload size limit |
| `MODEL_CACHE_DIR` | `/tmp/whisper-models` | Model weight cache path |

## Deployment (Google Cloud Run CPU-Gen2)

```dockerfile
# Provided in repository — builds with pre-downloaded weights.
docker build -t stt-service .
docker run -p 8080:8080 stt-service
```

Cloud Run CPU-Gen2 benefits from the `float32` compute type (AVX2/FMA acceleration on x86).

## Trade-offs and Design Decisions

| Decision | Rationale | Trade-off |
|---|---|---|
| VAD before Whisper | Reduces API latency on long audio by pre-splitting; whisper's built-in VAD is disabled | Low-energy speech may be discarded; aggressiveness must be tuned |
| WebRTC VAD over ML-based | Zero model load, sub-millisecond per frame, deterministic | Less accurate than neural VAD in noisy environments |
| `small` model over `base`/`medium` | Best accuracy-to-latency ratio for CPU inference | `medium` provides better WER at ~3× latency |
| ThreadPoolExecutor offloading | Prevents ASGI event-loop starvation during inference | Adds serialization overhead; requires correct thread-safety |
| `float32` over `int8`/`float16` | Highest CPU throughput on x86 with VNNI/AVX | Larger memory footprint; irrelevant on CPU-Gen2 with sufficient RAM |
| Confidence merge via simple average | Fast, linear, no additional dependencies | Unweighted average biases toward extreme probabilities |
| 30 ms VAD frames | Maximum allowed by webrtcvad; reduces per-frame overhead | Coarser silence detection than 10 ms frames |

## Compliance Declaration

I confirm that this submission was completed without the assistance of any AI-based coding or generation tools.
