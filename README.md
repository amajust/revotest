---
title: RevoTest STT
emoji: 🎙️
colorFrom: indigo
colorTo: blue
sdk: docker
pinned: false
---

# Advanced Speech-to-Text with Intelligent Segmentation

Multi-pass STT service built on FastAPI + faster-whisper with WebRTC VAD silence-based segmentation and confidence-driven re-segmentation.

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

**Request:** `multipart/form-data` with a `.wav` file (via upload or in-browser recording).

**Response (200):** Returns both `raw_results` (pass 2) and `post_processed_results` (pass 3).

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

Low-confidence segments (`avg_logprob < threshold`) get tagged `[Low Confidence]` in the text and `"low_confidence": true` in post-processed output.

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

## Dashboard

A PWA dashboard is served at `GET /` with two input modes:

- **Record** — capture microphone audio, converts to WAV in-browser, sends to API
- **Upload** — pick a `.wav` file directly

Both modes display raw and post-processed results side by side via tabbed panels. Low-confidence segments are highlighted in red.

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
