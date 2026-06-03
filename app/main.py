import asyncio
import logging
import os
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, UploadFile, File, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from faster_whisper import WhisperModel

from app.config import settings
from app.schemas import TranscriptionResponse, ErrorResponse
from app.services.stt_service import STTService

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

executor: ThreadPoolExecutor | None = None
model: WhisperModel | None = None
stt_service: STTService | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global executor, model, stt_service

    logger.info("Initializing STT service and model weights...")
    loop = asyncio.get_event_loop()

    executor = ThreadPoolExecutor(
        max_workers=settings.max_workers,
        thread_name_prefix="stt-worker",
    )

    model = await loop.run_in_executor(
        executor,
        lambda: WhisperModel(
            settings.model_size,
            device=settings.device,
            compute_type=settings.compute_type,
            download_root=settings.model_cache_dir,
        ),
    )

    stt_service = STTService(model=model, executor=executor, config=settings)
    logger.info("STT service ready")
    yield

    logger.info("Shutting down STT service...")
    if executor is not None:
        executor.shutdown(wait=True)
    if model is not None:
        del model


app = FastAPI(
    title="Advanced STT with Intelligent Segmentation",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.isdir(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.get("/")
async def root():
    return FileResponse(os.path.join(static_dir, "index.html"))


@app.post(
    "/api/v1/transcribe",
    response_model=TranscriptionResponse,
    responses={
        400: {"model": ErrorResponse},
        413: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
    summary="Transcribe an uploaded WAV file",
    description=(
        "Upload a mono 16-bit PCM WAV file. "
        "The service runs a three-pass pipeline: "
        "VAD-based silence segmentation, "
        "faster-whisper transcription, "
        "and confidence-based re-segregation."
    ),
)
async def transcribe(file: UploadFile = File(...)):
    if not file.filename or not file.filename.lower().endswith(".wav"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only .wav files are supported",
        )

    contents = await file.read()

    if len(contents) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Empty file uploaded",
        )

    max_bytes = settings.max_file_size_mb * 1024 * 1024
    if len(contents) > max_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds maximum size of {settings.max_file_size_mb}MB",
        )

    if stt_service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service not initialized",
        )

    try:
        return await stt_service.transcribe(contents)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception:
        logger.exception("Transcription failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Transcription processing failed",
        )
