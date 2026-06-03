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

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

executor = None
model = None
stt_service = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global executor, model, stt_service

    loop = asyncio.get_event_loop()
    executor = ThreadPoolExecutor(max_workers=settings.max_workers, thread_name_prefix="stt")
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
    logger.info("ready")
    yield

    if executor:
        executor.shutdown(wait=True)


app = FastAPI(title="STT", version="1.0.0", lifespan=lifespan)

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.isdir(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.get("/")
async def root():
    return FileResponse(os.path.join(static_dir, "index.html"))


@app.post("/api/v1/transcribe", response_model=TranscriptionResponse)
async def transcribe(file: UploadFile = File(...)):
    if not file.filename or not file.filename.lower().endswith(".wav"):
        raise HTTPException(400, "Only .wav files are supported")

    contents = await file.read()
    if not contents:
        raise HTTPException(400, "Empty file")

    if len(contents) > settings.max_file_size_mb * 1024 * 1024:
        raise HTTPException(413, f"File exceeds {settings.max_file_size_mb}MB limit")

    if stt_service is None:
        raise HTTPException(503, "Service not initialized")

    try:
        return await stt_service.transcribe(contents)
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception:
        logger.exception("transcription failed")
        raise HTTPException(500, "Processing failed")
