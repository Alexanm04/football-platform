import asyncio
import logging
import os
import uuid
from pathlib import Path

import cv2
from fastapi import APIRouter, BackgroundTasks, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse

from src.core.config import settings
from src.core.rate_limit import limiter
from src.modules.computer_vision.application.vision_service import VisionService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/vision", tags=["Computer Vision"])
vision_service = None

MAX_VIDEO_UPLOAD_MB = int(os.getenv("MAX_VIDEO_UPLOAD_MB", "25"))
MAX_VIDEO_UPLOAD_BYTES = MAX_VIDEO_UPLOAD_MB * 1024 * 1024
MAX_VIDEO_DURATION_SECONDS = int(os.getenv("MAX_VIDEO_DURATION_SECONDS", "45"))
MAX_CONCURRENT_VIDEOS = max(1, int(os.getenv("MAX_CONCURRENT_VIDEOS", "1")))
TEMP_INPUT_DIR = Path("/tmp/football_platform/uploads")
TEMP_OUTPUT_DIR = Path("/tmp/football_platform/outputs")
ALLOWED_MIME_TYPES = {"video/mp4", "video/quicktime", "video/x-msvideo"}
ALLOWED_EXTENSIONS = {
    ".mp4": "video/mp4",
    ".avi": "video/x-msvideo",
    ".mov": "video/quicktime",
}
CHUNK_SIZE = 1024 * 1024
VIDEO_SEMAPHORE = asyncio.Semaphore(MAX_CONCURRENT_VIDEOS)

TEMP_INPUT_DIR.mkdir(parents=True, exist_ok=True)
TEMP_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def get_vision_service():
    global vision_service

    if vision_service is None:
        logger.info("Inicializando VisionService...")
        vision_service = VisionService()

    return vision_service


def _cleanup(*paths: str | Path):
    for p in paths:
        file_path = str(p)
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
        except Exception:
            logger.warning("No se pudo borrar un archivo temporal")


def _validate_filename(filename: str | None) -> str:
    if not filename:
        raise HTTPException(status_code=415, detail="El archivo no tiene un nombre válido.")

    raw_name = filename.replace("\\", "/").strip()
    if not raw_name or ".." in raw_name or "\x00" in raw_name:
        raise HTTPException(status_code=415, detail="Nombre de archivo no permitido.")
    if raw_name.startswith("/") or raw_name.startswith("../") or "/" in raw_name or "\\" in raw_name:
        raise HTTPException(status_code=415, detail="Ruta de archivo no permitida.")

    ext = Path(raw_name).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=415, detail="Formato no permitido. Usa .mp4, .avi o .mov.")

    return ext


async def _read_upload_preview(file: UploadFile, max_bytes: int = 512) -> bytes:
    data = await file.read(max_bytes)
    await file.seek(0)
    return data


def _has_valid_magic_bytes(data: bytes, mime_type: str) -> bool:
    if mime_type in {"video/mp4", "video/quicktime"}:
        return b"ftyp" in data[:64]
    if mime_type == "video/x-msvideo":
        return data[:4] == b"RIFF" and data[8:12] == b"AVI"
    return False


async def _persist_upload(file: UploadFile, destination: Path) -> None:
    total_bytes = 0
    with destination.open("wb") as handle:
        while True:
            chunk = await file.read(CHUNK_SIZE)
            if not chunk:
                break
            total_bytes += len(chunk)
            if total_bytes > MAX_VIDEO_UPLOAD_BYTES:
                raise HTTPException(status_code=413, detail="El archivo supera el límite de 25 MB.")
            handle.write(chunk)


async def _validate_video_file(path: Path) -> None:
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise HTTPException(status_code=400, detail="El archivo de vídeo está corrupto o no es válido.")

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if frame_count <= 0 or fps is None or fps <= 0 or width <= 0 or height <= 0:
            raise HTTPException(status_code=400, detail="El vídeo no contiene frames válidos o FPS válido.")

        duration_seconds = frame_count / float(fps)
        if duration_seconds > MAX_VIDEO_DURATION_SECONDS:
            raise HTTPException(status_code=400, detail="El vídeo excede la duración máxima permitida.")

        if width > 3840 or height > 2160:
            raise HTTPException(status_code=415, detail="La resolución del vídeo supera el límite permitido.")
    finally:
        cap.release()


async def _process_video(input_path: Path, output_path: Path) -> None:
    service = get_vision_service()
    await asyncio.to_thread(service.process_video, str(input_path), str(output_path))


async def _run_video_with_slot(input_path: Path, output_path: Path) -> None:
    await VIDEO_SEMAPHORE.acquire()
    try:
        await _process_video(input_path, output_path)
    finally:
        VIDEO_SEMAPHORE.release()


def _cleanup_after_processing(
    task: asyncio.Task,
    input_path: Path,
    output_path: Path,
) -> None:
    try:
        task.result()
    except asyncio.CancelledError:
        logger.warning("El procesamiento de vídeo fue cancelado.")
    except Exception:
        logger.exception("El procesamiento de vídeo terminó con error.")
    finally:
        _cleanup(input_path, output_path)


@router.post("/process-video")
@limiter.limit("1/5minute")
async def process_football_video(
    request: Request,
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = None,
):
    ext = _validate_filename(file.filename)
    mime_type = (file.content_type or "").lower()
    if mime_type not in ALLOWED_MIME_TYPES:
        raise HTTPException(status_code=415, detail="Tipo MIME no permitido para vídeos.")

    preview = await _read_upload_preview(file)
    if not preview or not _has_valid_magic_bytes(preview, mime_type):
        raise HTTPException(status_code=400, detail="El vídeo está corrupto o no coincide con su tipo MIME.")

    file_id = str(uuid.uuid4())
    input_path = (TEMP_INPUT_DIR / f"{file_id}{ext}").resolve()
    output_path = (TEMP_OUTPUT_DIR / f"tracked_{file_id}.mp4").resolve()

    if input_path.parent != TEMP_INPUT_DIR.resolve():
        raise HTTPException(status_code=415, detail="La ruta final del archivo no es segura.")

    if output_path.parent != TEMP_OUTPUT_DIR.resolve():
        raise HTTPException(status_code=415, detail="La ruta de salida no es segura.")

    processing_task: asyncio.Task | None = None
    try:
        await _persist_upload(file, input_path)
        await _validate_video_file(input_path)

        processing_task = asyncio.create_task(
            _run_video_with_slot(input_path, output_path)
        )
        try:
            await asyncio.wait_for(
                asyncio.shield(processing_task),
                timeout=settings.VIDEO_PROCESS_TIMEOUT_SECONDS,
            )
        except asyncio.TimeoutError as exc:
            logger.warning(
                "El procesamiento del vídeo excedió el tiempo máximo; "
                "la tarea continuará hasta finalizar."
            )
            processing_task.add_done_callback(
                lambda task: _cleanup_after_processing(
                    task,
                    input_path,
                    output_path,
                )
            )
            raise HTTPException(
                status_code=500,
                detail="El procesamiento del vídeo excedió el tiempo máximo permitido.",
            ) from exc

        if not output_path.exists():
            raise HTTPException(status_code=500, detail="El vídeo procesado no se pudo generar.")

        if background_tasks is not None:
            background_tasks.add_task(_cleanup, input_path, output_path)

        return FileResponse(
            path=str(output_path),
            media_type="video/mp4",
            filename=output_path.name,
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Error procesando el vídeo")
        raise HTTPException(status_code=500, detail="Error interno procesando el vídeo.") from exc
    finally:
        if processing_task is None or processing_task.done():
            if input_path.exists() and (background_tasks is None or not output_path.exists()):
                _cleanup(input_path)
            if output_path.exists() and background_tasks is None:
                _cleanup(output_path)