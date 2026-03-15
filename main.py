"""
backend/main.py

Production-ready FastAPI backend for Darji Pro Virtual Try-On.
Uses Replicate's IDM-VTON model (best open-source VTON model).

Setup:
  pip install fastapi uvicorn python-multipart replicate httpx python-dotenv

Run:
  uvicorn main:app --host 0.0.0.0 --port 8000

Environment variables (.env file):
  REPLICATE_API_TOKEN=r8_your_token_here
  MAX_FILE_SIZE_MB=10
  ALLOWED_ORIGINS=*
"""

import os
import uuid
import asyncio
import logging
from typing import Optional
from datetime import datetime, timedelta
from contextlib import asynccontextmanager

import httpx
import replicate
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN")
MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", "10"))
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")

VTON_MODEL = "cuuupid/idm-vton:906425dbca90663ff5427624839572cc56ea7d380343d13e2a4c4b09d3f0c30f"

jobs: dict[str, dict] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not REPLICATE_API_TOKEN:
        raise RuntimeError("REPLICATE_API_TOKEN environment variable is not set!")
    logger.info("Darji Pro Try-On backend started.")
    yield
    logger.info("Backend shutting down.")


app = FastAPI(
    title="Darji Pro Virtual Try-On API",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def validate_image(file: UploadFile, field_name: str):
    allowed_types = {"image/jpeg", "image/jpg", "image/png", "image/webp"}
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"{field_name}: Only JPEG, PNG, or WebP images are allowed."
        )


async def read_and_validate(file: UploadFile, field_name: str) -> bytes:
    validate_image(file, field_name)
    data = await file.read()
    if len(data) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"{field_name}: File too large. Max {MAX_FILE_SIZE_MB}MB allowed."
        )
    return data


async def run_vton_model(job_id: str, person_bytes: bytes, shirt_bytes: Optional[bytes], pant_bytes: Optional[bytes]):
    try:
        jobs[job_id]["status"] = "processing"

        client = replicate.Client(api_token=REPLICATE_API_TOKEN)

        async def run_single_pass(person_data: bytes, garment_data: bytes, description: str) -> str:
            logger.info(f"[{job_id}] Running VTON pass: {description}")
            output = await asyncio.to_thread(
                client.run,
                VTON_MODEL,
                input={
                    "human_img": person_data,
                    "garm_img": garment_data,
                    "garment_des": description,
                    "is_checked": True,
                    "is_checked_crop": False,
                    "denoise_steps": 30,
                    "seed": 42,
                },
            )
            if isinstance(output, list) and len(output) > 0:
                return str(output[0])
            return str(output)

        result_url = None

        if shirt_bytes and pant_bytes:
            shirt_url = await run_single_pass(person_bytes, shirt_bytes, "shirt garment")
            async with httpx.AsyncClient() as http:
                resp = await http.get(shirt_url)
                intermediate_bytes = resp.content
            result_url = await run_single_pass(intermediate_bytes, pant_bytes, "pant garment")
        elif shirt_bytes:
            result_url = await run_single_pass(person_bytes, shirt_bytes, "shirt garment")
        elif pant_bytes:
            result_url = await run_single_pass(person_bytes, pant_bytes, "pant garment")

        jobs[job_id]["status"] = "succeeded"
        jobs[job_id]["result_url"] = result_url
        jobs[job_id]["completed_at"] = datetime.utcnow().isoformat()
        logger.info(f"[{job_id}] Job succeeded: {result_url}")

    except replicate.exceptions.ReplicateError as e:
        logger.error(f"[{job_id}] Replicate error: {e}")
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = f"AI model error: {str(e)}"

    except Exception as e:
        logger.error(f"[{job_id}] Unexpected error: {e}", exc_info=True)
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = "An unexpected error occurred. Please try again."


@app.get("/health")
async def health_check():
    return {"status": "ok", "service": "Darji Pro Try-On API"}


@app.post("/api/tryon/submit")
async def submit_tryon(
    background_tasks: BackgroundTasks,
    person: UploadFile = File(..., description="Customer full body photo"),
    shirt: Optional[UploadFile] = File(None, description="Shirt cloth image"),
    pant: Optional[UploadFile] = File(None, description="Pant cloth image"),
):
    if not shirt and not pant:
        raise HTTPException(
            status_code=400,
            detail="At least one cloth image (shirt or pant) is required."
        )

    person_bytes = await read_and_validate(person, "person")
    shirt_bytes = await read_and_validate(shirt, "shirt") if shirt else None
    pant_bytes = await read_and_validate(pant, "pant") if pant else None

    job_id = str(uuid.uuid4())
    jobs[job_id] = {
        "job_id": job_id,
        "status": "pending",
        "result_url": None,
        "error": None,
        "created_at": datetime.utcnow().isoformat(),
        "completed_at": None,
    }

    background_tasks.add_task(run_vton_model, job_id, person_bytes, shirt_bytes, pant_bytes)
    logger.info(f"[{job_id}] Job submitted")

    return JSONResponse({"job_id": job_id, "status": "pending"})


@app.get("/api/tryon/status/{job_id}")
async def get_status(job_id: str):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")

    return JSONResponse({
        "job_id": job["job_id"],
        "status": job["status"],
        "result_url": job.get("result_url"),
        "error": job.get("error"),
    })


@app.delete("/api/tryon/cleanup")
async def cleanup_old_jobs():
    cutoff = datetime.utcnow() - timedelta(hours=1)
    removed = 0
    for job_id in list(jobs.keys()):
        created = jobs[job_id].get("created_at")
        if created:
            created_dt = datetime.fromisoformat(created)
            if created_dt < cutoff:
                del jobs[job_id]
                removed += 1
    return {"removed": removed, "remaining": len(jobs)}