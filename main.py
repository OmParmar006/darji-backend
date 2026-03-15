"""
backend/main.py  —  Darji Pro Virtual Try-On
Uses HuggingFace IDM-VTON Space (FREE) via Gradio client.

Setup:
  pip install fastapi uvicorn python-multipart httpx python-dotenv gradio-client

Run:
  uvicorn main:app --host 0.0.0.0 --port 8000 --reload

Environment variables (.env):
  HF_TOKEN=hf_xxxxxxxxxxxxxxxxxx   <- get from huggingface.co/settings/tokens
  ALLOWED_ORIGINS=*
"""

import io
import os
import uuid
import base64
import asyncio
import logging
import traceback
from typing import Optional
from datetime import datetime, timedelta
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

HF_TOKEN            = os.getenv("HF_TOKEN", "")
MAX_FILE_SIZE_MB    = int(os.getenv("MAX_FILE_SIZE_MB", "10"))
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024
ALLOWED_ORIGINS     = os.getenv("ALLOWED_ORIGINS", "*").split(",")
HF_SPACE            = "yisol/IDM-VTON"

jobs: dict[str, dict] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Darji Pro Try-On backend started (HuggingFace mode).")
    if not HF_TOKEN:
        logger.warning("HF_TOKEN not set — requests may be rate limited.")
    yield


app = FastAPI(title="Darji Pro Virtual Try-On API", version="3.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


async def read_and_validate(file: UploadFile, field_name: str) -> bytes:
    data = await file.read()
    if len(data) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(status_code=413, detail=f"{field_name}: File too large.")
    logger.info(f"  {field_name}: {len(data)/1024:.1f} KB")
    return data


async def run_vton_model(
    job_id: str,
    person_bytes: bytes,
    garment_bytes: bytes,
    garment_desc: str,
):
    try:
        jobs[job_id]["status"] = "processing"
        logger.info(f"[{job_id}] Starting HuggingFace VTON — {garment_desc}")

        from gradio_client import Client, handle_file
        import tempfile

        def run_gradio():
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as pf:
                pf.write(person_bytes)
                person_path = pf.name
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as gf:
                gf.write(garment_bytes)
                garment_path = gf.name
            try:
                hf_headers = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}
                client = Client(HF_SPACE, headers=hf_headers)
                logger.info(f"[{job_id}] Connected to HF Space, running tryon...")
                result = client.predict(
                    dict={
                        "background": handle_file(person_path),
                        "layers": [],
                        "composite": None,
                    },
                    garm_img=handle_file(garment_path),
                    garment_des=garment_desc,
                    is_checked=True,
                    is_checked_crop=True,
                    denoise_steps=30,
                    seed=42,
                    api_name="/tryon",
                )
                logger.info(f"[{job_id}] Raw result type: {type(result)}")
                return result
            finally:
                try:
                    os.unlink(person_path)
                except Exception:
                    pass
                try:
                    os.unlink(garment_path)
                except Exception:
                    pass

        result = await asyncio.to_thread(run_gradio)

        # Extract image URL or path from Gradio response
        result_image = result[0] if isinstance(result, (list, tuple)) else result
        if isinstance(result_image, dict):
            result_image = (
                result_image.get("url")
                or result_image.get("path")
                or str(result_image)
            )

        if isinstance(result_image, str) and result_image.startswith("http"):
            result_url = result_image
        elif isinstance(result_image, str) and os.path.exists(result_image):
            with open(result_image, "rb") as f:
                img_data = f.read()
            b64 = base64.b64encode(img_data).decode("utf-8")
            result_url = f"data:image/jpeg;base64,{b64}"
        else:
            raise ValueError(f"Cannot extract image from result: {result!r}")

        jobs[job_id]["status"]       = "succeeded"
        jobs[job_id]["result_url"]   = result_url
        jobs[job_id]["completed_at"] = datetime.utcnow().isoformat()
        logger.info(f"[{job_id}] Job succeeded.")

    except Exception as e:
        logger.error(f"[{job_id}] Error:\n{traceback.format_exc()}")
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"]  = f"{type(e).__name__}: {str(e)}"


async def run_vton_two_pass(
    job_id: str,
    person_bytes: bytes,
    shirt_bytes: Optional[bytes],
    pant_bytes: Optional[bytes],
):
    try:
        if shirt_bytes and pant_bytes:
            # Pass 1: shirt
            await run_vton_model(job_id, person_bytes, shirt_bytes, "upper body shirt")
            if jobs[job_id]["status"] == "failed":
                return

            intermediate_url = jobs[job_id]["result_url"]
            jobs[job_id]["result_url"] = None
            jobs[job_id]["status"]     = "processing"

            # Download intermediate result
            if intermediate_url.startswith("data:image"):
                intermediate_bytes = base64.b64decode(intermediate_url.split(",")[1])
            else:
                async with httpx.AsyncClient(timeout=60.0) as http:
                    resp = await http.get(intermediate_url)
                    resp.raise_for_status()
                    intermediate_bytes = resp.content

            # Pass 2: pant on top of shirt result
            await run_vton_model(job_id, intermediate_bytes, pant_bytes, "lower body pants")

        elif shirt_bytes:
            await run_vton_model(job_id, person_bytes, shirt_bytes, "upper body shirt")

        elif pant_bytes:
            await run_vton_model(job_id, person_bytes, pant_bytes, "lower body pants")

    except Exception as e:
        logger.error(f"[{job_id}] Two-pass error:\n{traceback.format_exc()}")
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"]  = f"{type(e).__name__}: {str(e)}"


@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "hf_token_set": bool(HF_TOKEN),
        "active_jobs": len(jobs),
    }


@app.post("/api/tryon/submit")
async def submit_tryon(
    background_tasks: BackgroundTasks,
    person: UploadFile = File(...),
    shirt:  Optional[UploadFile] = File(None),
    pant:   Optional[UploadFile] = File(None),
):
    if not shirt and not pant:
        raise HTTPException(
            status_code=400,
            detail="At least one cloth image (shirt or pant) is required.",
        )

    person_bytes = await read_and_validate(person, "person")
    shirt_bytes  = await read_and_validate(shirt, "shirt") if shirt else None
    pant_bytes   = await read_and_validate(pant,  "pant")  if pant  else None

    job_id = str(uuid.uuid4())
    jobs[job_id] = {
        "job_id":       job_id,
        "status":       "pending",
        "result_url":   None,
        "error":        None,
        "created_at":   datetime.utcnow().isoformat(),
        "completed_at": None,
    }

    background_tasks.add_task(
        run_vton_two_pass, job_id, person_bytes, shirt_bytes, pant_bytes
    )
    logger.info(f"[{job_id}] Job queued.")
    return JSONResponse({"job_id": job_id, "status": "pending"})


@app.get("/api/tryon/status/{job_id}")
async def get_status(job_id: str):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")
    return JSONResponse({
        "job_id":     job["job_id"],
        "status":     job["status"],
        "result_url": job.get("result_url"),
        "error":      job.get("error"),
    })


@app.delete("/api/tryon/cleanup")
async def cleanup_old_jobs():
    cutoff = datetime.utcnow() - timedelta(hours=1)
    removed = 0
    for job_id in list(jobs.keys()):
        created = jobs[job_id].get("created_at")
        if created and datetime.fromisoformat(created) < cutoff:
            del jobs[job_id]
            removed += 1
    return {"removed": removed, "remaining": len(jobs)}