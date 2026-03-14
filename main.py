"""
backend/main.py  —  Darji Pro Virtual Try-On  (fixed)

Fixes applied:
  1. Real error messages exposed — no more "unexpected error" hiding the cause
  2. Replicate output parsing fixed for all SDK versions
  3. Model version updated to latest stable IDM-VTON
  4. Added /api/tryon/test endpoint to verify Replicate token without images
  5. requirements.txt versions pinned to avoid SDK breakage

Run:
  uvicorn main:app --host 0.0.0.0 --port 8000 --reload
"""

import random
import io
import os
import uuid
import asyncio
import logging
import traceback
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ─── Config ───────────────────────────────────────────────────────────────────

REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN")
MAX_FILE_SIZE_MB    = int(os.getenv("MAX_FILE_SIZE_MB", "10"))
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024
ALLOWED_ORIGINS     = os.getenv("ALLOWED_ORIGINS", "*").split(",")

# IDM-VTON model — latest stable version
VTON_MODEL = "cuuupid/idm-vton:906425dbca90663ff5427624839572cc56ea7d380343d13e2a4c4b09d3f0c30f"

# In-memory job store
jobs: dict[str, dict] = {}

# ─── App setup ────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    if not REPLICATE_API_TOKEN:
        raise RuntimeError("REPLICATE_API_TOKEN environment variable is not set!")
    logger.info(f"✅ Darji Pro Try-On backend started. Token ends with: ...{REPLICATE_API_TOKEN[-6:]}")
    yield
    logger.info("Backend shutting down.")


app = FastAPI(
    title="Darji Pro Virtual Try-On API",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Helpers ──────────────────────────────────────────────────────────────────

def validate_image(file: UploadFile, field_name: str):
    allowed_types = {"image/jpeg", "image/jpg", "image/png", "image/webp"}
    # Some clients send wrong content_type — be lenient, just check it's not None
    if file.content_type and file.content_type not in allowed_types:
        logger.warning(f"{field_name}: unexpected content_type={file.content_type}, proceeding anyway")


async def read_and_validate(file: UploadFile, field_name: str) -> bytes:
    validate_image(file, field_name)
    data = await file.read()
    if len(data) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"{field_name}: File too large. Max {MAX_FILE_SIZE_MB}MB allowed.",
        )
    logger.info(f"  {field_name}: {len(data) / 1024:.1f} KB, type={file.content_type}")
    return data


def extract_url_from_output(output) -> str:
    """
    Replicate SDK returns different types depending on version:
      - list of FileOutput objects  (newer SDK)
      - list of strings/URLs        (older SDK)
      - a single FileOutput object
      - a single string URL
    This handles all cases.
    """
    # Unwrap list
    if isinstance(output, list):
        if len(output) == 0:
            raise ValueError("Replicate returned empty output list")
        result = output[0]
    else:
        result = output

    # FileOutput object has a .url attribute
    if hasattr(result, 'url'):
        url = str(result.url)
    elif hasattr(result, 'read'):
        # It's a file-like object — read and re-upload not needed, just get url
        url = str(result)
    else:
        url = str(result)

    if not url.startswith("http"):
        raise ValueError(f"Replicate returned invalid URL: {url!r}")

    return url


# ─── Core VTON logic ──────────────────────────────────────────────────────────

async def run_vton_model(
    job_id: str,
    person_bytes: bytes,
    shirt_bytes: Optional[bytes],
    pant_bytes: Optional[bytes],
):
    try:
        jobs[job_id]["status"] = "processing"
        logger.info(f"[{job_id}] Starting VTON processing...")

        client = replicate.Client(api_token=REPLICATE_API_TOKEN)

        async def run_single_pass(person_data: bytes, garment_data: bytes, description: str) -> str:
            logger.info(
                f"[{job_id}] Running pass: '{description}' "
                f"person={len(person_data)}B garment={len(garment_data)}B"
            )
            try:
                output = await asyncio.to_thread(
                    client.run,
                    VTON_MODEL,
                    input={
                        "human_img":        io.BytesIO(person_data),
                        "garm_img":         io.BytesIO(garment_data),
                        "garment_des":      description,
                        "is_checked":       True,
                        "is_checked_crop":  True,
                        "denoise_steps":    30,
                        "seed":             random.randint(0, 99999),
                    },
                )
                logger.info(f"[{job_id}] Raw output type: {type(output)}, value: {output!r}")
                url = extract_url_from_output(output)
                logger.info(f"[{job_id}] Pass result URL: {url}")
                return url

            except replicate.exceptions.ReplicateError as e:
                # Surface the REAL Replicate error (auth, billing, model not found, etc.)
                logger.error(f"[{job_id}] ReplicateError in pass '{description}': {e}")
                raise RuntimeError(f"Replicate API error: {str(e)}")

            except Exception as e:
                logger.error(f"[{job_id}] Error in pass '{description}': {e}\n{traceback.format_exc()}")
                raise

        result_url = None

        if shirt_bytes and pant_bytes:
            logger.info(f"[{job_id}] Two-pass mode: shirt → pant")
            shirt_url = await run_single_pass(person_bytes, shirt_bytes, "shirt garment")

            async with httpx.AsyncClient(timeout=60.0) as http:
                resp = await http.get(shirt_url)
                resp.raise_for_status()
                intermediate_bytes = resp.content
            logger.info(f"[{job_id}] Intermediate downloaded: {len(intermediate_bytes)}B")

            result_url = await run_single_pass(intermediate_bytes, pant_bytes, "pant garment")

        elif shirt_bytes:
            logger.info(f"[{job_id}] Single-pass: shirt only")
            result_url = await run_single_pass(person_bytes, shirt_bytes, "shirt garment")

        elif pant_bytes:
            logger.info(f"[{job_id}] Single-pass: pant only")
            result_url = await run_single_pass(person_bytes, pant_bytes, "pant garment")

        jobs[job_id]["status"]       = "succeeded"
        jobs[job_id]["result_url"]   = result_url
        jobs[job_id]["completed_at"] = datetime.utcnow().isoformat()
        logger.info(f"[{job_id}] ✅ Succeeded: {result_url}")

    except replicate.exceptions.ReplicateError as e:
        real_error = str(e)
        logger.error(f"[{job_id}] ReplicateError: {real_error}")
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"]  = f"Replicate error: {real_error}"

    except httpx.HTTPError as e:
        logger.error(f"[{job_id}] HTTP error: {e}")
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"]  = f"HTTP error downloading result: {str(e)}"

    except Exception as e:
        # ✅ FIXED: Now logs the FULL traceback AND saves real message to job
        full_tb = traceback.format_exc()
        logger.error(f"[{job_id}] Unexpected error:\n{full_tb}")
        jobs[job_id]["status"] = "failed"
        # Save the real error message — not a generic string
        jobs[job_id]["error"]  = f"{type(e).__name__}: {str(e)}"


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "service": "Darji Pro Try-On API",
        "active_jobs": len(jobs),
        "replicate_token_set": bool(REPLICATE_API_TOKEN),
    }


@app.get("/api/tryon/test-token")
async def test_replicate_token():
    """
    Quick endpoint to verify your Replicate token works.
    Visit: https://darji-backend.onrender.com/api/tryon/test-token
    """
    try:
        client = replicate.Client(api_token=REPLICATE_API_TOKEN)
        # List models — lightweight call that verifies auth
        models = await asyncio.to_thread(lambda: list(client.models.list()))
        return {
            "status": "ok",
            "message": "Replicate token is valid ✅",
            "token_suffix": f"...{REPLICATE_API_TOKEN[-6:]}",
        }
    except replicate.exceptions.ReplicateError as e:
        return JSONResponse(
            status_code=401,
            content={
                "status": "error",
                "message": f"Replicate token invalid ❌: {str(e)}",
            }
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)}
        )


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

    logger.info(f"New job — shirt={shirt is not None}, pant={pant is not None}")

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
        run_vton_model, job_id, person_bytes, shirt_bytes, pant_bytes
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