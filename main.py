"""
backend/main.py  —  Darji Pro Virtual Try-On
Uses LightX API (free credits, no card needed).

Setup:
  pip install fastapi uvicorn python-multipart httpx python-dotenv

Run:
  uvicorn main:app --host 0.0.0.0 --port 8000 --reload

Environment variables:
  LIGHTX_API_KEY=your_key_here
  ALLOWED_ORIGINS=*
"""

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

LIGHTX_API_KEY      = os.getenv("LIGHTX_API_KEY", "")
MAX_FILE_SIZE_MB    = int(os.getenv("MAX_FILE_SIZE_MB", "10"))
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024
ALLOWED_ORIGINS     = os.getenv("ALLOWED_ORIGINS", "*").split(",")

LIGHTX_BASE_URL = "https://api.lightxeditor.com/external/api/v1"

jobs: dict[str, dict] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not LIGHTX_API_KEY:
        logger.warning("LIGHTX_API_KEY not set!")
    else:
        logger.info("Darji Pro Try-On backend started (LightX mode).")
    yield


app = FastAPI(title="Darji Pro Virtual Try-On API", version="6.0.0", lifespan=lifespan)

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


async def upload_image_to_lightx(image_bytes: bytes, client: httpx.AsyncClient) -> str:
    """Upload image bytes to LightX and get back a URL."""
    headers = {"x-api-key": LIGHTX_API_KEY}

    # Step 1: Request upload URL from LightX
    ext_resp = await client.post(
        f"{LIGHTX_BASE_URL}/upload-image-url",
        headers=headers,
        json={"uploadType": "imageUrl", "size": len(image_bytes), "contentType": "image/jpeg"},
    )
    ext_resp.raise_for_status()
    upload_data = ext_resp.json()

    upload_url   = upload_data["body"]["uploadUrl"]
    image_url    = upload_data["body"]["imageUrl"]

    # Step 2: Upload image bytes to the presigned URL
    put_resp = await client.put(
        upload_url,
        content=image_bytes,
        headers={"Content-Type": "image/jpeg"},
    )
    put_resp.raise_for_status()

    return image_url


async def run_lightx_tryon(
    job_id: str,
    person_bytes: bytes,
    garment_bytes: bytes,
):
    """Call LightX virtual outfit try-on API."""
    jobs[job_id]["status"] = "processing"

    try:
        headers = {
            "x-api-key": LIGHTX_API_KEY,
            "Content-Type": "application/json",
        }

        async with httpx.AsyncClient(timeout=120.0) as client:
            # Upload both images to LightX
            logger.info(f"[{job_id}] Uploading images to LightX...")
            person_url  = await upload_image_to_lightx(person_bytes, client)
            garment_url = await upload_image_to_lightx(garment_bytes, client)

            logger.info(f"[{job_id}] Submitting try-on request...")

            # Submit try-on request
            tryon_resp = await client.post(
                f"{LIGHTX_BASE_URL}/virtual-outfit-tryon",
                headers=headers,
                json={
                    "imageUrl": person_url,
                    "clothImageUrl": garment_url,
                },
            )
            tryon_resp.raise_for_status()
            tryon_data = tryon_resp.json()

            order_id = tryon_data.get("body", {}).get("orderId")
            if not order_id:
                raise Exception(f"No orderId returned: {tryon_data}")

            logger.info(f"[{job_id}] LightX order ID: {order_id}")

            # Poll for result
            for attempt in range(40):
                await asyncio.sleep(4)

                status_resp = await client.post(
                    f"{LIGHTX_BASE_URL}/order-status",
                    headers=headers,
                    json={"orderId": order_id},
                )

                if status_resp.status_code != 200:
                    continue

                status_data = status_resp.json()
                status = status_data.get("body", {}).get("status")

                logger.info(f"[{job_id}] LightX status: {status} (attempt {attempt+1})")

                if status == "active":
                    result_url = status_data.get("body", {}).get("output")
                    if result_url:
                        jobs[job_id]["status"]       = "succeeded"
                        jobs[job_id]["result_url"]   = result_url
                        jobs[job_id]["completed_at"] = datetime.utcnow().isoformat()
                        logger.info(f"[{job_id}] Job succeeded!")
                        return
                    else:
                        raise Exception("Status active but no output image.")

                elif status in ("failed", "error"):
                    raise Exception(f"LightX processing failed: {status_data}")

            raise Exception("Timed out waiting for LightX result.")

    except Exception as e:
        logger.error(f"[{job_id}] Error: {traceback.format_exc()}")
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"]  = str(e)


async def run_vton_two_pass(
    job_id: str,
    person_bytes: bytes,
    shirt_bytes: Optional[bytes],
    pant_bytes: Optional[bytes],
):
    try:
        if shirt_bytes and pant_bytes:
            # Pass 1: shirt
            await run_lightx_tryon(job_id, person_bytes, shirt_bytes)
            if jobs[job_id]["status"] == "failed":
                return

            intermediate_url = jobs[job_id]["result_url"]
            jobs[job_id]["result_url"] = None
            jobs[job_id]["status"]     = "processing"

            # Download intermediate result for pass 2
            async with httpx.AsyncClient(timeout=60.0) as http:
                resp = await http.get(intermediate_url)
                resp.raise_for_status()
                intermediate_bytes = resp.content

            # Pass 2: pant
            await run_lightx_tryon(job_id, intermediate_bytes, pant_bytes)

        elif shirt_bytes:
            await run_lightx_tryon(job_id, person_bytes, shirt_bytes)

        elif pant_bytes:
            await run_lightx_tryon(job_id, person_bytes, pant_bytes)

    except Exception as e:
        logger.error(f"[{job_id}] Two-pass error: {traceback.format_exc()}")
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"]  = str(e)


@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "lightx_key_set": bool(LIGHTX_API_KEY),
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
        raise HTTPException(status_code=400, detail="At least one cloth image required.")

    person_bytes = await read_and_validate(person, "person")
    shirt_bytes  = await read_and_validate(shirt, "shirt") if shirt else None
    pant_bytes   = await read_and_validate(pant,  "pant")  if pant  else None

    job_id = str(uuid.uuid4())
    jobs[job_id] = {
        "job_id": job_id, "status": "pending",
        "result_url": None, "error": None,
        "created_at": datetime.utcnow().isoformat(), "completed_at": None,
    }

    background_tasks.add_task(run_vton_two_pass, job_id, person_bytes, shirt_bytes, pant_bytes)
    logger.info(f"[{job_id}] Job queued.")
    return JSONResponse({"job_id": job_id, "status": "pending"})


@app.get("/api/tryon/status/{job_id}")
async def get_status(job_id: str):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")
    return JSONResponse({
        "job_id": job["job_id"], "status": job["status"],
        "result_url": job.get("result_url"), "error": job.get("error"),
    })


@app.delete("/api/tryon/cleanup")
async def cleanup_old_jobs():
    cutoff = datetime.utcnow() - timedelta(hours=1)
    removed = 0
    for job_id in list(jobs.keys()):
        if datetime.fromisoformat(jobs[job_id]["created_at"]) < cutoff:
            del jobs[job_id]
            removed += 1
    return {"removed": removed, "remaining": len(jobs)}