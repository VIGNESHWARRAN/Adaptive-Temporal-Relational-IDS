"""FastAPI Backend Server for Adaptive Temporal-Relational IDS Dashboard.

Provides REST API endpoints for:
- File-based flow triggering ('n' records)
- Batch size dynamic updates
- Active CSV file selection and watch-folder scanning
- Real-time detection metrics and historical flow telemetry
- Model switching and dataset selection
"""

import os
import sys
import shutil
from typing import Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Ensure repo root is on sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.ui.engine import IDSEngine

# Global engine instance holder
engine: Optional[IDSEngine] = None

app = FastAPI(
    title="Adaptive Temporal-Relational IDS Neural SOC",
    description="File-Based Network Flow Triggering & Real-Time Intrusion Detection System",
    version="2.0.0",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(STATIC_DIR, exist_ok=True)


class TriggerRequest(BaseModel):
    n: Optional[int] = None
    filename: Optional[str] = None


class BatchSizeRequest(BaseModel):
    n: int


class FileSelectRequest(BaseModel):
    filename: str


class SwitchModelRequest(BaseModel):
    dataset_name: str
    fusion_type: Optional[str] = "concat"


def get_engine() -> IDSEngine:
    global engine
    if engine is None:
        # Default initialization
        engine = IDSEngine(
            watch_dir=os.path.join(REPO_ROOT, "incoming_flows"),
            default_batch_size=1,
            dataset_name="nf_cse_cic_ids2018_v2",
            fusion_type="concat",
        )
    return engine


@app.get("/")
async def serve_index():
    index_file = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
    return JSONResponse({"status": "Dashboard UI is initializing"})


@app.get("/api/status")
async def api_status():
    eng = get_engine()
    return eng.get_status()


@app.post("/api/trigger")
async def api_trigger(req: TriggerRequest = None):
    eng = get_engine()
    n = req.n if req else None
    if req and req.filename and req.filename != eng.active_filename:
        eng.set_active_file(req.filename)
    res = eng.trigger(n=n)
    return res


@app.post("/api/batch_size")
async def api_set_batch_size(req: BatchSizeRequest):
    eng = get_engine()
    new_n = eng.set_batch_size(req.n)
    return {"success": True, "batch_size": new_n}


@app.post("/api/select_file")
async def api_select_file(req: FileSelectRequest):
    eng = get_engine()
    ok = eng.set_active_file(req.filename)
    if not ok:
        raise HTTPException(status_code=404, detail=f"File '{req.filename}' not found in watch folder.")
    return eng.get_status()


@app.post("/api/reset_pointer")
async def api_reset_pointer():
    eng = get_engine()
    eng.reset_pointer()
    return eng.get_status()


@app.post("/api/clear_history")
async def api_clear_history():
    eng = get_engine()
    eng.clear_history()
    return eng.get_status()


@app.get("/api/history")
async def api_get_history():
    eng = get_engine()
    return {
        "history": eng.history,
        "stats": eng.stats,
    }


@app.get("/api/models")
async def api_get_models():
    eng = get_engine()
    return {
        "current_dataset": eng.dataset_name,
        "current_fusion": eng.fusion_type,
        "available_datasets": eng.get_available_datasets(),
        "available_fusions": [
            "concat",
            "gated",
            "bilinear",
            "cross_attention",
            "decision_ensemble",
            "quantum_pqc",
        ],
    }


@app.post("/api/switch_model")
async def api_switch_model(req: SwitchModelRequest):
    eng = get_engine()
    try:
        eng.load_model(req.dataset_name, req.fusion_type or "concat")
        return {"success": True, "status": eng.get_status()}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/upload")
async def api_upload_csv(file: UploadFile = File(...)):
    eng = get_engine()
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only .csv files are supported.")

    dest_path = os.path.join(eng.watch_dir, file.filename)
    with open(dest_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    eng.set_active_file(file.filename)
    return {"success": True, "filename": file.filename, "status": eng.get_status()}


# Mount static assets
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
