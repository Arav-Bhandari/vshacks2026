"""Upload, analyze, fetch, and export protocol sessions."""
import asyncio
import json
import uuid

from fastapi import APIRouter, HTTPException, UploadFile
from fastapi.responses import Response

from app.config import UPLOAD_DIR
from app.database import db
from app.pipeline import run_pipeline
from app.services.usdm_export import export_usdm_json, export_usdm_xml

router = APIRouter(prefix="/api/protocol", tags=["protocol"])
sessions_router = APIRouter(tags=["protocol"])


@sessions_router.get("/api/sessions")
async def sessions():
    return {"sessions": db.list_sessions()}


@router.post("/upload")
async def upload(file: UploadFile):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "file must be a .pdf")
    session_id = str(uuid.uuid4())
    filename = f"{session_id}.pdf"
    contents = await file.read()
    (UPLOAD_DIR / filename).write_bytes(contents)
    db.create_session(session_id, filename)
    return {"session_id": session_id}


@router.post("/{session_id}/analyze")
async def analyze(session_id: str):
    if not db.get_session(session_id):
        raise HTTPException(404, "session not found")
    db.update_session(session_id, status="processing")
    asyncio.create_task(run_pipeline(session_id))
    return {"status": "started"}


@router.get("/{session_id}")
async def get_protocol(session_id: str):
    session = db.get_session(session_id)
    if not session:
        raise HTTPException(404, "session not found")
    return session


@router.get("/{session_id}/export/usdm")
async def export_json(session_id: str):
    session = db.get_session(session_id)
    if not session or not session.get("usdm"):
        raise HTTPException(404, "usdm not available")
    data = export_usdm_json(session["usdm"])
    return Response(
        content=json.dumps(data, indent=2),
        media_type="application/json",
        headers={
            "Content-Disposition": f"attachment; filename=usdm_{session_id}.json"
        },
    )


@router.get("/{session_id}/export/xml")
async def export_xml(session_id: str):
    session = db.get_session(session_id)
    if not session or not session.get("usdm"):
        raise HTTPException(404, "usdm not available")
    xml = export_usdm_xml(session["usdm"])
    return Response(
        content=xml,
        media_type="application/xml",
        headers={
            "Content-Disposition": f"attachment; filename=usdm_{session_id}.xml"
        },
    )
