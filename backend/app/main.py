"""CRAP FastAPI application entrypoint."""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.database import db
from app.routes import protocol, search, ws

app = FastAPI(title="CRAP - Comprehensive Review and Analysis Platform")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(protocol.router)
app.include_router(protocol.sessions_router)
app.include_router(search.router)
app.include_router(ws.router)


@app.on_event("startup")
async def startup():
    db.init_db()


@app.get("/api/health")
async def health():
    return {"status": "ok", "trial_count": db.trial_count()}
