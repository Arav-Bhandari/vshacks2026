"""API-level tests against a tmp DB (no full pipeline runs)."""
import fitz
from fastapi.testclient import TestClient

from app.database import db


def _make_client(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "test.db")
    db.init_db()
    from app.main import app

    return TestClient(app)


def test_health(tmp_path, monkeypatch):
    client = _make_client(tmp_path, monkeypatch)
    resp = client.get("/api/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_upload_rejects_non_pdf(tmp_path, monkeypatch):
    client = _make_client(tmp_path, monkeypatch)
    resp = client.post(
        "/api/protocol/upload", files={"file": ("notes.txt", b"hello", "text/plain")}
    )
    assert resp.status_code == 400


def test_upload_accepts_pdf(tmp_path, monkeypatch):
    client = _make_client(tmp_path, monkeypatch)
    doc = fitz.open()
    doc.new_page().insert_text((72, 72), "Protocol")
    pdf_bytes = doc.tobytes()
    doc.close()
    resp = client.post(
        "/api/protocol/upload",
        files={"file": ("protocol.pdf", pdf_bytes, "application/pdf")},
    )
    assert resp.status_code == 200
    assert "session_id" in resp.json()


def test_get_unknown_session_404(tmp_path, monkeypatch):
    client = _make_client(tmp_path, monkeypatch)
    resp = client.get("/api/protocol/does-not-exist")
    assert resp.status_code == 404


def test_sessions_list(tmp_path, monkeypatch):
    client = _make_client(tmp_path, monkeypatch)
    resp = client.get("/api/sessions")
    assert resp.status_code == 200
    assert resp.json() == {"sessions": []}
