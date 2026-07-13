import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from app.services import fda_analyzer as mod


def _text_message(text):
    message = SimpleNamespace(content=text)
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def test_parse_indices_braces():
    assert mod._parse_indices("{0,3}", 5) == [0, 3]


def test_parse_indices_plain_list():
    assert mod._parse_indices("0, 3", 5) == [0, 3]


def test_parse_indices_garbage_returns_empty():
    assert mod._parse_indices("no numbers here", 5) == []


def test_parse_indices_out_of_range_filtered():
    assert mod._parse_indices("{0, 99, 2}", 5) == [0, 2]


def test_parse_json_response_valid():
    result = mod._parse_json_response('{"compliance_score": 80, "summary": "ok"}')
    assert result["compliance_score"] == 80


def test_parse_json_response_with_fences():
    text = '```json\n{"compliance_score": 50}\n```'
    result = mod._parse_json_response(text)
    assert result["compliance_score"] == 50


def test_parse_json_response_no_json_raises():
    with pytest.raises(ValueError):
        mod._parse_json_response("not json at all")


def test_list_guidance_docs_with_tmp_manifest(tmp_path, monkeypatch):
    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps([
        {"filename": "doc.pdf", "category": "general", "title": "Doc", "url": "http://x"},
        {"filename": "missing.pdf", "category": "general", "title": "Missing", "url": "http://y"},
    ]))
    monkeypatch.setattr(mod, "FDA_DIR", tmp_path)
    monkeypatch.setattr(mod, "MANIFEST_PATH", manifest)
    docs = mod.list_guidance_docs()
    assert len(docs) == 1
    assert docs[0]["filename"] == "doc.pdf"


def test_list_guidance_docs_missing_manifest(tmp_path, monkeypatch):
    monkeypatch.setattr(mod, "MANIFEST_PATH", tmp_path / "nope.json")
    assert mod.list_guidance_docs() == []


@pytest.mark.asyncio
async def test_select_docs_parses_response():
    create = AsyncMock(return_value=_text_message("{0,1}"))
    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )
    docs = [
        {"category": "general", "title": "A"},
        {"category": "oncology", "title": "B"},
        {"category": "general", "title": "C"},
    ]
    indices = await mod._select_docs(client, docs, {"phase": "1"})
    assert indices == [0, 1]


@pytest.mark.asyncio
async def test_select_docs_falls_back_on_error():
    create = AsyncMock(side_effect=RuntimeError("boom"))
    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )
    docs = [
        {"category": "oncology", "title": "A"},
        {"category": "general", "title": "B"},
    ]
    indices = await mod._select_docs(client, docs, {"phase": "1"})
    assert indices == [1]


@pytest.mark.asyncio
async def test_analyze_fda_compliance_end_to_end(tmp_path, monkeypatch):
    pdf = tmp_path / "general" / "doc.pdf"
    pdf.parent.mkdir(parents=True)
    import fitz
    d = fitz.open()
    d.new_page().insert_text((72, 72), "guidance text")
    d.save(pdf)
    d.close()

    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps([
        {"filename": "general/doc.pdf", "category": "general", "title": "Doc", "url": "http://x"},
    ]))
    monkeypatch.setattr(mod, "FDA_DIR", tmp_path)
    monkeypatch.setattr(mod, "MANIFEST_PATH", manifest)

    select_msg = _text_message("{0}")
    analysis_msg = _text_message(json.dumps({
        "compliance_score": 90,
        "summary": "Looks good",
        "gaps": [],
        "strengths": ["Well documented"],
    }))
    create = AsyncMock(side_effect=[select_msg, analysis_msg])
    mock_client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )
    with patch.object(mod, "get_client", return_value=mock_client):
        result = await mod.analyze_fda_compliance({"phase": "2"})

    assert result["compliance_score"] == 90
    assert result["documents_used"] == [{"filename": "general/doc.pdf", "title": "Doc", "category": "general"}]
