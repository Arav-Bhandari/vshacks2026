import xml.etree.ElementTree as ET

import pytest

from app.services.llm_utils import _parse_json_response
from app.services.usdm_converter import normalize_phase
from app.services.usdm_export import export_usdm_json, export_usdm_xml


def test_normalize_phase_roman():
    assert normalize_phase("Phase II") == "Phase 2"
    assert normalize_phase("Phase III Trial") == "Phase 3 Trial"


def test_normalize_phase_no_space():
    assert normalize_phase("PHASE2") == "Phase 2"


def test_normalize_phase_empty():
    assert normalize_phase("") == ""
    assert normalize_phase(None) is None


def test_parse_json_response_fenced():
    text = '```json\n{"a": 1, "b": [1, 2, 3]}\n```'
    assert _parse_json_response(text) == {"a": 1, "b": [1, 2, 3]}


def test_parse_json_response_noisy():
    text = 'Sure, here is the JSON:\n{"x": "y"}\nHope that helps!'
    assert _parse_json_response(text) == {"x": "y"}


def test_parse_json_response_invalid():
    with pytest.raises(ValueError):
        _parse_json_response("no json here")


def test_export_usdm_json():
    result = export_usdm_json({"study": {"name": "Test"}})
    assert result["usdmVersion"] == "3.0"
    assert result["study"] == {"name": "Test"}
    assert "generatedAt" in result


def test_export_usdm_xml_roundtrip():
    usdm = {"study": {"name": "Test Study", "conditions": ["A", "B"]}}
    xml_str = export_usdm_xml(usdm)
    root = ET.fromstring(xml_str)
    assert root.tag == "USDM"
    assert root.find("study/name").text == "Test Study"
    assert [i.text for i in root.findall("study/conditions/item")] == ["A", "B"]


def test_parse_pdf_tiny():
    fitz = pytest.importorskip("fitz")
    pytest.importorskip("pdfplumber")
    import os
    import tempfile

    from app.services.pdf_parser import parse_pdf

    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Hello World")
    tmp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
    tmp.close()
    doc.save(tmp.name)
    doc.close()
    try:
        result = parse_pdf(tmp.name)
        assert "<!-- page 1 -->" in result
        assert "Hello World" in result
    finally:
        os.unlink(tmp.name)
