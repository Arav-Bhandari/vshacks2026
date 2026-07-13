"""Export USDM data as JSON or XML."""
import re
import xml.etree.ElementTree as ET
from datetime import datetime, timezone


def export_usdm_json(usdm: dict) -> dict:
    return {
        "usdmVersion": "3.0",
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "study": usdm.get("study", usdm),
    }


def export_usdm_xml(usdm: dict) -> str:
    root = ET.Element("USDM", version="3.0")
    _to_xml(usdm, root)
    return ET.tostring(root, encoding="unicode")


def _safe_tag(key: str) -> str:
    tag = re.sub(r"[^a-zA-Z0-9_]", "_", str(key)) or "item"
    return f"_{tag}" if tag[0].isdigit() else tag


def _to_xml(obj, parent: ET.Element) -> None:
    if isinstance(obj, dict):
        for key, val in obj.items():
            _to_xml(val, ET.SubElement(parent, _safe_tag(key)))
    elif isinstance(obj, list):
        for item in obj:
            _to_xml(item, ET.SubElement(parent, "item"))
    else:
        parent.text = "" if obj is None else str(obj)
