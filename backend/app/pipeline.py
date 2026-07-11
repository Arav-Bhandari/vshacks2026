"""Async orchestration of the upload -> analysis pipeline."""
import asyncio
import logging
import traceback

from app.config import UPLOAD_DIR
from app.database import db
from app.routes.ws import broadcast
from app.services.baseline import compute_baseline
from app.services.burden_analyzer import analyze_burden
from app.services.pdf_parser import parse_pdf
from app.services.similarity_engine import find_similar
from app.services.usdm_converter import convert_to_usdm

logger = logging.getLogger(__name__)


async def _progress(session_id: str, step: str, status: str, detail: str, pct: int):
    event = {"step": step, "status": status, "detail": detail, "pct": pct}
    db.update_session(session_id, progress=event)
    await broadcast(session_id, event)


async def _with_retry(coro_fn, *args):
    try:
        return await coro_fn(*args)
    except Exception:
        return await coro_fn(*args)


async def run_pipeline(session_id: str) -> None:
    session = db.get_session(session_id)
    if not session:
        return
    pdf_path = str(UPLOAD_DIR / session["filename"])

    try:
        await _progress(session_id, "parse", "running", "Extracting PDF text", 5)
        markdown = await asyncio.to_thread(parse_pdf, pdf_path)
        db.update_session(session_id, markdown=markdown)
        await _progress(session_id, "parse", "done", "PDF parsed", 15)

        await _progress(session_id, "usdm", "running", "Converting to USDM", 20)
        usdm = await _with_retry(convert_to_usdm, markdown)
        db.update_session(session_id, usdm=usdm)
        await _progress(session_id, "usdm", "done", "USDM built", 30)

        await _progress(session_id, "similar", "running", "Finding similar trials", 40)
        similar_trials = await asyncio.to_thread(find_similar, usdm, 50)
        db.update_session(session_id, similar_trials=similar_trials)
        await _progress(session_id, "similar", "done", "Similar trials found", 50)

        await _progress(session_id, "baseline", "running", "Computing baseline", 52)
        baseline = compute_baseline(similar_trials, 10)
        db.update_session(session_id, baseline=baseline)
        await _progress(session_id, "baseline", "done", "Baseline computed", 55)

        await _progress(session_id, "burden", "running", "Analyzing burden", 58)
        burden = analyze_burden(usdm)
        db.update_session(session_id, burden=burden)
        await _progress(session_id, "burden", "done", "Burden analyzed", 60)

        await _progress(session_id, "ml", "running", "Predicting duration/risk", 65)
        from app.ml.predictor import predict_duration_risk
        ml_prediction = await asyncio.to_thread(
            predict_duration_risk, usdm, baseline, burden
        )
        db.update_session(session_id, ml_prediction=ml_prediction)
        await _progress(session_id, "ml", "done", "Prediction complete", 70)

        await _progress(session_id, "fda", "running", "Checking FDA compliance", 75)
        from app.services.fda_analyzer import analyze_fda_compliance
        fda_analysis = await _with_retry(analyze_fda_compliance, usdm)
        db.update_session(session_id, fda_analysis=fda_analysis)
        await _progress(session_id, "fda", "done", "FDA analysis complete", 85)

        await _progress(session_id, "optimize", "running", "Optimizing protocol", 90)
        from app.services.optimizer import optimize_protocol
        optimized = await _with_retry(
            optimize_protocol, usdm, similar_trials, fda_analysis, burden
        )
        db.update_session(
            session_id, optimized_protocol=optimized, status="complete"
        )
        await _progress(session_id, "optimize", "done", "Pipeline complete", 100)

    except Exception as exc:
        logger.error("pipeline failed for %s\n%s", session_id, traceback.format_exc())
        db.update_session(session_id, status="error")
        await _progress(session_id, "pipeline", "error", str(exc), 0)
