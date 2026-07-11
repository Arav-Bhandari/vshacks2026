"""In-process WebSocket connection manager for pipeline progress events."""
from collections import defaultdict

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

router = APIRouter()

_connections: dict[str, set[WebSocket]] = defaultdict(set)


async def broadcast(session_id: str, event: dict) -> None:
    for ws in list(_connections.get(session_id, ())):
        try:
            await ws.send_json(event)
        except Exception:
            pass


@router.websocket("/ws/{session_id}")
async def ws_endpoint(websocket: WebSocket, session_id: str):
    await websocket.accept()
    _connections[session_id].add(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        _connections[session_id].discard(websocket)
