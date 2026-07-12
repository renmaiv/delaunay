"""FastAPI application: analysis endpoints + static SPA serving."""
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from server.analysis.orchestrator import AnalysisOrchestrator
from server.analysis.store import AnalysisStore, submit_analysis
from server.config_loader import load_config
from server.parsing.transcript_parser import TranscriptParseError, parse_transcript
from server.schemas import AnalysisResponse
from server.taxonomy import load_taxonomy

_DIST = Path(__file__).resolve().parent.parent / "frontend" / "dist"


@asynccontextmanager
async def lifespan(app: FastAPI):
    config = load_config(os.environ.get("APP_CONFIG_PATH", "config.yaml"))
    app.state.config = config
    app.state.orchestrator = AnalysisOrchestrator(config)
    app.state.store = AnalysisStore()
    yield


app = FastAPI(title="Semantic Observability", lifespan=lifespan)


def get_orchestrator(request: Request) -> AnalysisOrchestrator:
    return request.app.state.orchestrator


def get_store(request: Request) -> AnalysisStore:
    return request.app.state.store


@app.post("/api/analyze", response_model=AnalysisResponse)
async def analyze(
    request: Request,
    file: UploadFile = File(...),
    orchestrator: AnalysisOrchestrator = Depends(get_orchestrator),
    store: AnalysisStore = Depends(get_store),
):
    config = request.app.state.config
    analysis_cfg = config.get_analysis_config()
    max_bytes = int(analysis_cfg.get("max_upload_bytes", 2097152))
    max_turns = int(analysis_cfg.get("max_turns", 500))
    sync_threshold = int(analysis_cfg.get("sync_turn_threshold", 30))

    data = await file.read()
    if len(data) > max_bytes:
        raise HTTPException(status_code=413, detail=f"file too large (max {max_bytes} bytes)")

    try:
        conv = parse_transcript(data, file.filename or "upload")
    except TranscriptParseError as e:
        raise HTTPException(status_code=400, detail=str(e))

    if len(conv.turns) > max_turns:
        raise HTTPException(
            status_code=400,
            detail=f"conversation has {len(conv.turns)} turns; max is {max_turns}",
        )

    if len(conv.turns) <= sync_threshold:
        result = orchestrator.analyze(conv)
        import uuid
        return AnalysisResponse(
            analysis_id=uuid.uuid4().hex, status="completed", progress=1.0, result=result,
        )

    analysis_id = submit_analysis(store, orchestrator, conv)
    return JSONResponse(
        status_code=202,
        content=AnalysisResponse(
            analysis_id=analysis_id, status="pending", progress=0.0,
        ).model_dump(),
    )


@app.get("/api/analysis/{analysis_id}", response_model=AnalysisResponse)
async def get_analysis(analysis_id: str, store: AnalysisStore = Depends(get_store)):
    resp = store.get(analysis_id)
    if resp is None:
        raise HTTPException(status_code=404, detail="unknown analysis id")
    return resp


@app.get("/api/taxonomy")
async def taxonomy():
    return load_taxonomy()


@app.get("/api/health")
async def health(orchestrator: AnalysisOrchestrator = Depends(get_orchestrator)):
    return {"status": "ok", **orchestrator.capabilities()}


if _DIST.is_dir():
    app.mount("/", StaticFiles(directory=str(_DIST), html=True), name="spa")
else:
    @app.get("/")
    async def _no_frontend():
        return {"detail": "frontend not built; run npm run build in frontend/"}
