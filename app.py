import logging
import os
import tempfile
import uuid
from pathlib import Path
from typing import Optional

from fastapi import BackgroundTasks, FastAPI, File, Form, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request

from gnr_service import (
    BASE_DIR,
    MONOMER_IMG_DIR,
    PHOTO_DIR,
    SMILE_CAPPED_DIR,
    SMILE_DIR,
    SMILE_RAW_DIR,
    clean_pycache,
    ensure_output_dirs,
    judge_artifacts,
    run_generation_pipeline,
    save_input_smile_file,
    save_uploaded_smile_file,
)
from llm_provider import OpenAILLMProvider

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)

app = FastAPI(title="TH.Xie GNR Web")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

ensure_output_dirs()

app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
app.mount("/smile", StaticFiles(directory=str(SMILE_DIR)), name="smile")
app.mount("/photo", StaticFiles(directory=str(PHOTO_DIR)), name="photo")
app.mount("/predict_smile_raw", StaticFiles(directory=str(SMILE_RAW_DIR)), name="predict_smile_raw")
app.mount("/predict_smile_capped", StaticFiles(directory=str(SMILE_CAPPED_DIR)), name="predict_smile_capped")
app.mount("/monomer_imgs", StaticFiles(directory=str(MONOMER_IMG_DIR)), name="monomer_imgs")

# 存储后台评判状态（内存，进程级生命周期）
# {run_id: {"done": bool, "artifacts": [...], "token_usage": {...}}}
_judging_store = {}


def _to_web_url(path: Optional[str]) -> Optional[str]:
    if not path:
        return None
    rel = Path(path).resolve().relative_to(BASE_DIR).as_posix()
    return f"/{rel}"


def _artifact_to_view(item):
    return {
        "base_name": item.base_name,
        "k": item.k,
        "variant": item.variant,
        "cut_image": _to_web_url(item.cut_image),
        "raw_smiles": item.raw_smiles,
        "raw_smile_files": [_to_web_url(p) for p in item.raw_smile_files],
        "capped_smiles": item.capped_smiles,
        "capped_smile_files": [_to_web_url(p) for p in item.capped_smile_files],
        "monomer_images": [_to_web_url(p) for p in item.monomer_images],
        "smile_judgements": [
            {
                "smile": judgement.smile,
                "judgment": judgement.judgment,
                "model": judgement.model,
                "status": judgement.status,
                "error": judgement.error,
            }
            for judgement in item.smile_judgements
        ],
    }


def _judge_artifacts_background(run_id: str, artifacts):
    """后台评判，结果写入 _judging_store 供前端轮询。"""
    provider = OpenAILLMProvider.from_env()
    judge_artifacts(artifacts, provider)
    # 收集结果
    results = []
    for item in artifacts:
        results.append({
            "base_name": item.base_name,
            "judgements": [
                {
                    "smile": j.smile,
                    "judgment": j.judgment,
                    "model": j.model,
                    "status": j.status,
                    "error": j.error,
                }
                for j in item.smile_judgements
            ],
        })
    token_usage = {}
    if provider:
        token_usage = {
            "prompt_tokens": provider.total_prompt_tokens,
            "completion_tokens": provider.total_completion_tokens,
            "total_tokens": provider.total_tokens,
            "model": provider.model,
        }
    _judging_store[run_id] = {"done": True, "artifacts": results, "token_usage": token_usage}


def _render_index(request: Request, context: dict):
    return templates.TemplateResponse(request, "index.html", {"request": request, **context})


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return _render_index(
        request,
        {
            "result": None,
            "artifacts_view": [],
            "error": None,
            "saved_input": None,
            "input_smiles": "",
            "judging_async": False,
            "run_id": "",
        },
    )


@app.post("/run", response_class=HTMLResponse)
async def run_web(
    request: Request,
    background_tasks: BackgroundTasks,
    smile_text: str = Form(default=""),
    smile_file: Optional[UploadFile] = File(default=None),
):
    try:
        temp_input_path = None
        if smile_file and smile_file.filename:
            file_bytes = await smile_file.read()
            # 不再把上传的 smile 文件持久化到 smile/ 目录（避免生成 web_*.smi 文件）。
            # 原逻辑：saved_input = save_uploaded_smile_file(smile_file.filename, file_bytes)
            fd, temp_input_path = tempfile.mkstemp(suffix=".smi")
            with os.fdopen(fd, "wb") as tmp_f:
                tmp_f.write(file_bytes)
            saved_input = Path(temp_input_path)
        elif smile_text.strip():
            # 不再把网页输入的 smile 持久化到 smile/ 目录（避免生成 web_*.smi 文件）。
            # 原逻辑：saved_input = save_input_smile_file(smile_text.strip())
            fd, temp_input_path = tempfile.mkstemp(suffix=".smi")
            with os.fdopen(fd, "w", encoding="utf-8") as tmp_f:
                tmp_f.write(smile_text.strip() + "\n")
            saved_input = Path(temp_input_path)
        else:
            return _render_index(
                request,
                {
                    "result": None,
                    "artifacts_view": [],
                    "error": "请上传 smile 文件，或者直接输入 smile 码。",
                    "saved_input": None,
                    "input_smiles": "",
                    "judging_async": False,
                    "run_id": "",
                },
            )

        result = run_generation_pipeline(str(saved_input))
        # 为本次运行分配唯一 ID，供前端轮询评判结果
        run_id = uuid.uuid4().hex[:12]
        _judging_store[run_id] = {"done": False, "artifacts": [], "token_usage": {}}
        background_tasks.add_task(_judge_artifacts_background, run_id, result.artifacts)
        return _render_index(
            request,
            {
                "result": result,
                "artifacts_view": [_artifact_to_view(item) for item in result.artifacts],
                "error": None,
                "saved_input": str(saved_input),
                "input_smiles": result.input_smiles,
                "judging_async": True,
                "run_id": run_id,
            },
        )
    except Exception as exc:
        return _render_index(
            request,
            {
                "result": None,
                "artifacts_view": [],
                "error": f"运行失败: {exc}",
                "saved_input": None,
                "input_smiles": smile_text.strip(),
                "judging_async": False,
                "run_id": "",
            },
        )
    finally:
        if temp_input_path and os.path.exists(temp_input_path):
            try:
                os.remove(temp_input_path)
            except OSError:
                pass
        clean_pycache()


@app.get("/api/judgements/{run_id}")
def get_judgements(run_id: str):
    """前端轮询接口：返回当前评判进度、结果和 token 用量。"""
    entry = _judging_store.get(run_id)
    if entry is None:
        return JSONResponse({"error": "run_id not found"}, status_code=404)
    return JSONResponse(entry)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
