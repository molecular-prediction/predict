import logging
import os
import tempfile
import uuid
from pathlib import Path
from typing import Optional

from fastapi import BackgroundTasks, FastAPI, File, Form, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
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

# 存储 pipeline 结果（POST后 redirect 到 GET 结果页时使用）
# {run_id: {"result": ..., "artifacts_view": [...], "input_smiles": str, "error": str}}
_results_store = {}


def _read_smiles(file_path: str) -> str:
    """读取 smi 文件的第一行内容。"""
    p = Path(file_path)
    if not p.exists():
        return ""
    text = p.read_text(encoding="utf-8", errors="ignore").strip()
    return text.splitlines()[0].strip() if text else ""


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
    """后台逐条评判，每完成一条就更新 _judging_store，前端轮询可实时获取进度。"""
    provider = OpenAILLMProvider.from_env()
    store = _judging_store[run_id]
    # 初始化每个 artifact 的评判结果槽
    store["artifacts"] = [
        {"base_name": item.base_name, "judgements": []} for item in artifacts
    ]

    if provider is None:
        # 无 provider，直接标记全部为 disabled
        for art_idx, item in enumerate(artifacts):
            for smile_file in item.capped_smile_files:
                smile = _read_smiles(smile_file)
                if not smile:
                    continue
                j = {"smile": smile, "judgment": "未配置 LLM_API_KEY，跳过评判。",
                     "model": "", "status": "disabled", "error": ""}
                store["artifacts"][art_idx]["judgements"].append(j)
        store["done"] = True
        return

    total_prompt = 0
    total_completion = 0
    for art_idx, item in enumerate(artifacts):
        for smile_file in item.capped_smile_files:
            smile = _read_smiles(smile_file)
            if not smile:
                continue
            try:
                result = provider.judge_smiles(smile)
                j = {"smile": result.smile, "judgment": result.judgment,
                     "model": result.model, "status": result.status, "error": result.error or ""}
            except Exception as exc:
                j = {"smile": smile, "judgment": "", "model": provider.model,
                     "status": "error", "error": str(exc)}
            store["artifacts"][art_idx]["judgements"].append(j)
            # 实时更新 token 用量
            total_prompt = provider.total_prompt_tokens
            total_completion = provider.total_completion_tokens
            store["token_usage"] = {
                "prompt_tokens": total_prompt,
                "completion_tokens": total_completion,
                "total_tokens": total_prompt + total_completion,
                "model": provider.model,
            }
    store["done"] = True


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


@app.post("/run")
async def run_web(
    request: Request,
    background_tasks: BackgroundTasks,
    smile_text: str = Form(default=""),
    smile_file: Optional[UploadFile] = File(default=None),
):
    temp_input_path = None
    try:
        if smile_file and smile_file.filename:
            file_bytes = await smile_file.read()
            # 原逻辑：saved_input = save_uploaded_smile_file(smile_file.filename, file_bytes)
            fd, temp_input_path = tempfile.mkstemp(suffix=".smi")
            with os.fdopen(fd, "wb") as tmp_f:
                tmp_f.write(file_bytes)
            saved_input = Path(temp_input_path)
        elif smile_text.strip():
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
        run_id = uuid.uuid4().hex[:12]
        _judging_store[run_id] = {"done": False, "artifacts": [], "token_usage": {}}
        _results_store[run_id] = {
            "result": result,
            "artifacts_view": [_artifact_to_view(item) for item in result.artifacts],
            "input_smiles": result.input_smiles,
            "error": None,
        }
        background_tasks.add_task(_judge_artifacts_background, run_id, result.artifacts)
        # POST-Redirect-GET: 重定向到结果页，刷新不会重新提交表单
        return RedirectResponse(url=f"/result/{run_id}", status_code=303)
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


@app.get("/result/{run_id}", response_class=HTMLResponse)
def result_page(request: Request, run_id: str):
    """GET 结果页：刷新安全，不会重新提交表单。"""
    stored = _results_store.get(run_id)
    if not stored:
        return _render_index(
            request,
            {
                "result": None,
                "artifacts_view": [],
                "error": "结果已过期或不存在，请重新提交。",
                "saved_input": None,
                "input_smiles": "",
                "judging_async": False,
                "run_id": "",
            },
        )
    return _render_index(
        request,
        {
            "result": stored["result"],
            "artifacts_view": stored["artifacts_view"],
            "error": stored["error"],
            "saved_input": None,
            "input_smiles": stored["input_smiles"],
            "judging_async": True,
            "run_id": run_id,
        },
    )


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
