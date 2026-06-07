import logging
import os
import tempfile
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
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

# run_id -> 创建时间戳（用于 TTL 清理）
_run_created_at = {}
# 保护上面三个 store 的顶层增删（后台线程与请求线程并发访问）
_store_lock = threading.Lock()

# 已完成的结果在内存中保留的时长（秒）；超过后惰性清理
_RESULT_TTL_SECONDS = 30 * 60
# 内存中最多保留的 run 数量（硬上限，防止极端堆积）
_MAX_RUNS = 200


def _register_run(run_id: str) -> None:
    """登记一个新 run 并触发惰性清理。"""
    with _store_lock:
        _run_created_at[run_id] = time.time()
        _evict_runs_locked()


def _discard_run(run_id: str) -> None:
    """彻底移除一个 run 的所有内存条目。调用方需自行持锁或保证无并发。"""
    _judging_store.pop(run_id, None)
    _results_store.pop(run_id, None)
    _run_created_at.pop(run_id, None)


def _evict_runs_locked() -> None:
    """清理过期/超量的 run。必须在持有 _store_lock 时调用。

    只回收「已完成」(done=True) 的 run，避免删除后台线程仍在写入的在跑条目。
    """
    now = time.time()
    # 1) TTL：已完成且超过保留时长的条目
    for run_id, created in list(_run_created_at.items()):
        if now - created < _RESULT_TTL_SECONDS:
            continue
        entry = _judging_store.get(run_id)
        if entry is None or entry.get("done"):
            _discard_run(run_id)

    # 2) 硬上限：若仍超量，按创建时间从旧到新淘汰已完成的条目
    if len(_run_created_at) <= _MAX_RUNS:
        return
    for run_id, _ in sorted(_run_created_at.items(), key=lambda kv: kv[1]):
        if len(_run_created_at) <= _MAX_RUNS:
            break
        entry = _judging_store.get(run_id)
        if entry is None or entry.get("done"):
            _discard_run(run_id)


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
    """后台并行评判：线程池提交所有 SMILES，as_completed 在每个 future 返回时写回 store，前端轮询实时获取进度。"""
    provider = OpenAILLMProvider.from_env()
    store = _judging_store[run_id]
    lock = threading.Lock()
    # 预分配每个 artifact 内每条 SMILES 的固定槽位，保证并行回填后顺序稳定
    store["artifacts"] = []
    tasks = []  # (art_idx, slot_idx, smile)
    for art_idx, item in enumerate(artifacts):
        slots = []
        for smile_file in item.capped_smile_files:
            smile = _read_smiles(smile_file)
            if not smile:
                continue
            slot_idx = len(slots)
            slots.append({"smile": smile, "judgment": "", "model": "",
                          "status": "pending", "error": ""})
            tasks.append((art_idx, slot_idx, smile))
        store["artifacts"].append({"base_name": item.base_name, "judgements": slots})

    if provider is None:
        # 无 provider，直接把全部槽位标记为 disabled
        for art_idx, slot_idx, smile in tasks:
            store["artifacts"][art_idx]["judgements"][slot_idx] = {
                "smile": smile, "judgment": "未配置 LLM_API_KEY，跳过评判。",
                "model": "", "status": "disabled", "error": ""}
        store["done"] = True
        return

    if not tasks:
        store["done"] = True
        return

    def _judge_one(smile):
        try:
            r = provider.judge_smiles(smile)
            return {"smile": r.smile, "judgment": r.judgment, "model": r.model,
                    "status": r.status, "error": r.error or "",
                    "prompt_tokens": r.prompt_tokens, "completion_tokens": r.completion_tokens}
        except Exception as exc:
            return {"smile": smile, "judgment": "", "model": provider.model,
                    "status": "error", "error": str(exc),
                    "prompt_tokens": 0, "completion_tokens": 0}

    total_prompt = 0
    total_completion = 0
    with ThreadPoolExecutor(max_workers=100) as executor:
        future_map = {
            executor.submit(_judge_one, smile): (art_idx, slot_idx)
            for art_idx, slot_idx, smile in tasks
        }
        for future in as_completed(future_map):
            art_idx, slot_idx = future_map[future]
            j = future.result()
            prompt_tokens = j.get("prompt_tokens", 0)
            completion_tokens = j.get("completion_tokens", 0)
            with lock:
                store["artifacts"][art_idx]["judgements"][slot_idx] = j
                total_prompt += prompt_tokens
                total_completion += completion_tokens
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
        # 登记 run 并惰性清理已过期/超量的旧条目，防止内存无限增长
        _register_run(run_id)
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
