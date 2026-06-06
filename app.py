import logging
import os
import tempfile
from pathlib import Path
from typing import Optional

from fastapi import BackgroundTasks, FastAPI, File, Form, UploadFile
from fastapi.responses import HTMLResponse
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


def _judge_artifacts_background(artifacts):
    judge_artifacts(artifacts)


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
            # 改为写入临时文件，pipeline 读取后在 finally 中删除。
            fd, temp_input_path = tempfile.mkstemp(suffix=".smi")
            with os.fdopen(fd, "wb") as tmp_f:
                tmp_f.write(file_bytes)
            saved_input = Path(temp_input_path)
        elif smile_text.strip():
            # 不再把网页输入的 smile 持久化到 smile/ 目录（避免生成 web_*.smi 文件）。
            # 原逻辑：saved_input = save_input_smile_file(smile_text.strip())
            # 改为写入临时文件，pipeline 读取后在 finally 中删除。
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
                },
            )

        result = run_generation_pipeline(str(saved_input))
        background_tasks.add_task(_judge_artifacts_background, result.artifacts)
        return _render_index(
            request,
            {
                "result": result,
                "artifacts_view": [_artifact_to_view(item) for item in result.artifacts],
                "error": None,
                "saved_input": str(saved_input),
                "input_smiles": result.input_smiles,
                "judging_async": True,
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
            },
        )
    finally:
        # 删除网页输入产生的临时 smile 文件（如果有）。
        if temp_input_path and os.path.exists(temp_input_path):
            try:
                os.remove(temp_input_path)
            except OSError:
                pass
        clean_pycache()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
