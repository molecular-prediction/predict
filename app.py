from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, UploadFile
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
    run_pipeline,
    save_input_smile_file,
    save_uploaded_smile_file,
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
    }


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "result": None,
            "artifacts_view": [],
            "error": None,
            "saved_input": None,
            "input_smiles": "",
        },
    )


@app.post("/run", response_class=HTMLResponse)
async def run_web(
    request: Request,
    smile_text: str = Form(default=""),
    smile_file: Optional[UploadFile] = File(default=None),
):
    try:
        if smile_file and smile_file.filename:
            file_bytes = await smile_file.read()
            saved_input = save_uploaded_smile_file(smile_file.filename, file_bytes)
        elif smile_text.strip():
            saved_input = save_input_smile_file(smile_text.strip())
        else:
            return templates.TemplateResponse(
                "index.html",
                {
                    "request": request,
                    "result": None,
                    "error": "请上传 smile 文件，或者直接输入 smile 码。",
                    "saved_input": None,
                    "input_smiles": "",
                },
            )

        result = run_pipeline(str(saved_input))
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "result": result,
                "artifacts_view": [_artifact_to_view(item) for item in result.artifacts],
                "error": None,
                "saved_input": str(saved_input),
                "input_smiles": result.input_smiles,
            },
        )
    except Exception as exc:
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "result": None,
                "artifacts_view": [],
                "error": f"运行失败: {exc}",
                "saved_input": None,
                "input_smiles": smile_text.strip(),
            },
        )
    finally:
        clean_pycache()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
