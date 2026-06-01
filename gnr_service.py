import os
import re
import time
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from gnr_graph import (
    apply_path_to_all_tiles,
    build_edges_and_adj_geometric,
    mol_to_hex_grid,
    partition_into_tiles,
    read_smiles_and_generate_coords,
)
from gnr_pathfinder import EdgeCuttingPathFinder
from llm_provider import OpenAILLMProvider, SmileJudgement
from gnr_smiles import generate_monomer_smiles_periodic
from gnr_visualizer import draw_multi_cut_result


logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
SMILE_DIR = BASE_DIR / "smile"
PHOTO_DIR = BASE_DIR / "photo"
SMILE_RAW_DIR = BASE_DIR / "predict_smile_raw"
SMILE_CAPPED_DIR = BASE_DIR / "predict_smile_capped"
MONOMER_IMG_DIR = BASE_DIR / "monomer_imgs"

OUTPUT_DIRS = [PHOTO_DIR, SMILE_RAW_DIR, SMILE_CAPPED_DIR, MONOMER_IMG_DIR]


@dataclass
class OutputArtifact:
    base_name: str
    k: int
    variant: int
    cut_image: Optional[str]
    raw_smiles: List[str]
    raw_smile_files: List[str]
    capped_smiles: List[str]
    capped_smile_files: List[str]
    monomer_images: List[str]
    smile_judgements: List[SmileJudgement]


@dataclass
class PredictionRun:
    input_file: str
    input_smiles: str
    total_width: int
    hex_count: int
    found_any_global: bool
    artifacts: List[OutputArtifact]
    message: str


def ensure_output_dirs() -> None:
    for directory in [SMILE_DIR, *OUTPUT_DIRS]:
        directory.mkdir(parents=True, exist_ok=True)


def clean_pycache() -> None:
    for root, dirs, _files in os.walk(BASE_DIR):
        for dirname in dirs:
            if dirname == "__pycache__":
                path = Path(root) / dirname
                try:
                    import shutil

                    shutil.rmtree(path)
                except Exception:
                    pass


def save_input_smile_file(smile_text: str, filename: Optional[str] = None) -> Path:
    ensure_output_dirs()
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    safe_name = Path(filename).name if filename else f"web_input_{timestamp}.smi"
    if not safe_name:
        safe_name = f"web_input_{timestamp}.smi"
    if not safe_name.endswith(".smi"):
        safe_name = f"{safe_name}.smi"

    target = SMILE_DIR / f"web_{timestamp}_{safe_name}"
    target.write_text(smile_text.strip() + "\n", encoding="utf-8")
    return target


def save_uploaded_smile_file(upload_filename: str, file_bytes: bytes) -> Path:
    ensure_output_dirs()
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    safe_name = Path(upload_filename).name or f"upload_{timestamp}.smi"
    if not safe_name.endswith(".smi"):
        safe_name = f"{safe_name}.smi"
    target = SMILE_DIR / f"web_{timestamp}_{safe_name}"
    target.write_bytes(file_bytes)
    return target


def _read_smiles_from_file(file_path: Path) -> str:
    if not file_path.exists():
        return ""
    text = file_path.read_text(encoding="utf-8", errors="ignore").strip()
    return text.splitlines()[0].strip() if text else ""


def _clear_artifact_files(base_name: str) -> None:
    patterns = [
        PHOTO_DIR / f"{base_name}.png",
        SMILE_RAW_DIR / f"{base_name}_raw*.smi",
        SMILE_CAPPED_DIR / f"{base_name}_capped*.smi",
        MONOMER_IMG_DIR / f"{base_name}_monomer*.png",
    ]
    for pattern in patterns:
        for path in pattern.parent.glob(pattern.name):
            try:
                path.unlink()
            except OSError:
                logger.warning("Failed to remove stale artifact: %s", path)


def _clear_previous_cut_outputs() -> None:
    for directory, pattern in [
        (PHOTO_DIR, "cut_method_*.png"),
        (SMILE_RAW_DIR, "cut_method_*.smi"),
        (SMILE_CAPPED_DIR, "cut_method_*.smi"),
        (MONOMER_IMG_DIR, "cut_method_*.png"),
    ]:
        for path in directory.glob(pattern):
            try:
                path.unlink()
            except OSError:
                logger.warning("Failed to remove previous cut output: %s", path)


def _collect_outputs_since(start_ts: float) -> List[OutputArtifact]:
    artifact_map: Dict[str, OutputArtifact] = {}

    raw_pattern = re.compile(r"^(cut_method_k(\d+)_v(\d+))_raw(?:_(\d+))?$")
    capped_pattern = re.compile(r"^(cut_method_k(\d+)_v(\d+))_capped(?:_(\d+))?$")
    image_pattern = re.compile(r"^(cut_method_k(\d+)_v(\d+))_monomer(?:_(\d+))?$")
    cut_pattern = re.compile(r"^(cut_method_k(\d+)_v(\d+))$")

    def get_artifact(base_name: str, k: int, variant: int) -> OutputArtifact:
        if base_name not in artifact_map:
            artifact_map[base_name] = OutputArtifact(
                base_name=base_name,
                k=k,
                variant=variant,
                cut_image=None,
                raw_smiles=[],
                raw_smile_files=[],
                capped_smiles=[],
                capped_smile_files=[],
                monomer_images=[],
                smile_judgements=[],
            )
        return artifact_map[base_name]

    for path in PHOTO_DIR.glob("*.png"):
        if path.stat().st_mtime < start_ts:
            continue
        match = cut_pattern.match(path.stem)
        if not match:
            continue
        base_name, k_str, variant_str = match.groups()
        artifact = get_artifact(base_name, int(k_str), int(variant_str))
        artifact.cut_image = str(path)

    for path in SMILE_RAW_DIR.glob("*.smi"):
        if path.stat().st_mtime < start_ts:
            continue
        match = raw_pattern.match(path.stem)
        if not match:
            continue
        base_name, k_str, variant_str, _suffix = match.groups()
        artifact = get_artifact(base_name, int(k_str), int(variant_str))
        artifact.raw_smile_files.append(str(path))
        artifact.raw_smiles.append(_read_smiles_from_file(path))

    for path in SMILE_CAPPED_DIR.glob("*.smi"):
        if path.stat().st_mtime < start_ts:
            continue
        match = capped_pattern.match(path.stem)
        if not match:
            continue
        base_name, k_str, variant_str, _suffix = match.groups()
        artifact = get_artifact(base_name, int(k_str), int(variant_str))
        artifact.capped_smile_files.append(str(path))
        artifact.capped_smiles.append(_read_smiles_from_file(path))

    for path in MONOMER_IMG_DIR.glob("*.png"):
        if path.stat().st_mtime < start_ts:
            continue
        match = image_pattern.match(path.stem)
        if not match:
            continue
        base_name, k_str, variant_str, _suffix = match.groups()
        artifact = get_artifact(base_name, int(k_str), int(variant_str))
        artifact.monomer_images.append(str(path))

    for artifact in artifact_map.values():
        artifact.raw_smile_files.sort()
        artifact.capped_smile_files.sort()
        artifact.monomer_images.sort()

    return sorted(artifact_map.values(), key=lambda item: (item.k, item.variant, item.base_name))


def _judge_artifacts(artifacts: List[OutputArtifact], provider: Optional[OpenAILLMProvider]) -> None:
    for artifact in artifacts:
        logger.info(
            "Start judging artifact: base_name=%s k=%s variant=%s capped_count=%s",
            artifact.base_name,
            artifact.k,
            artifact.variant,
            len(artifact.capped_smile_files),
        )
        judgements: List[SmileJudgement] = []
        for index, smile_file in enumerate(artifact.capped_smile_files, start=1):
            smile = _read_smiles_from_file(Path(smile_file))
            if not smile:
                logger.warning(
                    "Skip empty SMILES file: artifact=%s file=%s",
                    artifact.base_name,
                    smile_file,
                )
                continue
            if provider is None:
                logger.warning(
                    "LLM provider unavailable, skipping judgement: artifact=%s index=%s smiles=%s",
                    artifact.base_name,
                    index,
                    smile,
                )
                judgements.append(
                    SmileJudgement(
                        smile=smile,
                        judgment="未配置 LLM_API_KEY，跳过评判。",
                        model="",
                        status="disabled",
                        error="OpenAI provider unavailable",
                    )
                )
                continue
            try:
                logger.info(
                    "Judging SMILES: artifact=%s index=%s smiles=%s",
                    artifact.base_name,
                    index,
                    smile,
                )
                result = provider.judge_smiles(smile)
                logger.info(
                    "Judgement done: artifact=%s index=%s model=%s status=%s",
                    artifact.base_name,
                    index,
                    result.model,
                    result.status,
                )
                judgements.append(result)
            except Exception as exc:
                logger.exception(
                    "Judgement failed: artifact=%s index=%s smiles=%s",
                    artifact.base_name,
                    index,
                    smile,
                )
                judgements.append(
                    SmileJudgement(
                        smile=smile,
                        judgment="",
                        model=provider.model,
                        status="error",
                        error=str(exc),
                    )
                )
        artifact.smile_judgements = judgements
        logger.info(
            "Finished judging artifact: base_name=%s success_count=%s",
            artifact.base_name,
            sum(1 for item in judgements if item.status == "ok"),
        )


def run_pipeline(
    input_file: str,
    max_k_attempts: int = 5,
    llm_provider: Optional[OpenAILLMProvider] = None,
) -> PredictionRun:
    ensure_output_dirs()
    _clear_previous_cut_outputs()
    start_ts = time.time()
    provider = llm_provider if llm_provider is not None else OpenAILLMProvider.from_env()
    logger.info("Pipeline started: input_file=%s provider=%s", input_file, type(provider).__name__ if provider else "None")

    mol = read_smiles_and_generate_coords(input_file)
    hexes, total_width = mol_to_hex_grid(mol)
    _edges, all_adj = build_edges_and_adj_geometric(hexes)

    found_any_global = False

    for k in range(1, max_k_attempts + 1):
        if k > total_width:
            continue

        tiles = partition_into_tiles(hexes, k_cols=k)
        if not tiles:
            continue

        # 动态寻找具有最完整高度（跨越行数最多）的图块作为模板，而不是死板地取 Tile 0
        best_tidx = max(tiles.keys(), key=lambda t: len({h.row for h in tiles[t]}))
        template_tile = tiles[best_tidx]

        if len({h.row for h in template_tile}) <= 1:
            continue

        template_finder = EdgeCuttingPathFinder(template_tile, all_adj, k_cols=k)
        all_possible_paths = template_finder.find_all_paths()
        if not all_possible_paths:
            continue

        variant_count = 0
        for path in all_possible_paths:
            global_plan = apply_path_to_all_tiles(path, template_tile, tiles, all_adj)
            if not global_plan.is_complete:
                logger.info(
                    "Skip incomplete global cut plan: k=%s reason=%s",
                    k,
                    global_plan.invalid_reason,
                )
                continue

            variant_count += 1
            found_any_global = True
            base_name = f"cut_method_k{k}_v{variant_count}"
            img_path = PHOTO_DIR / f"{base_name}.png"
            raw_smi_path = SMILE_RAW_DIR / f"{base_name}_raw.smi"
            capped_smi_path = SMILE_CAPPED_DIR / f"{base_name}_capped.smi"
            monomer_img_path = MONOMER_IMG_DIR / f"{base_name}_monomer.png"

            _clear_artifact_files(base_name)
            draw_multi_cut_result(hexes, global_plan, k, variant_count, str(img_path))
            monomer_result = generate_monomer_smiles_periodic(
                mol,
                hexes,
                global_plan,
                k,
                total_width,
                str(raw_smi_path),
                str(capped_smi_path),
                str(monomer_img_path),
            )
            if not monomer_result.is_valid:
                if monomer_result.failure_reason == "no kept hexes after cut":
                    _clear_artifact_files(base_name)
                    variant_count -= 1
                    continue
                logger.info(
                    "Cut plan kept without complete monomer outputs: k=%s variant=%s reason=%s",
                    k,
                    variant_count,
                    monomer_result.failure_reason,
                )

            if variant_count >= 5:
                break

    artifacts = _collect_outputs_since(start_ts)
    _judge_artifacts(artifacts, provider)
    message = "全部完成" if found_any_global else "未找到任何有效的切割方案"
    logger.info(
        "Pipeline finished: input_file=%s found_any_global=%s artifact_count=%s message=%s",
        input_file,
        found_any_global,
        len(artifacts),
        message,
    )

    return PredictionRun(
        input_file=str(Path(input_file).resolve()),
        input_smiles=_read_smiles_from_file(Path(input_file)),
        total_width=total_width,
        hex_count=len(hexes),
        found_any_global=found_any_global,
        artifacts=artifacts,
        message=message,
    )
