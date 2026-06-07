import os
import re
import time
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from gnr_graph import (
    apply_path_to_all_tiles,
    build_edges_and_adj_geometric,
    classify_edge_type,
    infer_minimal_period_cols,
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
    """并行调用 LLM 评判所有 artifact 的 capped SMILES（并发上限 100）。"""
    if provider is None:
        # 无 provider 时直接标记所有为 disabled，不需要并行
        for artifact in artifacts:
            judgements = []
            for smile_file in artifact.capped_smile_files:
                smile = _read_smiles_from_file(Path(smile_file))
                if not smile:
                    continue
                judgements.append(
                    SmileJudgement(
                        smile=smile,
                        judgment="未配置 LLM_API_KEY，跳过评判。",
                        model="",
                        status="disabled",
                        error="OpenAI provider unavailable",
                    )
                )
            artifact.smile_judgements = judgements
        return

    # 收集所有需要评判的任务：(artifact_index, smile_index, smile_str)
    tasks = []
    for art_idx, artifact in enumerate(artifacts):
        for smi_idx, smile_file in enumerate(artifact.capped_smile_files):
            smile = _read_smiles_from_file(Path(smile_file))
            if not smile:
                continue
            tasks.append((art_idx, smi_idx, smile))

    logger.info("LLM judging: %d SMILES across %d artifacts, max_workers=100", len(tasks), len(artifacts))

    # 预分配结果槽
    results_map: Dict[Tuple[int, int], SmileJudgement] = {}

    def _judge_one(art_idx: int, smi_idx: int, smile: str) -> Tuple[int, int, SmileJudgement]:
        try:
            result = provider.judge_smiles(smile)
            return (art_idx, smi_idx, result)
        except Exception as exc:
            logger.exception("Judgement failed: smile=%s", smile)
            return (art_idx, smi_idx, SmileJudgement(
                smile=smile,
                judgment="",
                model=provider.model,
                status="error",
                error=str(exc),
            ))

    with ThreadPoolExecutor(max_workers=100) as executor:
        futures = {
            executor.submit(_judge_one, art_idx, smi_idx, smile): (art_idx, smi_idx)
            for art_idx, smi_idx, smile in tasks
        }
        for future in as_completed(futures):
            art_idx, smi_idx, judgement = future.result()
            results_map[(art_idx, smi_idx)] = judgement

    # 按原始顺序回填结果
    for art_idx, artifact in enumerate(artifacts):
        judgements = []
        for smi_idx in range(len(artifact.capped_smile_files)):
            if (art_idx, smi_idx) in results_map:
                judgements.append(results_map[(art_idx, smi_idx)])
        artifact.smile_judgements = judgements
        logger.info(
            "Finished judging artifact: base_name=%s success_count=%s",
            artifact.base_name,
            sum(1 for item in judgements if item.status == "ok"),
        )


def judge_artifacts(
    artifacts: List[OutputArtifact],
    provider: Optional[OpenAILLMProvider] = None,
) -> None:
    active_provider = provider if provider is not None else OpenAILLMProvider.from_env()
    logger.info(
        "Judgement started: artifact_count=%s provider=%s",
        len(artifacts),
        type(active_provider).__name__ if active_provider else "None",
    )
    _judge_artifacts(artifacts, active_provider)
    logger.info("Judgement finished: artifact_count=%s", len(artifacts))


def run_generation_pipeline(
    input_file: str,
    max_k_attempts: Optional[int] = 5,
) -> PredictionRun:
    ensure_output_dirs()
    _clear_previous_cut_outputs()
    start_ts = time.time()
    logger.info("Generation pipeline started: input_file=%s", input_file)

    mol = read_smiles_and_generate_coords(input_file)
    hexes, total_width = mol_to_hex_grid(mol)
    minimal_period_cols = infer_minimal_period_cols(hexes, total_width)
    edge_type = classify_edge_type(hexes, mol)
    _edges, all_adj = build_edges_and_adj_geometric(hexes)

    found_any_global = False
    seen_product_signatures = set()
    max_k = total_width if max_k_attempts is None else min(max_k_attempts, total_width)
    # armchair 的周期单元生成（_generate_armchair_monomers）只依赖竖直堆窗口、
    # 不依赖 k；若沿用 zigzag 的 k=1..max_k 全扫描，会把同一批产物在每个 k 下重复
    # 落盘一份（去重签名含 k 拦不住）。armchair 仅在真实堆周期对应的单一 k 产出，
    # 使 cut_method_k*_v* 的 k 语义与 §2 竖直堆周期一致。zigzag 行为一字不变。
    if edge_type == "armchair":
        armchair_k = max(1, min(minimal_period_cols, max_k))
        k_values = [armchair_k]
    else:
        k_values = list(range(1, max_k + 1))
    logger.info(
        "Detected ribbon period: total_width=%s minimal_period_cols=%s max_k=%s edge_type=%s k_values=%s",
        total_width,
        minimal_period_cols,
        max_k,
        edge_type,
        k_values,
    )

    for k in k_values:
        tiles = partition_into_tiles(hexes, k_cols=k)
        if not tiles:
            continue

        max_row_count = max(len({h.row for h in tile_hexes}) for tile_hexes in tiles.values())
        if max_row_count <= 1:
            continue

        variant_count = 0
        template_tiles = [
            tile_hexes
            for _tidx, tile_hexes in sorted(tiles.items())
            if len({h.row for h in tile_hexes}) == max_row_count
            and min(h.col for h in tile_hexes) < minimal_period_cols
        ]

        for template_tile in template_tiles:
            template_hex_by_id = {h.id: h for h in template_tile}
            template_finder = EdgeCuttingPathFinder(template_tile, all_adj, k_cols=k)
            all_possible_paths = template_finder.find_all_paths()
            if not all_possible_paths:
                continue

            for path in all_possible_paths:
                relative_path_signature = []
                for u, v in path:
                    h1 = template_hex_by_id.get(u)
                    h2 = template_hex_by_id.get(v)
                    if h1 is None or h2 is None:
                        continue
                    diff = h2.relative_col - h1.relative_col
                    shift = 0
                    if diff > 1.5:
                        shift = -1
                    elif diff < -1.5:
                        shift = 1
                    relative_path_signature.append(
                        (h1.row, h1.relative_col, h2.row, h2.relative_col, shift)
                    )
                relative_path_signature = tuple(relative_path_signature)

                global_plan = apply_path_to_all_tiles(path, template_tile, tiles, all_adj)
                if not global_plan.is_complete:
                    logger.info(
                        "Skip incomplete global cut plan: k=%s reason=%s",
                        k,
                        global_plan.invalid_reason,
                    )
                    continue

                next_variant = variant_count + 1
                base_name = f"cut_method_k{k}_v{next_variant}"
                img_path = PHOTO_DIR / f"{base_name}.png"
                raw_smi_path = SMILE_RAW_DIR / f"{base_name}_raw.smi"
                capped_smi_path = SMILE_CAPPED_DIR / f"{base_name}_capped.smi"
                monomer_img_path = MONOMER_IMG_DIR / f"{base_name}_monomer.png"

                _clear_artifact_files(base_name)
                monomer_result = generate_monomer_smiles_periodic(
                    mol,
                    hexes,
                    global_plan,
                    k,
                    total_width,
                    str(raw_smi_path),
                    str(capped_smi_path),
                    str(monomer_img_path),
                    edge_type=edge_type,
                )
                if not monomer_result.is_valid:
                    logger.info(
                        "Skip cut plan without complete monomer outputs: k=%s candidate_variant=%s reason=%s",
                        k,
                        next_variant,
                        monomer_result.failure_reason,
                    )
                    continue

                product_signature = (
                    k,
                    tuple(sorted(monomer_result.raw_smiles)),
                )
                if product_signature in seen_product_signatures:
                    logger.info(
                        "Skip duplicate cut product: k=%s candidate_variant=%s top=%s bottom=%s",
                        k,
                        next_variant,
                        global_plan.top_exit_direction,
                        global_plan.bottom_exit_direction,
                    )
                    _clear_artifact_files(base_name)
                    continue
                seen_product_signatures.add(product_signature)

                if monomer_result.capped_smiles:
                    renamed_capped_files = []
                    renamed_monomer_images = []
                    temp_pairs = []
                    for file_index, path_text in enumerate(monomer_result.capped_files, start=1):
                        source = Path(path_text)
                        if not source.exists():
                            continue
                        temp_path = source.with_name(f"{source.stem}.dedupe_tmp{source.suffix}")
                        source.rename(temp_path)
                        final_path = SMILE_CAPPED_DIR / f"{base_name}_capped_{file_index}.smi"
                        temp_pairs.append((temp_path, final_path, renamed_capped_files))
                    for file_index, path_text in enumerate(monomer_result.monomer_images, start=1):
                        source = Path(path_text)
                        if not source.exists():
                            continue
                        temp_path = source.with_name(f"{source.stem}.dedupe_tmp{source.suffix}")
                        source.rename(temp_path)
                        final_path = MONOMER_IMG_DIR / f"{base_name}_monomer_{file_index}.png"
                        temp_pairs.append((temp_path, final_path, renamed_monomer_images))
                    for temp_path, final_path, output_list in temp_pairs:
                        temp_path.rename(final_path)
                        output_list.append(str(final_path))
                    monomer_result.capped_files = renamed_capped_files
                    monomer_result.monomer_images = renamed_monomer_images

                variant_count = next_variant
                found_any_global = True
                draw_multi_cut_result(hexes, global_plan, k, variant_count, str(img_path))

        if variant_count == 0:
            logger.info("No valid cut variants accepted for k=%s", k)

    artifacts = _collect_outputs_since(start_ts)
    message = "全部完成" if found_any_global else "未找到任何有效的切割方案"
    logger.info(
        "Generation pipeline finished: input_file=%s found_any_global=%s artifact_count=%s message=%s",
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


def run_pipeline(
    input_file: str,
    max_k_attempts: Optional[int] = 5,
    llm_provider: Optional[OpenAILLMProvider] = None,
) -> PredictionRun:
    result = run_generation_pipeline(input_file, max_k_attempts=max_k_attempts)
    judge_artifacts(result.artifacts, llm_provider)
    return result
