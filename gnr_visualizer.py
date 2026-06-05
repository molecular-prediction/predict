import os
from pathlib import Path

_CACHE_DIR = Path(__file__).resolve().parent / ".cache"
(_CACHE_DIR / "matplotlib").mkdir(parents=True, exist_ok=True)
(_CACHE_DIR / "fontconfig").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE_DIR / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE_DIR))
os.environ.setdefault("FC_CACHEDIR", str(_CACHE_DIR / "fontconfig"))

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from typing import List, Dict, Tuple
from gnr_types import BenzeneHex, GlobalCutPlan


def _estimate_tile_width(all_hexes: List[BenzeneHex], k: int) -> float:
    """Estimate the horizontal period used only to unwrap periodic cut lines."""
    by_coord = {(h.row, h.col): h for h in all_hexes}
    widths = []
    for h in all_hexes:
        counterpart = by_coord.get((h.row, h.col + k))
        if counterpart is not None:
            widths.append(counterpart.cx - h.cx)
    if widths:
        widths.sort()
        return widths[len(widths) // 2]

    xs = sorted({round(h.cx, 6) for h in all_hexes})
    if len(xs) < 2:
        return 0.0
    spacings = [b - a for a, b in zip(xs, xs[1:]) if b > a]
    if not spacings:
        return 0.0
    spacings.sort()
    return spacings[len(spacings) // 2] * k


def _unwrap_path_segments(
    path: List[Tuple[int, int]],
    id_map: Dict[int, BenzeneHex],
    tile_width: float,
) -> List[Tuple[Tuple[float, float], Tuple[float, float]]]:
    if not path:
        return []

    unwrapped = []
    previous_end = None
    for u, v in path:
        h1, h2 = id_map[u], id_map[v]
        start = (h1.cx, h1.cy)
        end = (h2.cx, h2.cy)

        if previous_end is not None and abs(tile_width) > 1e-6:
            best_shift = min(
                range(-4, 5),
                key=lambda shift: (
                    start[0] + shift * tile_width - previous_end[0]
                ) ** 2 + (start[1] - previous_end[1]) ** 2,
            )
            start = (start[0] + best_shift * tile_width, start[1])
            end = (end[0] + best_shift * tile_width, end[1])

        unwrapped.append((start, end))
        previous_end = end

    return unwrapped


def _shift_segment_near(
    start: Tuple[float, float],
    end: Tuple[float, float],
    target: Tuple[float, float],
    tile_width: float,
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    if abs(tile_width) <= 1e-6:
        return start, end

    best_shift = min(
        range(-4, 5),
        key=lambda shift: (
            start[0] + shift * tile_width - target[0]
        ) ** 2 + (start[1] - target[1]) ** 2,
    )
    return (
        (start[0] + best_shift * tile_width, start[1]),
        (end[0] + best_shift * tile_width, end[1]),
    )


def draw_multi_cut_result(all_hexes: List[BenzeneHex], tiles_cutting_edges: Dict[int, List[Tuple[int, int]]] | GlobalCutPlan, k: int, variant_idx: int, filename: str):
    fig, ax = plt.subplots(figsize=(12, 5))
    endpoint_extensions = []
    if isinstance(tiles_cutting_edges, GlobalCutPlan):
        endpoint_extensions = tiles_cutting_edges.endpoint_extensions
        tiles_cutting_edges = tiles_cutting_edges.cutting_edges

    all_removed_ids = set()
    for path in tiles_cutting_edges.values():
        for (u, v) in path:
            all_removed_ids.add(u)
            all_removed_ids.add(v)

    for h in all_hexes:
        fc = '#FFDDDD' if h.id in all_removed_ids else '#F0F0F0'
        ec = 'red' if h.id in all_removed_ids else 'gray'
        poly = Polygon(h.vertices, closed=True, facecolor=fc, edgecolor=ec, linewidth=1.0, zorder=1)
        ax.add_patch(poly)

    id_map = {h.id: h for h in all_hexes}
    tile_width = _estimate_tile_width(all_hexes, k)
    unwrapped_paths = {}
    for tile_idx, path in tiles_cutting_edges.items():
        unwrapped_paths[tile_idx] = _unwrap_path_segments(path, id_map, tile_width)
        for start, end in unwrapped_paths[tile_idx]:
            ax.plot([start[0], end[0]], [start[1], end[1]], color='red', linewidth=2.5, zorder=10)

    for extension in endpoint_extensions:
        start, end = extension.start, extension.end
        path_segments = unwrapped_paths.get(extension.tile_index, [])
        if path_segments:
            target = path_segments[0][0] if extension.edge == "top" else path_segments[-1][1]
            start, end = _shift_segment_near(start, end, target, tile_width)
        ax.plot(
            [start[0], end[0]],
            [start[1], end[1]],
            color='red',
            linewidth=2.5,
            zorder=10,
            linestyle='-',
        )

    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(f"Cut Method K={k} | Variant {variant_idx}", fontsize=14)
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
