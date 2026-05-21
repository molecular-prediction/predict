import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from typing import List, Dict, Tuple
from gnr_types import BenzeneHex

def draw_multi_cut_result(all_hexes: List[BenzeneHex], tiles_cutting_edges: Dict[int, List[Tuple[int, int]]], k: int, variant_idx: int, filename: str):
    fig, ax = plt.subplots(figsize=(12, 5))

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
    for tile_idx, path in tiles_cutting_edges.items():
        for (u, v) in path:
            h1, h2 = id_map[u], id_map[v]
            ax.plot([h1.cx, h2.cx], [h1.cy, h2.cy], color='red', linewidth=2.5, zorder=10)

    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(f"Cut Method K={k} | Variant {variant_idx}", fontsize=14)
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
