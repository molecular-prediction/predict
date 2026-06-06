import os
import math
from typing import List, Tuple, Dict
from rdkit import Chem
from rdkit.Chem import AllChem
from gnr_types import BenzeneHex, CutExtension, Edge, GlobalCutPlan

def read_smiles_and_generate_coords(file_path: str):
    if not os.path.exists(file_path):
        print(f"[警告] 文件 {file_path} 未找到，使用内置测试分子")
        smiles = "C1=CC2=C(C=C1)C3=CC4=C(C=C3)C5=CC6=C(C=C5)C7=CC8=C(C=C7)C9=CC%10=C(C=C9)C%11=CC=C(C=C%11)C%10=C8C6=C42"
    else:
        with open(file_path, 'r') as f:
            smiles = f.readline().strip()

    mol = Chem.MolFromSmiles(smiles)
    if not mol: raise ValueError("SMILES 解析失败")
    mol = Chem.AddHs(mol)
    AllChem.Compute2DCoords(mol)
    # 加上 sanitize=False 避免大分子去氢时进行 Kekulize 验证
    mol = Chem.RemoveHs(mol, sanitize=False)
    return mol


def mol_to_hex_grid(mol) -> Tuple[List[BenzeneHex], int]:
    ssr = Chem.GetSymmSSSR(mol)
    conf = mol.GetConformer()
    if not ssr: raise ValueError("无六元环")

    raw_rings = []
    for ring in ssr:
        if len(ring) != 6: continue

        atom_coords = []
        xs, ys = [], []
        for idx in ring:
            pos = conf.GetAtomPosition(idx)
            atom_coords.append((pos.x, pos.y))
            xs.append(pos.x)
            ys.append(pos.y)

        cx, cy = sum(xs) / 6.0, sum(ys) / 6.0
        atom_coords.sort(key=lambda p: math.atan2(p[1] - cy, p[0] - cx))

        p1 = conf.GetAtomPosition(ring[0])
        p2 = conf.GetAtomPosition(ring[1])
        dist = math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)

        raw_rings.append({
            "cx": cx, "cy": cy, "size": dist,
            "indices": list(ring),
            "vertices": atom_coords
        })

    avg_ring_size = sum(ring["size"] for ring in raw_rings) / len(raw_rings)
    sorted_by_y = sorted(raw_rings, key=lambda x: x['cy'], reverse=True)
    rows = []
    current_row = [sorted_by_y[0]]
    Y_THRESHOLD = avg_ring_size * 0.55
    for ring in sorted_by_y[1:]:
        if abs(ring['cy'] - current_row[0]['cy']) < Y_THRESHOLD:
            current_row.append(ring)
        else:
            rows.append(current_row)
            current_row = [ring]
    rows.append(current_row)

    hexes = []
    hid = 0
    max_col = 0
    for r_idx, row_rings in enumerate(rows):
        row_rings.sort(key=lambda x: x['cx'])
        for c_idx, ring in enumerate(row_rings):
            h = BenzeneHex(
                id=hid, row=r_idx, col=c_idx,
                cx=ring['cx'], cy=ring['cy'], size=ring['size'],
                atom_indices=ring['indices'],
                vertices=ring['vertices']
            )
            hexes.append(h)
            hid += 1
            if c_idx > max_col: max_col = c_idx

    return hexes, max_col + 1


def build_edges_and_adj_geometric(hexes: List[BenzeneHex]) -> Tuple[List[Edge], Dict[int, List[int]]]:
    edges = []
    adj = {h.id: [] for h in hexes}
    if not hexes: return edges, adj
    avg_size = sum(h.size for h in hexes) / len(hexes)
    neighbor_dist_ideal = math.sqrt(3) * avg_size
    tolerance = neighbor_dist_ideal * 0.4
    for i in range(len(hexes)):
        for j in range(i + 1, len(hexes)):
            h1, h2 = hexes[i], hexes[j]
            dist = math.sqrt((h1.cx - h2.cx) ** 2 + (h1.cy - h2.cy) ** 2)
            if abs(dist - neighbor_dist_ideal) < tolerance:
                adj[h1.id].append(h2.id)
                adj[h2.id].append(h1.id)
                edges.append(Edge(h1.id, h2.id))
    return edges, adj

def partition_into_tiles(hexes: List[BenzeneHex], k_cols: int) -> Dict[int, List[BenzeneHex]]:
    groups = {}
    for h in hexes:
        tile_index = h.col // k_cols
        h.relative_col = h.col % k_cols
        groups.setdefault(tile_index, []).append(h)
    return groups


def _cluster_1d(values: List[float], tol: float) -> List[List[float]]:
    """将一维数值按间距 tol 聚类（升序）。path/apply 不依赖此函数。"""
    if not values:
        return []
    sorted_values = sorted(values)
    clusters = [[sorted_values[0]]]
    for value in sorted_values[1:]:
        if abs(value - clusters[-1][-1]) < tol:
            clusters[-1].append(value)
        else:
            clusters.append([value])
    return clusters


def _vertical_stack_heights(hexes: List[BenzeneHex]) -> List[int]:
    """按 cx 把环心聚成竖直堆，返回每堆环数（按 cx 升序）。

    armchair 蜂窝相邻堆错开半周期，grid 的 col 分桶会把这些堆合并，
    因此此处直接用 cx 聚类还原真实的竖直堆叠序列。
    """
    if not hexes:
        return []
    avg_ring_size = sum(h.size for h in hexes) / len(hexes)
    tol = avg_ring_size * 0.6
    centers = [sum(c) / len(c) for c in _cluster_1d([h.cx for h in hexes], tol)]
    if not centers:
        return []
    heights = [0] * len(centers)
    for h in hexes:
        nearest = min(range(len(centers)), key=lambda i: abs(centers[i] - h.cx))
        heights[nearest] += 1
    return heights


def classify_edge_type(hexes: List[BenzeneHex], mol=None) -> str:
    """判别 GNR 边缘类型：返回 "zigzag" 或 "armchair"。

    判据 1（几何，主）：按 cx 聚成竖直堆得到每堆环数 heights，取内部
    core = heights[1:-1]（去掉两端残缺堆）。若 core 内堆高不一致（交替）
    → armchair；否则 → zigzag。

    实测：7ac core 全 1、4.smi core 全 2 → zigzag；
    gnr_3_other core=[3,2,...]、gnr_2_other core=[1,2,...] → armchair。
    """
    heights = _vertical_stack_heights(hexes)
    if len(heights) <= 2:
        # 太短无法判断交替，默认按 zigzag（与现有行为一致，不引入新分支）
        return "zigzag"
    core = heights[1:-1]
    if len(set(core)) > 1:
        return "armchair"
    return "zigzag"


def _minimal_sequence_period(seq: List[int]) -> int:
    """求整数序列的最小重复周期 p（seq[i]==seq[i%p] 对所有 i 成立）。"""
    n = len(seq)
    if n == 0:
        return 1
    for period in range(1, n + 1):
        if all(seq[i] == seq[i % period] for i in range(n)):
            return period
    return n


def infer_minimal_period_cols(hexes: List[BenzeneHex], total_width: int) -> int:
    if total_width <= 1:
        return max(total_width, 1)

    # armchair：grid 的 col 分桶会把错开半周期的竖直堆合并，导致 row_counts
    # 在每列均匀、误判周期=1。改用竖直堆环数序列求真实最小周期（实测=2）。
    if classify_edge_type(hexes) == "armchair":
        heights = _vertical_stack_heights(hexes)
        stack_period = _minimal_sequence_period(heights)
        return max(1, min(stack_period, total_width))

    row_counts_by_col = []
    for col in range(total_width):
        rows = sorted(h.row for h in hexes if h.col == col)
        row_counts_by_col.append(tuple(rows))

    for period in range(1, total_width + 1):
        if total_width % period != 0:
            continue
        ok = True
        for col in range(total_width):
            if row_counts_by_col[col] != row_counts_by_col[col % period]:
                ok = False
                break
        if ok:
            return period
    return total_width


def _direction_from_dx(dx: float) -> str:
    return "right" if dx >= 0 else "left"


def _extension_from_hexes(
    tile_index: int,
    edge: str,
    direction: str,
    endpoint_hex: BenzeneHex,
    neighbor_hex: BenzeneHex,
) -> CutExtension:
    if edge == "top":
        dx = endpoint_hex.cx - neighbor_hex.cx
        dy = endpoint_hex.cy - neighbor_hex.cy
    else:
        dx = endpoint_hex.cx - neighbor_hex.cx
        dy = endpoint_hex.cy - neighbor_hex.cy

    length = math.hypot(dx, dy) or 1.0
    scale = max(endpoint_hex.size * 1.3, length * 0.85) / length
    end = (endpoint_hex.cx + dx * scale, endpoint_hex.cy + dy * scale)
    return CutExtension(
        tile_index=tile_index,
        edge=edge,
        direction=direction,
        start=(endpoint_hex.cx, endpoint_hex.cy),
        end=end,
    )

def apply_path_to_all_tiles(template_path: List[Tuple[int, int]],
                            template_tile: List[BenzeneHex],
                            all_tiles: Dict[int, List[BenzeneHex]],
                            full_adj: Dict[int, List[int]]) -> GlobalCutPlan:
    """
    将模板路径应用到所有图块，支持跨图块（周期性）连接。
    核心逻辑：
    1. 解析模板路径中每一步的“相对位移”（Shift）。
    2. 如果模板中 u->v 是跨越边界的（例如从左边缘跳到右边缘），
       则在全局应用时，连接 Tile[i] 的 u 和 Tile[i + shift] 的 v。
    """
    plan = GlobalCutPlan()
    id_to_hex_template = {h.id: h for h in template_tile}
    
    # 1. 解析路径签名，包含 shift 信息
    # Signature item: (r1, c1, r2, c2, tile_shift)
    path_signature_with_shift = []
    
    # 获取K值（通过模板的最大相对列推断，或者假设满填充）
    # 更稳妥的是通过 template_tile 计算
    if not template_tile:
        plan.invalid_reason = "empty template tile"
        return plan
    if not template_path:
        plan.invalid_reason = "empty template path"
        return plan

    max_rel_col = max(h.relative_col for h in template_tile)
    k_cols = max_rel_col + 1

    top_start_id = template_path[0][0]
    top_next_id = template_path[0][1]
    bottom_prev_id = template_path[-1][0]
    bottom_end_id = template_path[-1][1]
    if any(node_id not in id_to_hex_template for node_id in [top_start_id, top_next_id, bottom_prev_id, bottom_end_id]):
        plan.invalid_reason = "template path contains ids outside template tile"
        return plan

    top_start = id_to_hex_template[top_start_id]
    top_next = id_to_hex_template[top_next_id]
    bottom_prev = id_to_hex_template[bottom_prev_id]
    bottom_end = id_to_hex_template[bottom_end_id]
    top_row = min(h.row for h in template_tile)
    bottom_row = max(h.row for h in template_tile)
    if top_start.row != top_row:
        plan.invalid_reason = "template path does not start on top row"
        return plan
    if bottom_end.row != bottom_row:
        plan.invalid_reason = "template path does not end on bottom row"
        return plan
    top_dx = top_start.cx - top_next.cx
    bottom_dx = bottom_end.cx - bottom_prev.cx
    if abs(top_dx) < 1e-6:
        plan.invalid_reason = "top endpoint has no left/right exit direction"
        return plan
    if abs(bottom_dx) < 1e-6:
        plan.invalid_reason = "bottom endpoint has no left/right exit direction"
        return plan
    plan.top_exit_direction = _direction_from_dx(top_dx)
    plan.bottom_exit_direction = _direction_from_dx(bottom_dx)
    
    for (u, v) in template_path:
        h1 = id_to_hex_template[u]
        h2 = id_to_hex_template[v]
        
        # 启发式判断 Tile Shift
        # 正常邻居 c2 - c1 应该在 [-1, 1] 之间。
        # 如果 c2 - c1 很大（正数），说明 h2 在右边很远，其实是 wrap 到了左边（Tile -1）
        # 如果 c2 - c1 很小（负数），说明 h2 在左边很远，其实是 wrap 到了右边（Tile +1）
        # 注意：这里的 shift 是指 "v 所在的 tile" 相对于 "u 所在的 tile" 的偏移
        
        diff = h2.relative_col - h1.relative_col
        shift = 0
        if diff > 1.5: # 比如 0 -> 3 (K=4), diff=3.  Shift = -1 (h2其实在左边)
            shift = -1
        elif diff < -1.5: # 比如 3 -> 0 (K=4), diff=-3. Shift = +1 (h2其实在右边)
            shift = 1
            
        path_signature_with_shift.append((h1.row, h1.relative_col, h2.row, h2.relative_col, shift))

    template_coords = {(h.row, h.relative_col) for h in template_tile}
    expected_tiles = []
    for tidx, tile_hexes in all_tiles.items():
        tile_coords = {(h.row, h.relative_col) for h in tile_hexes}
        if template_coords.issubset(tile_coords):
            expected_tiles.append(tidx)

    plan.expected_tile_count = len(expected_tiles)
    if not expected_tiles:
        plan.invalid_reason = "no complete tiles matching template"
        return plan
    skipped_tiles = []
    id_map = {h.id: h for hexes in all_tiles.values() for h in hexes}
    
    # 2. 全局应用
    # 遍历每一个 tile 作为起点 tile
    for tidx in sorted(expected_tiles):
        tile_hexes = all_tiles[tidx]
        # 建立当前 tile 的坐标映射 (row, rel_col) -> global_id
        current_coord_map = {(h.row, h.relative_col): h.id for h in tile_hexes}
        
        current_tile_path = []
        skip_reason = ""
        
        for (r1, c1, r2, c2, shift) in path_signature_with_shift:
            # 起点必须在当前 tile 中
            if (r1, c1) not in current_coord_map:
                skip_reason = f"tile {tidx} missing start coordinate {(r1, c1)}"
                break
            u_id = current_coord_map[(r1, c1)]
            
            # 终点可能在 neighbor tile 中
            target_tile_idx = tidx + shift
            target_tile_hexes = all_tiles.get(target_tile_idx)
            
            if not target_tile_hexes:
                skip_reason = f"tile {tidx} missing target tile {target_tile_idx}"
                break
                
            target_coord_map = {(h.row, h.relative_col): h.id for h in target_tile_hexes}
            
            if (r2, c2) not in target_coord_map:
                skip_reason = f"tile {target_tile_idx} missing target coordinate {(r2, c2)}"
                break
                
            v_id = target_coord_map[(r2, c2)]
            
            # 验证物理连通性
            if v_id in full_adj.get(u_id, []):
                current_tile_path.append((u_id, v_id))
            else:
                skip_reason = f"tile {tidx} mapped edge {(u_id, v_id)} is not physically adjacent"
                break

        if skip_reason:
            skipped_tiles.append(skip_reason)
            continue

        if len(current_tile_path) != len(path_signature_with_shift):
            skipped_tiles.append(f"tile {tidx} mapped incomplete path")
            continue

        top_start_global = current_coord_map[(top_start.row, top_start.relative_col)]
        top_next_tile = tidx + path_signature_with_shift[0][4]
        top_next_hexes = all_tiles.get(top_next_tile, [])
        top_next_map = {(h.row, h.relative_col): h for h in top_next_hexes}
        top_next_global_hex = top_next_map.get((top_next.row, top_next.relative_col))

        bottom_end_tile = tidx + path_signature_with_shift[-1][4]
        bottom_end_hexes = all_tiles.get(bottom_end_tile, [])
        bottom_end_map = {(h.row, h.relative_col): h for h in bottom_end_hexes}
        bottom_end_global_hex = bottom_end_map.get((bottom_end.row, bottom_end.relative_col))
        bottom_prev_global = current_coord_map[(bottom_prev.row, bottom_prev.relative_col)]

        if top_next_global_hex is None or top_start_global not in id_map:
            skipped_tiles.append(f"tile {tidx} cannot build top endpoint extension")
            continue
        if bottom_end_global_hex is None or bottom_prev_global not in id_map:
            skipped_tiles.append(f"tile {tidx} cannot build bottom endpoint extension")
            continue

        top_start_global_hex = id_map[top_start_global]
        bottom_prev_global_hex = id_map[bottom_prev_global]
        plan.endpoint_extensions.append(
            _extension_from_hexes(
                tidx,
                "top",
                plan.top_exit_direction or "",
                top_start_global_hex,
                top_next_global_hex,
            )
        )
        plan.endpoint_extensions.append(
            _extension_from_hexes(
                tidx,
                "bottom",
                plan.bottom_exit_direction or "",
                bottom_end_global_hex,
                bottom_prev_global_hex,
            )
        )

        plan.cutting_edges[tidx] = current_tile_path
        plan.complete_tile_count += 1
            
    plan.is_complete = bool(plan.cutting_edges) and bool(plan.endpoint_extensions)
    if not plan.is_complete:
        plan.invalid_reason = "global plan did not complete any tile"
    elif skipped_tiles:
        plan.invalid_reason = "; ".join(skipped_tiles[:3])
    return plan
