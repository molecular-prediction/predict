import os
import math
from typing import List, Tuple, Dict
from rdkit import Chem
from rdkit.Chem import AllChem
from gnr_types import BenzeneHex, Edge

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

    sorted_by_y = sorted(raw_rings, key=lambda x: x['cy'], reverse=True)
    rows = []
    current_row = [sorted_by_y[0]]
    Y_THRESHOLD = 0.5
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

def apply_path_to_all_tiles(template_path: List[Tuple[int, int]],
                            template_tile: List[BenzeneHex],
                            all_tiles: Dict[int, List[BenzeneHex]],
                            full_adj: Dict[int, List[int]]) -> Dict[int, List[Tuple[int, int]]]:
    """
    将模板路径应用到所有图块，支持跨图块（周期性）连接。
    核心逻辑：
    1. 解析模板路径中每一步的“相对位移”（Shift）。
    2. 如果模板中 u->v 是跨越边界的（例如从左边缘跳到右边缘），
       则在全局应用时，连接 Tile[i] 的 u 和 Tile[i + shift] 的 v。
    """
    id_to_hex_template = {h.id: h for h in template_tile}
    
    # 1. 解析路径签名，包含 shift 信息
    # Signature item: (r1, c1, r2, c2, tile_shift)
    path_signature_with_shift = []
    
    # 获取K值（通过模板的最大相对列推断，或者假设满填充）
    # 更稳妥的是通过 template_tile 计算
    if not template_tile: return {}
    max_rel_col = max(h.relative_col for h in template_tile)
    k_cols = max_rel_col + 1 # 近似值，用于启发式判断
    
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

    global_cutting_plan = {}
    
    # 2. 全局应用
    # 遍历每一个 tile 作为起点 tile
    for tidx, tile_hexes in all_tiles.items():
        # 建立当前 tile 的坐标映射 (row, rel_col) -> global_id
        current_coord_map = {(h.row, h.relative_col): h.id for h in tile_hexes}
        
        current_tile_path = []
        valid_path_for_this_tile = True
        
        for (r1, c1, r2, c2, shift) in path_signature_with_shift:
            # 起点必须在当前 tile 中
            if (r1, c1) not in current_coord_map:
                valid_path_for_this_tile = False
                break
            u_id = current_coord_map[(r1, c1)]
            
            # 终点可能在 neighbor tile 中
            target_tile_idx = tidx + shift
            target_tile_hexes = all_tiles.get(target_tile_idx)
            
            if not target_tile_hexes:
                # 如果目标 tile 不存在（比如边缘），则跳过这一条边，还是视为非法？
                # 对于无限延伸的模拟，如果切出界了，通常就不切了（或者该链条在此断开）。
                # 为了保持严谨，如果连不上，我们这里视为断开，不添加该边。
                # 但不应该让整个 plan 失败，只是这一条边没有了。
                continue
                
            target_coord_map = {(h.row, h.relative_col): h.id for h in target_tile_hexes}
            
            if (r2, c2) not in target_coord_map:
                # 目标点在目标 tile 里没有（可能形状不规则）
                continue
                
            v_id = target_coord_map[(r2, c2)]
            
            # 验证物理连通性
            if v_id in full_adj.get(u_id, []):
                current_tile_path.append((u_id, v_id))
            else:
                # 逻辑上应该连通但物理上不连通？可能是距离阈值问题，或者判断失误
                # 这里做个保险
                pass

        if current_tile_path:
            global_cutting_plan[tidx] = current_tile_path
            
    return global_cutting_plan
