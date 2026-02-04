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
    mol = Chem.RemoveHs(mol)
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
    id_to_hex_template = {h.id: h for h in template_tile}
    path_signature = []
    for (u, v) in template_path:
        h1 = id_to_hex_template[u]
        h2 = id_to_hex_template[v]
        path_signature.append((h1.row, h1.relative_col, h2.row, h2.relative_col))

    global_cutting_plan = {}
    for tidx, tile_hexes in all_tiles.items():
        coord_map = {(h.row, h.relative_col): h.id for h in tile_hexes}
        current_tile_path = []
        for (r1, c1, r2, c2) in path_signature:
            if (r1, c1) not in coord_map or (r2, c2) not in coord_map: return None
            u_id = coord_map[(r1, c1)]
            v_id = coord_map[(r2, c2)]
            if v_id not in full_adj.get(u_id, []): return None
            current_tile_path.append((u_id, v_id))
        global_cutting_plan[tidx] = current_tile_path
    return global_cutting_plan
