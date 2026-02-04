import networkx as nx
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import AllChem
from matplotlib.patches import Polygon
import os
import math
import time
from dataclasses import dataclass
from typing import List, Tuple, Dict, Set


# ==========================================
# 1. 数据结构
# ==========================================

@dataclass
class BenzeneHex:
    id: int
    row: int
    col: int
    cx: float
    cy: float
    size: float
    atom_indices: List[int]
    vertices: List[Tuple[float, float]] = None
    relative_col: int = 0


@dataclass
class Edge:
    hex1_id: int
    hex2_id: int


# ==========================================
# 2. GNR Loader & Bridge
# ==========================================

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


# ==========================================
# 3. 路径搜索 (PathFinder)
# ==========================================

class EdgeCuttingPathFinder:
    def __init__(self, hexes_subset: List[BenzeneHex], full_adj: Dict[int, List[int]]):
        self.hexes = hexes_subset
        self.full_adj = full_adj
        self.id_to_hex = {h.id: h for h in hexes_subset}
        self.found_paths = []
        self.start_time = 0
        self.time_limit = 3.0
        self.max_paths = 20
        self.MAX_HORIZ_RUN = 2

    def find_all_paths(self):
        self.start_time = time.time()
        self.found_paths = []
        all_rows = [h.row for h in self.hexes]
        if not all_rows: return []
        top_row, bottom_row = min(all_rows), max(all_rows)
        top_hexes = [h for h in self.hexes if h.row == top_row]

        for start_hex in sorted(top_hexes, key=lambda h: len(self.full_adj.get(h.id, []))):
            if self._should_stop(): break
            self._dfs(start_hex.id, {start_hex.id}, [], {start_hex.row}, bottom_row, 0)
        return self.found_paths

    def _dfs(self, curr_id, visited, path, covered_rows, bottom_row, current_horiz_run):
        if self._should_stop(): return
        curr_h = self.id_to_hex[curr_id]
        if curr_h.row == bottom_row:
            self.found_paths.append(path.copy())
            return

        raw_neighbors = self.full_adj.get(curr_id, [])
        valid_neighbors = [nid for nid in raw_neighbors if nid in self.id_to_hex]
        neighbors_sorted = sorted(valid_neighbors, key=lambda n: (
            1 if self.id_to_hex[n].row == curr_h.row else 0,
            self.id_to_hex[n].row,
            len(self.full_adj[n])
        ))

        for nei_id in neighbors_sorted:
            if self._should_stop(): return
            if nei_id in visited: continue
            nei_h = self.id_to_hex[nei_id]
            if nei_h.row < curr_h.row: continue

            is_horiz = (nei_h.row == curr_h.row)
            if is_horiz:
                if current_horiz_run >= self.MAX_HORIZ_RUN: continue
                new_horiz_run = current_horiz_run + 1
            else:
                new_horiz_run = 0

            visited.add(nei_id)
            path.append((min(curr_id, nei_id), max(curr_id, nei_id)))
            self._dfs(nei_id, visited, path, covered_rows | {nei_h.row}, bottom_row, new_horiz_run)
            path.pop()
            visited.remove(nei_id)

    def _should_stop(self):
        return (time.time() - self.start_time > self.time_limit) or (len(self.found_paths) >= self.max_paths)


# ==========================================
# 4. SMILES 生成器 (智能滑动窗口提取法)
# ==========================================

def extract_submol_from_hexes(original_mol, hexes_subset: List[BenzeneHex]):
    """辅助函数：根据一组 Hex 生成 RDKit 分子对象"""
    atom_indices = set()
    for h in hexes_subset:
        for aid in h.atom_indices:
            atom_indices.add(aid)

    if not atom_indices: return None

    rw_mol = Chem.RWMol()
    old_to_new_map = {}
    sorted_old_indices = sorted(list(atom_indices))

    for old_idx in sorted_old_indices:
        atom = original_mol.GetAtomWithIdx(old_idx)
        new_idx = rw_mol.AddAtom(Chem.Atom(atom.GetSymbol()))
        rw_mol.GetAtomWithIdx(new_idx).SetFormalCharge(atom.GetFormalCharge())
        old_to_new_map[old_idx] = new_idx

    for old_idx in sorted_old_indices:
        orig_atom = original_mol.GetAtomWithIdx(old_idx)
        for neighbor in orig_atom.GetNeighbors():
            n_idx = neighbor.GetIdx()
            if n_idx in atom_indices and n_idx > old_idx:
                bond = original_mol.GetBondBetweenAtoms(old_idx, n_idx)
                if bond:
                    rw_mol.AddBond(old_to_new_map[old_idx], old_to_new_map[n_idx], bond.GetBondType())

    return rw_mol


def generate_monomer_smiles_periodic(original_mol, all_hexes: List[BenzeneHex],
                                     global_cutting_plan: Dict[int, List[Tuple[int, int]]],
                                     k: int, max_col: int, filename: str):
    """
    智能周期性提取法：
    1. 全局移除红线环。
    2. 在 GNR 中间位置使用滑动窗口 (Sliding Window)，窗口宽度为 K。
    3. 寻找一个相位 (Shift)，使得该窗口内提取出的分子是“最大连通”的。
    这能解决单体跨越 Col 边界导致被切断的问题。
    """

    # 1. 确定全局移除的环
    removed_hex_ids = set()
    for path in global_cutting_plan.values():
        for (u, v) in path:
            removed_hex_ids.add(u)
            removed_hex_ids.add(v)

    # 2. 筛选保留的环
    kept_hexes = [h for h in all_hexes if h.id not in removed_hex_ids]
    if not kept_hexes:
        print("    [失败] 所有环都被切除")
        return

    # 3. 设定滑动窗口的搜索区域 (避免边缘效应，取中间一段)
    # 我们选择分子中间的一个周期作为基准
    mid_col = max_col // 2
    # 将 mid_col 对齐到 K 的倍数作为基准搜索起点
    base_start_col = (mid_col // k) * k
    if base_start_col < 0: base_start_col = 0

    best_mol = None
    max_atoms = 0

    # 4. 尝试 K 种相位偏移 (Shift 0, 1, ..., K-1)
    # 只要有一种偏移能“完美框住”单体（即不切断蓝色键），我们就选它
    for shift in range(k):
        # 窗口范围: [start, end)
        window_start = base_start_col + shift
        window_end = window_start + k

        # 收集窗口内的保留环
        window_hexes = [h for h in kept_hexes if window_start <= h.col < window_end]
        if not window_hexes: continue

        # 生成分子
        mol = extract_submol_from_hexes(original_mol, window_hexes)
        if not mol: continue

        # 检查连通性
        # 我们希望提取出的是一个完整的单体，而不是断裂的碎片
        frags = Chem.GetMolFrags(mol.GetMol(), asMols=True, sanitizeFrags=False)
        if not frags: continue

        # 获取最大的碎片（假设单体是最大的那部分）
        largest_frag = max(frags, key=lambda m: m.GetNumAtoms())

        # 评分：优先选择原子数最多的完整碎片
        # 逻辑是：如果窗口切坏了单体，原子数会变少（因为一部分原子被切到窗口外了）
        # 只有当窗口完美对齐时，原子数才是最大的
        if largest_frag.GetNumAtoms() > max_atoms:
            max_atoms = largest_frag.GetNumAtoms()
            best_mol = largest_frag

    # 5. 保存结果
    if best_mol:
        try:
            Chem.SanitizeMol(best_mol)
            smi = Chem.MolToSmiles(best_mol)
            with open(filename, 'w') as f:
                f.write(smi)
            print(f"    [SMILES] 已保存: {filename} (原子数: {max_atoms})")
        except Exception as e:
            print(f"    [SMILES 错误] {filename}: {e}")
    else:
        print(f"    [SMILES 警告] 未能提取有效单体: {filename}")


# ==========================================
# 5. 绘图与主流程
# ==========================================

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


def draw_multi_cut_result(all_hexes, tiles_cutting_edges, k, variant_idx, filename):
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


if __name__ == "__main__":
    input_file = "smile/gnr_7ac_segment.smi"
    photo_dir = "photo"
    smile_dir = "predict_smile"
    os.makedirs(photo_dir, exist_ok=True)
    os.makedirs(smile_dir, exist_ok=True)

    max_k_attempts = 5

    print(">>> 步骤1: 读取并构建全局图...")
    try:
        mol = read_smiles_and_generate_coords(input_file)
        hexes, total_width = mol_to_hex_grid(mol)
        edges, all_adj = build_edges_and_adj_geometric(hexes)
        print(f"    分子总宽: {total_width} 列, 苯环数: {len(hexes)}")
    except Exception as e:
        print(f"出错: {e}")
        exit()

    print(f">>> 步骤2: 开始尝试多种切割方案 (K=1 ~ {max_k_attempts})...")
    found_any_global = False

    for k in range(1, max_k_attempts + 1):
        if k > total_width: continue

        tiles = partition_into_tiles(hexes, k_cols=k)
        if not tiles or not tiles.get(0): continue
        template_tile = tiles.get(0)

        if len({h.row for h in template_tile}) <= 1: continue

        template_finder = EdgeCuttingPathFinder(template_tile, all_adj)
        all_possible_paths = template_finder.find_all_paths()

        if not all_possible_paths: continue

        print(f"    K={k}: 模板找到 {len(all_possible_paths)} 种路径，正在处理...")

        variant_count = 0
        for path in all_possible_paths:
            global_plan = apply_path_to_all_tiles(path, template_tile, tiles, all_adj)

            if global_plan:
                variant_count += 1
                found_any_global = True

                base_name = f"cut_method_k{k}_v{variant_count}"
                img_path = os.path.join(photo_dir, base_name + ".png")
                draw_multi_cut_result(hexes, global_plan, k, variant_count, img_path)

                # --- 核心修改 ---
                smi_path = os.path.join(smile_dir, base_name + ".smi")
                # 传入 total_width 用于定位中间区域
                generate_monomer_smiles_periodic(mol, hexes, global_plan, k, total_width, smi_path)

                if variant_count >= 5: break

    if not found_any_global:
        print("\n未找到任何有效的切割方案。")
    else:
        print(f"\n全部完成！\n图片路径: {photo_dir}\nSMILES路径: {smile_dir}")