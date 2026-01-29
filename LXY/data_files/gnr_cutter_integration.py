import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from rdkit import Chem
from rdkit.Chem import AllChem
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
    vertices: List[Tuple[float, float]] = None  # 存储六个顶点的坐标
    relative_col: int = 0


@dataclass
class Edge:
    hex1_id: int
    hex2_id: int


# ==========================================
# 2. GNR Loader & Bridge (几何处理升级)
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
    AllChem.Compute2DCoords(mol)
    return mol


def mol_to_hex_grid(mol) -> Tuple[List[BenzeneHex], int]:
    ssr = Chem.GetSymmSSSR(mol)
    conf = mol.GetConformer()
    if not ssr: raise ValueError("无六元环")

    raw_rings = []
    for ring in ssr:
        if len(ring) != 6: continue

        # 获取顶点并计算中心
        xs = [conf.GetAtomPosition(idx).x for idx in ring]
        ys = [conf.GetAtomPosition(idx).y for idx in ring]
        cx, cy = sum(xs) / 6.0, sum(ys) / 6.0

        # 顶点排序（关键：按角度排序，保证画出的是凸六边形）
        vertices = []
        for idx in ring:
            pos = conf.GetAtomPosition(idx)
            vertices.append((pos.x, pos.y))
        vertices.sort(key=lambda p: math.atan2(p[1] - cy, p[0] - cx))

        p1 = conf.GetAtomPosition(ring[0])
        p2 = conf.GetAtomPosition(ring[1])
        dist = math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)

        raw_rings.append({
            "cx": cx, "cy": cy, "size": dist,
            "indices": list(ring), "vertices": vertices
        })

    # 网格化逻辑
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
            h = BenzeneHex(hid, r_idx, c_idx, ring['cx'], ring['cy'], ring['size'], ring['indices'], ring['vertices'])
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
# 3. 核心路径搜索 (保持不变)
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
# 4. 模式匹配 (保持不变)
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


# ==========================================
# 5. 绘图增强 (六边形 + 甲基处理)
# ==========================================

def get_methyl_vector(direction: str):
    """返回不同方向的单位向量"""
    # 坐标系：y向上为正，x向右为正
    # 根据用户图片：左上(~135°), 右上(~45°), 左下(~225°), 右下(~315°)
    if direction == 'TL': return (-1.0, 1.0)  # Top-Left
    if direction == 'TR': return (1.0, 1.0)  # Top-Right
    if direction == 'BL': return (-1.0, -1.0)  # Bottom-Left
    if direction == 'BR': return (1.0, -1.0)  # Bottom-Right
    return (0, 0)


def draw_methyl_stick(ax, hex_obj: BenzeneHex, direction: str):
    """在六边形上画出甲基（红色短棒）"""
    vec = get_methyl_vector(direction)

    # 1. 寻找最佳附着顶点 (Project Logic)
    # 我们遍历六边形的6个顶点，找到在这个方向上“投影最大”的那个点
    best_v = None
    max_dot = -float('inf')

    for (vx, vy) in hex_obj.vertices:
        # 计算点积： (vx - cx) * dx + (vy - cy) * dy
        # 这里简化直接用坐标，因为方向是对称的
        dot = (vx - hex_obj.cx) * vec[0] + (vy - hex_obj.cy) * vec[1]
        if dot > max_dot:
            max_dot = dot
            best_v = (vx, vy)

    if best_v:
        # 2. 画线：从顶点向外延伸
        stick_len = hex_obj.size * 0.6  # 棒长
        # 归一化向量
        norm = math.sqrt(vec[0] ** 2 + vec[1] ** 2)
        dx = (vec[0] / norm) * stick_len
        dy = (vec[1] / norm) * stick_len

        start_x, start_y = best_v
        end_x, end_y = start_x + dx, start_y + dy

        ax.plot([start_x, end_x], [start_y, end_y], color='red', linewidth=2.5, zorder=20)


def draw_multi_cut_result(all_hexes, tiles_cutting_edges, k, variant_idx,
                          methyl_config, filename):
    """
    methyl_config: Tuple[str, str] -> (Top_Dir, Bottom_Dir)
    例如 ('TL', 'BR') 表示顶部甲基在左上，底部甲基在右下
    """
    fig, ax = plt.subplots(figsize=(12, 5))

    # 1. 绘制所有苯环 (六边形)
    id_map = {h.id: h for h in all_hexes}
    for h in all_hexes:
        poly = Polygon(h.vertices, closed=True, facecolor='#F0F0F0', edgecolor='gray', linewidth=1.0, zorder=1)
        ax.add_patch(poly)

    # 2. 绘制切割路径
    for tile_idx, path in tiles_cutting_edges.items():
        # A. 画路径红线
        path_nodes = set()
        for (u, v) in path:
            h1, h2 = id_map[u], id_map[v]
            ax.plot([h1.cx, h2.cx], [h1.cy, h2.cy], color='red', linewidth=2.5, zorder=10)
            path_nodes.add(u)
            path_nodes.add(v)

        # B. 甲基处理 (Methyl Processing)
        if not path_nodes: continue

        # 找到该路径中 最上面(Row最小) 和 最下面(Row最大) 的苯环
        # 注意：一个Tile可能有多个最上/最下的环，我们取路径中涉及到的
        relevant_hexes = [id_map[uid] for uid in path_nodes]
        min_row = min(h.row for h in relevant_hexes)
        max_row = max(h.row for h in relevant_hexes)

        # 筛选出起止点（可能有多个，取x最小的作为左侧基准，或根据逻辑取一个）
        # 这里假设单线切割，通常只有一个入口和一个出口
        top_candidates = [h for h in relevant_hexes if h.row == min_row]
        bottom_candidates = [h for h in relevant_hexes if h.row == max_row]

        # 简单策略：取x坐标居中的一个，或者列表第一个
        start_hex = top_candidates[0]
        end_hex = bottom_candidates[0]

        # 绘制顶部甲基
        draw_methyl_stick(ax, start_hex, methyl_config[0])  # Top Direction
        # 绘制底部甲基
        draw_methyl_stick(ax, end_hex, methyl_config[1])  # Bottom Direction

    ax.set_aspect('equal')
    ax.axis('off')
    # 标题增加甲基信息
    title_str = f"K={k} | Var {variant_idx} | Methyl: Top-{methyl_config[0]} & Bot-{methyl_config[1]}"
    ax.set_title(title_str, fontsize=12)

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[输出] 已保存: {filename}")


# ==========================================
# 主程序
# ==========================================

if __name__ == "__main__":
    input_file = "smile/gnr_7ac_segment.smi"
    save_dir = "photo"
    os.makedirs(save_dir, exist_ok=True)
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

    print(f">>> 步骤2: 开始尝试多种切割方案...")
    found_any_global = False

    # 定义4种甲基组合 (Top, Bottom)
    # TL=TopLeft, TR=TopRight, BL=BottomLeft, BR=BottomRight
    methyl_combinations = [
        ('TL', 'BL'),  # 1. 上左 + 下左
        ('TL', 'BR'),  # 2. 上左 + 下右
        ('TR', 'BL'),  # 3. 上右 + 下左
        ('TR', 'BR')  # 4. 上右 + 下右
    ]

    for k in range(1, max_k_attempts + 1):
        if k > total_width: continue
        tiles = partition_into_tiles(hexes, k_cols=k)
        if not tiles or not tiles.get(0): continue

        template_tile = tiles.get(0)
        if len({h.row for h in template_tile}) <= 1: continue

        template_finder = EdgeCuttingPathFinder(template_tile, all_adj)
        all_possible_paths = template_finder.find_all_paths()

        if not all_possible_paths: continue

        print(f"    K={k}: 模板找到 {len(all_possible_paths)} 种路径，正在验证并生成变体...")

        path_count = 0
        for path in all_possible_paths:
            global_plan = apply_path_to_all_tiles(path, template_tile, tiles, all_adj)
            if global_plan:
                path_count += 1
                found_any_global = True

                # --- 核心修改：对每一种可行切法，生成4张甲基位置图 ---
                for m_idx, m_config in enumerate(methyl_combinations):
                    # 文件名格式: cut_k{k}_p{path}_m{methyl}.png
                    fname = f"cut_k{k}_path{path_count}_m{m_idx + 1}_{m_config[0]}_{m_config[1]}.png"
                    out_path = os.path.join(save_dir, fname)

                    draw_multi_cut_result(hexes, global_plan, k, path_count,
                                          methyl_config=m_config,
                                          filename=out_path)

                # 限制输出数量，防止太多
                if path_count >= 3: break

    if not found_any_global:
        print("\n未找到任何有效的切割方案。")
    else:
        print(f"\n全部完成！图片已保存在 {save_dir} 文件夹。")