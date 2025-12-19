import networkx as nx
import matplotlib.pyplot as plt
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

    # 辅助属性：相对列索引（用于跨Tile匹配）
    relative_col: int = 0


@dataclass
class Edge:
    hex1_id: int
    hex2_id: int


# ==========================================
# 2. GNR Loader & Bridge
# ==========================================

def read_smiles_and_generate_coords(file_path: str):
    """读取SMILES并生成2D坐标"""
    if not os.path.exists(file_path):
        print(f"[警告] 文件 {file_path} 未找到，使用内置测试分子")
        # 使用一个足够长的 GNR 片段
        smiles = "C1=CC2=C(C=C1)C3=CC4=C(C=C3)C5=CC6=C(C=C5)C7=CC8=C(C=C7)C9=CC%10=C(C=C9)C%11=CC=C(C=C%11)C%10=C8C6=C42"
    else:
        with open(file_path, 'r') as f:
            smiles = f.readline().strip()

    mol = Chem.MolFromSmiles(smiles)
    if not mol: raise ValueError("SMILES 解析失败")
    AllChem.Compute2DCoords(mol)
    return mol


def mol_to_hex_grid(mol) -> Tuple[List[BenzeneHex], int]:
    """RDKit 分子 -> Hex 列表"""
    ssr = Chem.GetSymmSSSR(mol)
    conf = mol.GetConformer()
    if not ssr: raise ValueError("无六元环")

    raw_rings = []
    for ring in ssr:
        if len(ring) != 6: continue
        xs = [conf.GetAtomPosition(idx).x for idx in ring]
        ys = [conf.GetAtomPosition(idx).y for idx in ring]
        cx, cy = sum(xs) / 6.0, sum(ys) / 6.0

        # 估算大小
        p1 = conf.GetAtomPosition(ring[0])
        p2 = conf.GetAtomPosition(ring[1])
        dist = math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)
        raw_rings.append({"cx": cx, "cy": cy, "size": dist, "indices": list(ring)})

    # 网格化
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
            h = BenzeneHex(hid, r_idx, c_idx, ring['cx'], ring['cy'], ring['size'], ring['indices'])
            hexes.append(h)
            hid += 1
            if c_idx > max_col: max_col = c_idx

    return hexes, max_col + 1


def build_edges_and_adj_geometric(hexes: List[BenzeneHex]) -> Tuple[List[Edge], Dict[int, List[int]]]:
    """基于物理距离建立邻接表"""
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
# 3. 核心升级：寻找所有路径的 PathFinder
# ==========================================

# ==========================================
# 3. 核心升级：支持自定义横向长度的 PathFinder
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

        # --- 【关键参数】在此处修改 ---
        # 允许连续横向移动的最大边数
        # 1 = 旧模式 (走1步横的必须拐弯)
        # 2 = 允许连续走2步横的 (涉及3个苯环)
        # 3 = 允许连续走3步横的
        # 这个参数比较关键 其可以被选择为1 2 3 代表横切一次最多能跨越的苯环数 现在默认选择为2 后续还需要继续优化
        self.MAX_HORIZ_RUN = 2
        # ---------------------------

    def find_all_paths(self):
        """寻找所有可能的切割路径"""
        self.start_time = time.time()
        self.found_paths = []

        all_rows = [h.row for h in self.hexes]
        if not all_rows: return []
        top_row, bottom_row = min(all_rows), max(all_rows)

        top_hexes = [h for h in self.hexes if h.row == top_row]

        for start_hex in sorted(top_hexes, key=lambda h: len(self.full_adj.get(h.id, []))):
            if self._should_stop(): break

            visited = {start_hex.id}
            # 修改点：将原来的 False (布尔值) 改为 0 (计数器)
            self._dfs(start_hex.id, visited, [], {start_hex.row}, bottom_row, current_horiz_run=0)

        return self.found_paths

    def _dfs(self, curr_id, visited, path, covered_rows, bottom_row, current_horiz_run):
        if self._should_stop(): return

        curr_h = self.id_to_hex[curr_id]

        if curr_h.row == bottom_row:
            self.found_paths.append(path.copy())
            return

        raw_neighbors = self.full_adj.get(curr_id, [])
        valid_neighbors = [nid for nid in raw_neighbors if nid in self.id_to_hex]

        # 排序：优先尝试“非横向”的移动，这样更容易到底部
        # 横向移动 (row不变) 排在后面
        neighbors_sorted = sorted(valid_neighbors, key=lambda n: (
            1 if self.id_to_hex[n].row == curr_h.row else 0,  # 横向的排后面
            self.id_to_hex[n].row,
            len(self.full_adj[n])
        ))

        for nei_id in neighbors_sorted:
            if self._should_stop(): return
            if nei_id in visited: continue

            nei_h = self.id_to_hex[nei_id]
            if nei_h.row < curr_h.row: continue  # 不走回头路(不往上走)

            # --- 【逻辑修改核心】 ---
            is_horiz = (nei_h.row == curr_h.row)

            if is_horiz:
                # 如果这一步是横着走，检查是否超标
                if current_horiz_run >= self.MAX_HORIZ_RUN:
                    continue  # 超过最大步数，禁止走
                new_horiz_run = current_horiz_run + 1
            else:
                # 如果这一步是斜着/竖着走，计数器归零
                new_horiz_run = 0

            # -----------------------

            visited.add(nei_id)
            path.append((min(curr_id, nei_id), max(curr_id, nei_id)))

            self._dfs(nei_id, visited, path, covered_rows | {nei_h.row}, bottom_row, new_horiz_run)

            path.pop()
            visited.remove(nei_id)

    def _should_stop(self):
        return (time.time() - self.start_time > self.time_limit) or (len(self.found_paths) >= self.max_paths)
# ==========================================
# 4. 模式匹配与绘图
# ==========================================

def partition_into_tiles(hexes: List[BenzeneHex], k_cols: int) -> Dict[int, List[BenzeneHex]]:
    """分块，并标记相对列坐标"""
    groups = {}
    for h in hexes:
        tile_index = h.col // k_cols
        h.relative_col = h.col % k_cols  # 关键：标记它在块内的相对位置
        groups.setdefault(tile_index, []).append(h)
    return groups


def apply_path_to_all_tiles(template_path: List[Tuple[int, int]],
                            template_tile: List[BenzeneHex],
                            all_tiles: Dict[int, List[BenzeneHex]],
                            full_adj: Dict[int, List[int]]) -> Dict[int, List[Tuple[int, int]]]:
    """
    尝试将 template_path (在第一个块里找到的路径) 推广到所有块。
    如果成功，返回所有块的切割边；如果失败（某块切不了），返回 None。
    """
    id_to_hex_template = {h.id: h for h in template_tile}

    # 1. 将路径转换为“相对坐标签名”
    # 签名格式: [(u_row, u_rel_col, v_row, v_rel_col), ...]
    path_signature = []
    for (u, v) in template_path:
        h1 = id_to_hex_template[u]
        h2 = id_to_hex_template[v]
        path_signature.append((h1.row, h1.relative_col, h2.row, h2.relative_col))

    # 2. 尝试在其他块中复现这个签名
    global_cutting_plan = {}

    for tidx, tile_hexes in all_tiles.items():
        # 建立相对坐标索引: (row, rel_col) -> id
        coord_map = {(h.row, h.relative_col): h.id for h in tile_hexes}

        current_tile_path = []
        for (r1, c1, r2, c2) in path_signature:
            # 检查这个块里有没有这两个对应位置的苯环
            if (r1, c1) not in coord_map or (r2, c2) not in coord_map:
                return None  # 结构不完整，匹配失败

            u_id = coord_map[(r1, c1)]
            v_id = coord_map[(r2, c2)]

            # 检查这两个环在物理上是不是连着的 (在adj里)
            # (虽然坐标对应，但可能中间缺键)
            is_connected = False
            if v_id in full_adj.get(u_id, []):
                is_connected = True

            if not is_connected:
                return None  # 物理连接不存在，匹配失败

            current_tile_path.append((u_id, v_id))

        global_cutting_plan[tidx] = current_tile_path

    return global_cutting_plan


def draw_multi_cut_result(all_hexes, tiles_cutting_edges, k, variant_idx, filename):
    fig, ax = plt.subplots(figsize=(12, 5))

    for h in all_hexes:
        circle = plt.Circle((h.cx, h.cy), h.size * 0.9, color='#F0F0F0', fill=True)
        ax.add_patch(circle)
        edge_poly = plt.Circle((h.cx, h.cy), h.size * 0.9, color='gray', fill=False, linewidth=0.5)
        ax.add_patch(edge_poly)

    id_map = {h.id: h for h in all_hexes}
    for tile_idx, path in tiles_cutting_edges.items():
        for (u, v) in path:
            h1, h2 = id_map[u], id_map[v]
            ax.plot([h1.cx, h2.cx], [h1.cy, h2.cy], color='red', linewidth=2.5, zorder=10)

    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(f"Cut Method K={k} | Variant {variant_idx}", fontsize=14)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
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

    print(f">>> 步骤2: 开始尝试多种切割方案 (K=1 ~ {max_k_attempts})...")

    found_any_global = False

    for k in range(1, max_k_attempts + 1):
        if k > total_width: continue

        # A. 分块
        tiles = partition_into_tiles(hexes, k_cols=k)
        if not tiles: continue

        # B. 选取第一个块作为“模板” (Template Tile)
        # 假设第一个块(Index 0)是完整的，我们就在它上面找所有切法
        template_tile = tiles.get(0)
        if not template_tile: continue  # 没找到第0块

        # 只有当这块Tile有“厚度”时才切
        if len({h.row for h in template_tile}) <= 1: continue

        # C. 在模板块上寻找【所有】可能的路径
        # 这里传入局部的 id_to_hex (template_tile)，防止跑出去
        template_finder = EdgeCuttingPathFinder(template_tile, all_adj)
        all_possible_paths = template_finder.find_all_paths()

        if not all_possible_paths:
            # print(f"    K={k} 模板块未找到路径")
            continue

        print(f"    K={k}: 在模板块中找到了 {len(all_possible_paths)} 种潜在切法，正在验证全局匹配...")

        # D. 验证每一种切法是否能应用到其他所有块
        variant_count = 0
        for path in all_possible_paths:
            global_plan = apply_path_to_all_tiles(path, template_tile, tiles, all_adj)

            if global_plan:
                variant_count += 1
                found_any_global = True
                out_name = os.path.join(save_dir, f"cut_method_k{k}_v{variant_count}.png")
                draw_multi_cut_result(hexes, global_plan, k, variant_count, out_name)

                # 限制每个K值最多输出前5种变体，避免图片太多
                if variant_count >= 5:
                    break

        if variant_count == 0:
            print(f"    K={k} 方案不可行 (无法全局匹配)")

    if not found_any_global:
        print("\n未找到任何有效的切割方案。")
    else:
        print(f"\n全部完成！图片已保存在 {save_dir} 文件夹。")