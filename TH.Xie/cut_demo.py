# improved_cut_demo.py -- 基于边切割的改进版本
from dataclasses import dataclass, asdict
from typing import List, Tuple, Dict, Optional, Set
import math, json, time, os, traceback
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import matplotlib.cm as cm

# ===================
# 数据类：苯环与边
# ===================
@dataclass
class BenzeneHex:
    id: int
    row: int
    col: int
    repeat_index: int
    cx: float
    cy: float
    size: float
    tag: str = ""

@dataclass
class Edge:
    hex1_id: int
    hex2_id: int
    is_cut: bool = False
    def __hash__(self):
        return hash((min(self.hex1_id, self.hex2_id), max(self.hex1_id, self.hex2_id)))
    def __eq__(self, other):
        if not isinstance(other, Edge):
            return False
        return {self.hex1_id, self.hex2_id} == {other.hex1_id, other.hex2_id}

# -----------------------
# 网格构建（与之前一致）
# -----------------------
def parse_pattern(pattern_str: str, on_char: str = "1") -> Tuple[List[Tuple[int,int]], int, int]:
    lines = [line.rstrip("\n") for line in pattern_str.strip("\n").splitlines()]
    if not lines:
        raise ValueError("pattern_str is empty")
    width = max(len(line) for line in lines)
    height = len(lines)
    cells = []
    for r, line in enumerate(lines):
        line = line.ljust(width)
        for c, ch in enumerate(line):
            if ch == on_char:
                cells.append((r, c))
    return cells, height, width

def grid_to_center(row: int, col: int, size: float = 1.0) -> Tuple[float, float]:
    w = math.sqrt(3) * size
    v_step = 1.5 * size
    x = w * (col + 0.5 * (row % 2))
    y = - v_step * row
    return x, y

def build_polymer(pattern_str: str, repeat: int, size: float = 1.0) -> Tuple[List[BenzeneHex], int]:
    cells, height, width = parse_pattern(pattern_str)
    hexes: List[BenzeneHex] = []
    hid = 0
    for rep in range(repeat):
        for (r, c) in cells:
            global_col = c + rep * width
            cx, cy = grid_to_center(r, global_col, size=size)
            hexes.append(BenzeneHex(
                id=hid, row=r, col=c, repeat_index=rep, cx=cx, cy=cy, size=size, tag=""
            ))
            hid += 1
    return hexes, width

def neighbor_offsets_for_row(row:int) -> List[Tuple[int,int]]:
    if row % 2 == 0:
        return [(0,-1),(0,1),(-1,0),(1,0),(-1,-1),(1,-1)]
    else:
        return [(0,-1),(0,1),(-1,1),(1,1),(-1,0),(1,0)]

# ===================
# 构建边与邻接表
# ===================
def build_edges_and_adj(hexes: List[BenzeneHex], unit_width: int) -> Tuple[List[Edge], Dict[int, List[int]]]:
    pos_map = {}
    for h in hexes:
        global_col = h.col + h.repeat_index * unit_width
        pos_map[(h.row, global_col)] = h.id
    edges = []
    adj: Dict[int, List[int]] = {h.id: [] for h in hexes}
    edge_set = set()
    for h in hexes:
        global_col = h.col + h.repeat_index * unit_width
        r = h.row
        for dx, dy in neighbor_offsets_for_row(r):
            nr = r + dx
            nc = global_col + dy
            nei = pos_map.get((nr, nc))
            if nei is not None:
                adj[h.id].append(nei)
                edge_key = (min(h.id, nei), max(h.id, nei))
                if edge_key not in edge_set:
                    edges.append(Edge(h.id, nei, False))
                    edge_set.add(edge_key)
    return edges, adj

# ===================
# EdgeCuttingPathFinder（已改）
# ===================
class EdgeCuttingPathFinder:
    """
    找切割路径（基于边），修改点：
      - 首步优先斜向（row != current_row），若存在斜向邻居则第一步只能从这些中选；
      - 禁止路径中连续两条水平边（即上一条移动为水平时，不再选择水平邻居作为下一步）。
    """
    def __init__(self, hexes: List[BenzeneHex], edges: List[Edge], adj: Dict[int, List[int]], unit_width: int):
        self.hexes = hexes
        self.edges = edges
        self.adj = adj
        self.unit_width = unit_width
        self.id_to_hex = {h.id: h for h in hexes}
        self.best_cutting_path: List[Tuple[int,int]] = []
        self.best_visited_hexes: Set[int] = set()
        self.start_time = 0.0
        self.time_limit = None

    def find_cutting_path(self, time_limit: float = 3.0) -> List[Tuple[int, int]]:
        self.start_time = time.time()
        self.time_limit = time_limit
        self.best_cutting_path = []
        self.best_visited_hexes = set()

        # 顶层/底层
        all_rows = [h.row for h in self.hexes]
        top_row = min(all_rows)
        bottom_row = max(all_rows)

        top_hexes = [h for h in self.hexes if h.row == top_row]
        # 按度数小的先尝试（启发式）
        for start_hex in sorted(top_hexes, key=lambda h: len(self.adj[h.id])):
            if self._time_exceeded():
                break
            visited_hexes = {start_hex.id}
            cutting_edges: List[Tuple[int,int]] = []
            covered_rows = {start_hex.row}
            # 初始 last_move_horizontal = False
            self._dfs_cut(start_hex.id, visited_hexes, cutting_edges, covered_rows, bottom_row, last_move_horizontal=False)
            # 如果找到了能到底部的合法路径就可以提前停止
            if self.best_cutting_path:
                # verify last visited hex reaches bottom (保守检查)
                last_nodes = set()
                for e in self.best_cutting_path:
                    last_nodes.add(e[0]); last_nodes.add(e[1])
                # 若找到的路径覆盖了底部行中的节点则接受
                if any(self.id_to_hex[n].row == bottom_row for n in last_nodes):
                    break

        return self.best_cutting_path

    def _dfs_cut(self, current_hex: int, visited_hexes: Set[int], cutting_edges: List[Tuple[int,int]],
                 covered_rows: Set[int], bottom_row: int, last_move_horizontal: bool):
        if self._time_exceeded():
            return

        current_row = self.id_to_hex[current_hex].row

        # 更新最佳路径（以覆盖行数为优先）
        if len(covered_rows) > len({self.id_to_hex[h].row for h in self.best_visited_hexes}):
            self.best_cutting_path = cutting_edges.copy()
            self.best_visited_hexes = visited_hexes.copy()

        # 若已经到底并覆盖足够多行，则记录并返回
        if current_row == bottom_row:
            all_rows = {h.row for h in self.hexes}
            if len(covered_rows) >= len(all_rows) * 0.8:  # 保持你原来的80%覆盖阈值
                self.best_cutting_path = cutting_edges.copy()
                self.best_visited_hexes = visited_hexes.copy()
                return

        # 获取候选邻居（只允许同行或更低行的移动）
        neighbors = self.adj[current_hex]
        # 排序：优先更低的行，再按度数小的
        neighbors_sorted = sorted(neighbors, key=lambda n: (self.id_to_hex[n].row, len(self.adj[n])))

        # --- 规则 A：第一步强制/优先选择斜向（row != current_row） ---
        # 如果当前 cutting_edges 为空（即这是从起点的第一次扩展），尝试只使用斜向邻居列表（若存在）
        if len(cutting_edges) == 0:
            diag_neighbors = [n for n in neighbors_sorted if self.id_to_hex[n].row != current_row]
            if diag_neighbors:
                neighbors_sorted = diag_neighbors  # 强制使用斜向邻居优先集合
                # 仍保持 degree 排序
                neighbors_sorted.sort(key=lambda n: (self.id_to_hex[n].row, len(self.adj[n])))

        # --- 规则 B：如果上一移动是水平（last_move_horizontal == True），优先避免再次水平 ---
        # 我们在遍历时跳过那些会导致连续两条水平边的邻居（即当前 neighbor 与 current 在同一行）
        for nei in neighbors_sorted:
            if self._time_exceeded():
                return
            nei_row = self.id_to_hex[nei].row

            # 只允许 non-upward moves（不能回到上一行）
            if nei_row < current_row:
                continue

            if nei in visited_hexes:
                continue

            # 判断此步是否为水平移动
            is_horizontal = (nei_row == current_row)

            # 禁止连续两步水平：如果上一步是水平且此步也是水平，则跳过该 neighbor
            if last_move_horizontal and is_horizontal:
                # skip to avoid two horizontal moves in a row
                continue

            # 记录切割的边（统一用 (min,max) 形式）
            edge = (min(current_hex, nei), max(current_hex, nei))

            # 走这一条边
            visited_hexes.add(nei)
            cutting_edges.append(edge)
            new_covered = covered_rows | {nei_row}

            # 递归，传递是否为水平移动以供下一步判断
            self._dfs_cut(nei, visited_hexes, cutting_edges, new_covered, bottom_row, last_move_horizontal=is_horizontal)

            # 回溯
            cutting_edges.pop()
            visited_hexes.remove(nei)

            # 如果已经找到很好的路径（覆盖大量节点），可提前返回
            if self.best_cutting_path and len(self.best_visited_hexes) >= len(self.hexes) * 0.9:
                return

        # 如果因为上面严格规则导致无法探索（例如上一步是水平且所有邻居都是水平），
        # 我们容许回退并放宽一次（允许水平）。这是为了避免过度严格而无法找到任何路径。
        # 注意：这部分会在当前递归层只在先前遍历没有产生结果时执行，防止产生连续水平优先级冲突。
        if not self.best_cutting_path:
            # second pass: allow horizontal even if last_move_horizontal==True (fallback)
            for nei in neighbors_sorted:
                if self._time_exceeded():
                    return
                nei_row = self.id_to_hex[nei].row
                if nei_row < current_row:
                    continue
                if nei in visited_hexes:
                    continue
                is_horizontal = (nei_row == current_row)
                # If this neighbor was previously skipped due to last_move_horizontal, now allow it:
                # record and recurse as fallback
                edge = (min(current_hex, nei), max(current_hex, nei))
                visited_hexes.add(nei)
                cutting_edges.append(edge)
                new_covered = covered_rows | {nei_row}
                self._dfs_cut(nei, visited_hexes, cutting_edges, new_covered, bottom_row, last_move_horizontal=is_horizontal)
                cutting_edges.pop()
                visited_hexes.remove(nei)
                if self.best_cutting_path and len(self.best_visited_hexes) >= len(self.hexes) * 0.9:
                    return

    def _time_exceeded(self) -> bool:
        if self.time_limit is None:
            return False
        return (time.time() - self.start_time) > self.time_limit
# -----------------------
# 检查切割后的连通性
# -----------------------
def check_connectivity_after_cut(hexes: List[BenzeneHex], adj: Dict[int, List[int]],
                                 cut_edges: Set[Tuple[int, int]]) -> bool:
    """检查切割后是否仍然连通"""
    if not hexes:
        return True
    
    # 构建去除切割边后的邻接表
    adj_after_cut = {h.id: [] for h in hexes}
    for h_id, neighbors in adj.items():
        for nei in neighbors:
            edge = (min(h_id, nei), max(h_id, nei))
            if edge not in cut_edges:
                adj_after_cut[h_id].append(nei)
    
    # BFS检查连通性
    start = hexes[0].id
    visited = {start}
    queue = [start]
    
    while queue:
        current = queue.pop(0)
        for nei in adj_after_cut[current]:
            if nei not in visited:
                visited.add(nei)
                queue.append(nei)
    
    return len(visited) == len(hexes)

# -----------------------
# 检查相邻tile之间的桥接
# -----------------------
def check_bridge_between_tiles(tile_a: List[BenzeneHex], tile_b: List[BenzeneHex],
                               cut_edges_a: Set[Tuple[int, int]], cut_edges_b: Set[Tuple[int, int]],
                               all_adj: Dict[int, List[int]]) -> bool:
    """检查两个相邻tile之间是否有至少一条未切割的边连接"""
    hex_ids_b = {h.id for h in tile_b}
    
    for hex_a in tile_a:
        for nei in all_adj.get(hex_a.id, []):
            if nei in hex_ids_b:
                edge = (min(hex_a.id, nei), max(hex_a.id, nei))
                # 如果这条边在两个tile中都没有被切割，则存在桥接
                if edge not in cut_edges_a and edge not in cut_edges_b:
                    return True
    
    return False

# -----------------------
# 分区和切割主流程
# -----------------------
def partition_into_tiles(hexes: List[BenzeneHex], unit_width: int, k: int) -> Dict[int, List[BenzeneHex]]:
    """将苯环分成tiles"""
    groups: Dict[int, List[BenzeneHex]] = {}
    tile_size_cols = k * unit_width
    for h in hexes:
        global_col = h.col + h.repeat_index * unit_width
        tile_index = global_col // tile_size_cols
        groups.setdefault(tile_index, []).append(h)
    return groups

def try_cutting_with_edge_tracking(
    pattern_str: str,
    total_repeats: int,
    max_expand: int,
    out_count: int = 3,
    size: float = 0.8,
    time_limit_per_tile: float = 3.0,
    require_equal_tile_size: bool = True,
    require_bridge_between_tiles: bool = True
):
    """主函数：尝试不同的切割方案"""
    
    cwd = os.getcwd()
    print(f"[DEBUG] 当前工作目录 = {cwd}")
    out_dir = os.path.join(cwd, "cuts_out")
    os.makedirs(out_dir, exist_ok=True)
    info_path = os.path.join(out_dir, "_INFO.txt")
    
    try:
        # 构建聚合物
        hexes, unit_width = build_polymer(pattern_str, repeat=total_repeats, size=size)
        print(f"[DEBUG] 构建聚合物: 总苯环数={len(hexes)}, 单元宽度={unit_width}")
        
        unit_count = len([h for h in hexes if h.repeat_index == 0])
        print(f"[DEBUG] 每个最小单元的苯环数 = {unit_count}")
        
        # 构建边和邻接表
        all_edges, all_adj = build_edges_and_adj(hexes, unit_width)
        print(f"[DEBUG] 总边数 = {len(all_edges)}")
        
        methods = []
        
        for k in range(1, max_expand + 1):
            try:
                if require_equal_tile_size and (total_repeats % k != 0):
                    print(f"[DEBUG] 跳过 k={k}，因为 total_repeats % k != 0")
                    continue
                
                # 分割成tiles
                tiles = partition_into_tiles(hexes, unit_width, k)
                expected_per_tile = k * unit_count
                
                if not all(len(lst) == expected_per_tile for lst in tiles.values()):
                    print(f"[DEBUG] k={k} 被拒绝: 每个tile的苯环数不匹配")
                    continue
                
                print(f"[DEBUG] 尝试 k={k}，tile数量={len(tiles)}")
                
                tiles_cutting_edges = {}
                all_ok = True
                
                # 对每个tile寻找切割路径
                for tidx, tile_hexes in sorted(tiles.items()):
                    # 构建该tile的边和邻接表
                    tile_edges, tile_adj = build_edges_and_adj(tile_hexes, unit_width)
                    
                    # 寻找切割路径
                    finder = EdgeCuttingPathFinder(tile_hexes, tile_edges, tile_adj, unit_width)
                    cutting_path = finder.find_cutting_path(time_limit=time_limit_per_tile)
                    
                    if not cutting_path:
                        print(f"[DEBUG] k={k} tile {tidx} 未找到切割路径")
                        all_ok = False
                        break
                    
                    # 检查切割后的连通性
                    cut_edges_set = set(cutting_path)
                    if not check_connectivity_after_cut(tile_hexes, tile_adj, cut_edges_set):
                        print(f"[DEBUG] k={k} tile {tidx} 切割后不连通")
                        all_ok = False
                        break
                    
                    tiles_cutting_edges[tidx] = cutting_path
                    print(f"[DEBUG] k={k} tile {tidx} OK，切割边数 = {len(cutting_path)}")
                
                if not all_ok:
                    continue
                
                # 检查相邻tiles之间的桥接
                if require_bridge_between_tiles:
                    sorted_idxs = sorted(tiles.keys())
                    bridge_ok = True
                    
                    for i in range(len(sorted_idxs) - 1):
                        idx_a = sorted_idxs[i]
                        idx_b = sorted_idxs[i + 1]
                        
                        cut_a = set(tiles_cutting_edges[idx_a])
                        cut_b = set(tiles_cutting_edges[idx_b])
                        
                        if not check_bridge_between_tiles(tiles[idx_a], tiles[idx_b],
                                                         cut_a, cut_b, all_adj):
                            print(f"[DEBUG] k={k} tiles {idx_a}和{idx_b}之间无桥接")
                            bridge_ok = False
                            break
                    
                    if not bridge_ok:
                        continue
                
                methods.append({
                    "k": k,
                    "tile_count": len(tiles),
                    "tile_size_cols": k * unit_width,
                    "tiles_cutting_edges": tiles_cutting_edges,
                    "tiles": tiles
                })
                print(f"[DEBUG] k={k} 接受！")
                
            except Exception as e:
                print(f"[ERROR] 处理 k={k} 时异常: {e}")
                traceback.print_exc()
                continue
        
        # 写入信息文件
        with open(info_path, "w", encoding="utf-8") as f:
            f.write(f"输入模式:\n{pattern_str}\n")
            f.write(f"总重复次数={total_repeats}, 最大扩展={max_expand}\n")
            f.write(f"找到的方法数量={len(methods)}\n")
            f.write("方法k值列表=" + ",".join(str(m["k"]) for m in methods) + "\n")
        
        if not methods:
            nof = os.path.join(out_dir, "无有效方法.txt")
            with open(nof, "w", encoding="utf-8") as f:
                f.write("在给定参数下未找到有效的切割方法。详见 _INFO.txt\n")
            print(f"[INFO] 未找到方法。已写入 {info_path} 和 {nof}")
            return methods, []
        
        # 输出前out_count个方法
        methods = sorted(methods, key=lambda x: x["k"])
        selected = methods[:out_count]
        id_map = {h.id: h for h in hexes}
        
        for method in selected:
            k = method["k"]
            tiles_cutting_edges = method["tiles_cutting_edges"]
            tiles = method["tiles"]
            
            # 生成JSON输出
            out_json = {
                "pattern": pattern_str,
                "unit_width": unit_width,
                "k": k,
                "tile_count": method["tile_count"],
                "tile_size_cols": method["tile_size_cols"],
                "tiles": []
            }
            
            for tidx in sorted(tiles.keys()):
                cutting_edges = tiles_cutting_edges[tidx]
                tile_hexes = tiles[tidx]
                
                out_json["tiles"].append({
                    "tile_index": int(tidx),
                    "cutting_edges": [(int(e[0]), int(e[1])) for e in cutting_edges],
                    "tile_hexes": [asdict(h) for h in tile_hexes]
                })
            
            jf = os.path.join(out_dir, f"method_k{k}.json")
            with open(jf, "w", encoding="utf-8") as f:
                json.dump(out_json, f, ensure_ascii=False, indent=2)
            
            # 生成可视化
            png = os.path.join(out_dir, f"method_k{k}.png")
            draw_cutting_visualization(hexes, id_map, unit_width, k, tiles,
                                      tiles_cutting_edges, png)
            
            print(f"[输出] 已写入 {jf} 和 {png}")
        
        print(f"[完成] 找到方法总数 = {len(methods)}, 输出数量 = {len(selected)}。见 {out_dir}")
        return methods, selected
        
    except Exception as e:
        print("[致命错误] 主流程异常:", e)
        traceback.print_exc()
        with open(info_path, "a", encoding="utf-8") as f:
            f.write("异常:\n")
            f.write(traceback.format_exc())
        return [], []

# -----------------------
# 可视化函数
# -----------------------
def draw_cutting_visualization(all_hexes: List[BenzeneHex], id_map: Dict[int, BenzeneHex],
                               unit_width: int, k: int, tiles: Dict[int, List[BenzeneHex]],
                               tiles_cutting_edges: Dict[int, List[Tuple[int, int]]],
                               filename: str):
    """绘制切割可视化图"""
    fig, ax = plt.subplots(figsize=(14, 5))
    cmap = cm.get_cmap("tab10")
    
    # 绘制所有苯环
    for h in all_hexes:
        verts = []
        for m in range(6):
            ang = math.radians(90 + 60*m)
            x = h.cx + h.size * math.cos(ang)
            y = h.cy + h.size * math.sin(ang)
            verts.append((x, y))
        poly = Polygon(verts, closed=True, edgecolor="lightgray",
                      facecolor="white", linewidth=0.6)
        ax.add_patch(poly)
    
    # 为每个tile绘制切割边
    for i, (tidx, cutting_edges) in enumerate(sorted(tiles_cutting_edges.items())):
        color = cmap(i % 10)
        
        # 绘制切割边（用红色粗线）
        for edge in cutting_edges:
            h1 = id_map[edge[0]]
            h2 = id_map[edge[1]]
            ax.plot([h1.cx, h2.cx], [h1.cy, h2.cy],
                   color='red', linewidth=2.5, alpha=0.7, zorder=10)
        
        # 标注tile编号
        tile_hexes = tiles[tidx]
        xs = [h.cx for h in tile_hexes]
        ys = [h.cy for h in tile_hexes]
        center_x = sum(xs) / len(xs)
        center_y = sum(ys) / len(ys)
        ax.text(center_x, center_y, f"Tile {tidx}",
               color=color, fontsize=11, fontweight="bold",
               ha='center', va='center',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 设置坐标轴
    all_x = [h.cx for h in all_hexes]
    all_y = [h.cy for h in all_hexes]
    if all_x and all_y:
        margin = 2.0 * all_hexes[0].size
        ax.set_xlim(min(all_x) - margin, max(all_x) + margin)
        ax.set_ylim(min(all_y) - margin, max(all_y) + margin)
    
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(f"切割方案 k={k} (红线=切割边)", fontsize=14, pad=20)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=250, bbox_inches='tight')
    plt.close(fig)

# -----------------------
# 主入口
# -----------------------
if __name__ == "__main__":
    # ========== 用户输入区域 ==========
    pattern_str = """
1
1
1
"""
    total_repeats = 8      # 水平方向重复次数
    max_expand = 6         # 尝试的最大tile_repeat
    out_count = 3          # 最多输出多少种切割方法
    size = 0.8             # 六边形边长（绘图用）
    time_limit_per_tile = 3.0  # 每个tile上寻找路径的超时（秒）
    require_equal_tile_size = True
    require_bridge_between_tiles = True
    
    # ========== 运行 ==========
    methods, selected = try_cutting_with_edge_tracking(
        pattern_str=pattern_str,
        total_repeats=total_repeats,
        max_expand=max_expand,
        out_count=out_count,
        size=size,
        time_limit_per_tile=time_limit_per_tile,
        require_equal_tile_size=require_equal_tile_size,
        require_bridge_between_tiles=require_bridge_between_tiles
    )
    
    print("\n" + "="*60)
    print("找到的候选k值:", [m["k"] for m in methods])
    print("选择输出的k值:", [m["k"] for m in selected])
    print("输出已保存在 ./cuts_out/ 目录下（JSON + PNG）")
    print("="*60)
