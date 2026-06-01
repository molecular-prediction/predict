import time
from typing import List, Dict, Tuple
from gnr_types import BenzeneHex

class EdgeCuttingPathFinder:
    def __init__(self, hexes_subset: List[BenzeneHex], full_adj: Dict[int, List[int]], k_cols: int):
        self.hexes = hexes_subset
        self.full_adj = full_adj
        self.k_cols = k_cols
        self.id_to_hex = {h.id: h for h in hexes_subset}
        self.found_paths = []
        self.start_time = 0
        self.time_limit = 3.0
        self.max_paths = 20
        self.MAX_HORIZ_RUN = 2
        
        # 1. 计算上下边缘
        all_rows = [h.row for h in self.hexes]
        if all_rows:
            self.top_row = min(all_rows)
            self.bottom_row = max(all_rows)
        else:
            self.top_row = 0
            self.bottom_row = 0

        # 2. 建立坐标映射表
        self.coord_map = {}
        for h in self.hexes:
            self.coord_map[(h.row, h.relative_col)] = h.id

        # 3. 预计算每一行的边缘范围
        self.row_bounds = {}
        if self.hexes:
            for h in self.hexes:
                if h.row not in self.row_bounds:
                    self.row_bounds[h.row] = {'min': h.relative_col, 'max': h.relative_col}
                else:
                    self.row_bounds[h.row]['min'] = min(self.row_bounds[h.row]['min'], h.relative_col)
                    self.row_bounds[h.row]['max'] = max(self.row_bounds[h.row]['max'], h.relative_col)

        # 4. [关键修改] 动态学习合法的移动偏移量 (Offset Learning)
        # 蜂窝网格中，奇数行和偶数行的对齐方式不同，导致邻居的相对坐标 (dr, dc) 不同
        # 我们遍历现有图块内部的连接，记录下所有“合法的物理移动”
        self.allowed_offsets_by_parity = {0: set(), 1: set()} # 0为偶数行，1为奇数行
        
        for h in self.hexes:
            parity = h.row % 2
            # 遍历该节点的物理邻居
            for nid in self.full_adj.get(h.id, []):
                if nid not in self.id_to_hex: continue
                n = self.id_to_hex[nid]
                
                dr = n.row - h.row
                dc = n.relative_col - h.relative_col
                
                # 只记录向下的移动 或者 同行的移动 (用于后续生成逻辑邻居)
                # 即使 full_adj 是双向的，我们只关心 dr >= 0 的模式用于 DFS
                if dr >= 0:
                    self.allowed_offsets_by_parity[parity].add((dr, dc))
        
        # 如果图块太小(K=1/2)导致内部连接样本不足，手动补充标准蜂窝偏移量作为保底
        # 这是一个启发式的补救措施
        if not self.allowed_offsets_by_parity[0] or not self.allowed_offsets_by_parity[1]:
            # 标准偏移假设
            standard_offsets = {(0, -1), (0, 1), (1, 0), (1, -1), (1, 1)}
            if not self.allowed_offsets_by_parity[0]: self.allowed_offsets_by_parity[0] = standard_offsets
            if not self.allowed_offsets_by_parity[1]: self.allowed_offsets_by_parity[1] = standard_offsets

    def _is_edge_hex(self, h: BenzeneHex) -> bool:
        if h.row not in self.row_bounds: return False
        bounds = self.row_bounds[h.row]
        return (h.relative_col == bounds['min']) or (h.relative_col == bounds['max'])

    def find_all_paths(self):
        self.start_time = time.time()
        self.found_paths = []
        top_hexes = [h for h in self.hexes if h.row == self.top_row]
        for start_hex in sorted(top_hexes, key=lambda h: len(self.full_adj.get(h.id, []))):
            if self._should_stop(): break
            self._dfs(start_hex.id, {start_hex.id}, [], {start_hex.row}, 0)
        return self.found_paths

    def _dfs(self, curr_id, visited, path, covered_rows, current_horiz_run):
        if self._should_stop(): return
        curr_h = self.id_to_hex[curr_id]
        
        if curr_h.row == self.bottom_row:
            self.found_paths.append(path.copy())
            return

        # --- 获取邻居集合 ---
        
        # 1. 物理邻居
        physical_neighbor_ids = set(self.full_adj.get(curr_id, []))
        
        # 2. [修正] 基于学习到的偏移量生成逻辑邻居
        logical_neighbors_ids = set()
        
        parity = curr_h.row % 2
        valid_offsets = self.allowed_offsets_by_parity[parity]
        
        for r_off, c_off in valid_offsets:
            target_r = curr_h.row + r_off
            target_c = curr_h.relative_col + c_off
            
            # 周期性取模
            wrapped_c = target_c % self.k_cols
            
            if (target_r, wrapped_c) in self.coord_map:
                nei_id = self.coord_map[(target_r, wrapped_c)]
                is_periodic_wrap = target_c != wrapped_c
                if nei_id != curr_id and is_periodic_wrap:
                    logical_neighbors_ids.add(nei_id)

        all_potential_neighbors = physical_neighbor_ids.union(logical_neighbors_ids)
        valid_neighbors = [nid for nid in all_potential_neighbors if nid in self.id_to_hex]
        
        # 排序
        neighbors_sorted = sorted(valid_neighbors, key=lambda n: (
            1 if self.id_to_hex[n].row == curr_h.row else 0,
            self.id_to_hex[n].row,
        ))

        for nei_id in neighbors_sorted:
            if self._should_stop(): return
            if nei_id in visited: continue
            nei_h = self.id_to_hex[nei_id]
            
            if nei_h.row < curr_h.row: continue

            is_horiz = (nei_h.row == curr_h.row)

            if is_horiz:
                if curr_h.row == self.top_row: continue
                if curr_h.row == self.bottom_row: continue
                if self._is_edge_hex(curr_h) or self._is_edge_hex(nei_h): continue
                
                if current_horiz_run >= self.MAX_HORIZ_RUN: continue
                new_horiz_run = current_horiz_run + 1
            else:
                new_horiz_run = 0

            visited.add(nei_id)
            path.append((curr_id, nei_id))
            self._dfs(nei_id, visited, path, covered_rows | {nei_h.row}, new_horiz_run)
            path.pop()
            visited.remove(nei_id)

    def _should_stop(self):
        return (time.time() - self.start_time > self.time_limit) or (len(self.found_paths) >= self.max_paths)
