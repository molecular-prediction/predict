import time
from typing import List, Dict, Tuple
from gnr_types import BenzeneHex

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
