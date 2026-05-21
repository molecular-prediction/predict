from dataclasses import dataclass
from typing import List, Tuple

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
