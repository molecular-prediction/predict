from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

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


@dataclass
class CutExtension:
    tile_index: int
    edge: str
    direction: str
    start: Tuple[float, float]
    end: Tuple[float, float]


@dataclass
class GlobalCutPlan:
    cutting_edges: Dict[int, List[Tuple[int, int]]] = field(default_factory=dict)
    endpoint_extensions: List[CutExtension] = field(default_factory=list)
    top_exit_direction: Optional[str] = None
    bottom_exit_direction: Optional[str] = None
    is_complete: bool = False
    invalid_reason: str = ""
    expected_tile_count: int = 0
    complete_tile_count: int = 0
