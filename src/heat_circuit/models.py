from dataclasses import dataclass
from typing import Tuple
from enum import Enum


# -----------------------
# 境界条件の種類
# -----------------------
class BoundaryType(Enum):
    INTERIOR = 0  # 内部ノード
    DIRICHLET = 1  # 温度固定


# -----------------------
# ノード／エッジの定義 (networkx風)
# -----------------------
@dataclass
class Node:
    id: int
    pos: Tuple[float, float, float]  # 位置情報 (x, y, z) [m]
    temperature: float = None  # 温度 [K]
    heat: float = 0.0  # 発熱量 [W]
    boundary_type: BoundaryType = BoundaryType.INTERIOR  # 境界条件の種類
    boundary_value: float = None  # 境界値（温度 or 熱流束） [K or W]


@dataclass
class Edge:
    u: int  # 始点ノードID
    v: int  # 終点ノードID
    resistance: float  # 熱抵抗 [K/W]

    @property
    def conductance(self) -> float:
        if self.resistance == 0:
            return 0.0
        else:
            return 1.0 / self.resistance
