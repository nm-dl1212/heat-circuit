import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Dict, List, Tuple
from itertools import product
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


class Graph:
    def __init__(self):
        self.nodes: Dict[int, Node] = {}
        self.edges: List[Edge] = []

    def add_node(self, node: Node):
        """
        ノードを追加。

        * param node: 追加ノード
        """
        self.nodes[node.id] = node

    def add_edge(
        self, u: int, v: int, distance: float = None, resistance: float = None
    ):
        """
        ノードuとvの間にエッジを追加。

        * param u: ノードID
        * param v: ノードID
        * param distance: ノード間距離 [m] (Noneの場合はノードの位置情報から計算)
        * param resistance: 熱抵抗 [K/W] (Noneの場合は距離と熱伝導率、断面積から計算)
        """
        # ノード間の距離を計算
        if distance is None:
            p1 = self.nodes[u].pos
            p2 = self.nodes[v].pos
            distance = float(np.linalg.norm(np.subtract(p1, p2)))

        # 熱抵抗を計算
        if resistance is None:
            resistance = float(distance / (k * area))

        # エッジを追加
        e = Edge(u, v, resistance)
        self.edges.append(e)

    def neighbors(self, u: int) -> List[Edge]:
        """
        ノードuに接続するエッジのリストを返す。

        * param u: ノードID
        * return: 接続エッジのリスト
        """
        res: List[Edge] = []
        for e in self.edges:
            if e.u == u:
                res.append(e)
            elif e.v == u:
                # return a reversed view so caller sees e.u==u and e.v==neighbor
                res.append(Edge(u, e.u, e.resistance))
        return res


# -----------------------
# パラメータ
# -----------------------
nx, ny = 51, 51
dx, dy = 0.001, 0.001  # ノード間距離 [m]
k = 236.0  # 熱伝導率 [W/mK] (アルミニウム)
thickness = 0.001  # 板厚 [m]
area = thickness * dy  # 断面積 [m^2]
T_boundary = 300.0  # 境界温度 [K]

# 熱源 [W]
Q = np.zeros((nx, ny))
Q[20:30, 20:30] = 24.0 / 100
Q[5:7, 18:20] = 12.0 / 4
Q[12:14, 40:42] = 12.0 / 4


# -----------------------
# グラフの構築
# -----------------------
G = Graph()


# まず全ノードを追加
for i, j in product(range(nx), range(ny)):
    # 境界条件を判定
    if i == 0 or i == nx - 1 or j == 0 or j == ny - 1:
        boundary_type = BoundaryType.DIRICHLET
        boundary_value = T_boundary
    else:
        boundary_type = BoundaryType.INTERIOR
        boundary_value = None

    node = Node(
        id=i * ny + j,
        pos=(i * dx, j * dy, 0.0),
        temperature=None,
        heat=float(Q[i, j]),
        boundary_type=boundary_type,
        boundary_value=boundary_value,
    )
    G.add_node(node)


# 次に全エッジを追加
for i, j in product(range(nx), range(ny)):
    u = i * ny + j
    if i + 1 < nx:
        v = (i + 1) * ny + j
        G.add_edge(u, v)
    if j + 1 < ny:
        v = i * ny + (j + 1)
        G.add_edge(u, v)


# -----------------------
# 連立方程式を組み立てる
# -----------------------
N = nx * ny
A = np.zeros((N, N))
b = np.zeros(N)

for node_id, node in G.nodes.items():
    # 境界ノード：温度固定条件
    if node.boundary_type == BoundaryType.DIRICHLET:
        A[node_id, node_id] = 1.0
        b[node_id] = node.boundary_value
        continue

    # 内部ノード: 隣接エッジのコンダクタンスを集める
    neigh = G.neighbors(node_id)
    g_sum = 0.0
    for e in neigh:
        g = e.conductance
        g_sum += g
        A[node_id, e.v] = g

    A[node_id, node_id] = -g_sum
    b[node_id] = -node.heat


# 連立方程式を解く
T = np.linalg.solve(A, b)
T = T.reshape((nx, ny))

# ノードに温度を書き戻す
for i, j in product(range(nx), range(ny)):
    G.nodes[i * ny + j].temperature = float(T[i, j])


# -----------------------
# 可視化（保存のみ、表示は行わない）
# -----------------------
plt.figure(figsize=(6, 5))
plt.imshow(T, origin="lower", cmap="coolwarm")
plt.colorbar(label="Temperature [K]")
plt.title("Thermal Network Simulation (Plate)")
plt.xlabel("x")
plt.ylabel("y")
plt.savefig("thermal_network_simulation.png", dpi=200)

if __name__ == "__main__":
    print("simulation finished, image saved: thermal_network_simulation.png")
