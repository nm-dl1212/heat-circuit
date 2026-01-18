import numpy as np
from itertools import product
from .models import Node, BoundaryType
from .graph import Graph


def build_graph(
    nx: int,
    ny: int,
    dx: float,
    dy: float,
    k: float,
    thickness: float,
    Q: np.ndarray,
    BC: np.ndarray,
) -> Graph:
    """
    グラフを構築する。

    * param nx: x方向のノード数
    * param ny: y方向のノード数
    * param dx: x方向の間隔 [m]
    * param dy: y方向の間隔 [m]
    * param k: 熱伝導率 [W/mK]
    * param thickness: 板厚 [m]
    * param Q: 熱源分布 [W]
    * param BC: 境界条件 [K]
    * return: 構築されたグラフ
    """
    G = Graph()
    area = thickness * dy  # 断面積 [m^2]

    # まず全ノードを追加
    for i, j in product(range(nx), range(ny)):
        # 境界条件を判定
        if BC[i, j] != 0.0:
            boundary_type = BoundaryType.DIRICHLET
            boundary_value = BC[i, j]
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
        u = i * ny + j  # 始点ノードID

        # 横方向にエッジを追加
        if i + 1 < nx:
            v = (i + 1) * ny + j  # 終点ノードID
            resistance = float(dx / (k * area))
            G.add_edge(u, v, resistance)

        # 縦方向にエッジを追加
        if j + 1 < ny:
            v = i * ny + (j + 1)  # 終点ノードID
            resistance = float(dy / (k * area))
            G.add_edge(u, v, resistance)

    return G


def solve_thermal_network(G: Graph, nx: int, ny: int) -> np.ndarray:
    """
    熱ネットワークの連立方程式を解く。

    * param G: グラフ
    * param nx: x方向のノード数
    * param ny: y方向のノード数
    * return: 温度分布 [K]
    """
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

    return T
