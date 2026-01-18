from typing import Dict, List
from .models import Node, Edge


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

    def add_edge(self, u: int, v: int, resistance: float = None):
        """
        ノードuとvの間にエッジを追加。

        * param u: ノードID
        * param v: ノードID
        * param resistance: 熱抵抗 [K/W]
        """
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
