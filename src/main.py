import numpy as np
from heat_circuit import build_graph, solve_thermal_network, plot_and_save


# -----------------------
# パラメータ
# -----------------------
nx, ny = 51, 51
dx, dy = 0.001, 0.001  # ノード間距離 [m]
k = 236.0  # 熱伝導率 [W/mK] (アルミニウム)
thickness = 0.001  # 板厚 [m]

# 熱源 [W]
Q = np.zeros((nx, ny))
Q[20:30, 20:30] = 24.0 / 100
Q[5:7, 18:20] = 12.0 / 4
Q[12:14, 40:42] = 12.0 / 4

# 境界条件
BC = np.zeros((nx, ny))
BC[[0, -1], :] = 300.0
BC[:, [0, -1]] = 300.0


def main():
    # グラフの構築
    G = build_graph(nx, ny, dx, dy, k, thickness, Q, BC)

    # 熱ネットワークの解く
    T = solve_thermal_network(G, nx, ny)

    # 可視化
    plot_and_save(T)


if __name__ == "__main__":
    main()
    print("simulation finished, image saved: thermal_network_simulation.png")
