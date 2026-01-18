import numpy as np
from heat_circuit import build_graph, solve_thermal_network, plot_and_save


# -----------------------
# パラメータ
# -----------------------
nx, ny = 51, 51
dx, dy = 0.002, 0.002  # ノード間距離 [m]
k = 236.0  # 熱伝導率 [W/mK] (アルミニウム)
thickness = 0.002  # 板厚 [m]

# 自然空冷条件
T_air = 15.0 + 273.15  # 空気温度 [K] (15℃)
h_conv = 10.0  # 対流係数 [W/(m²K)] (自然空冷)

# 熱源 [W]
Q = np.zeros((nx, ny))
Q[20:30, 20:30] = 6.0 / 100
Q[5:7, 18:20] = 1.0 / 4
Q[8:14, 40:42] = 2.6 / 12


def main():
    # グラフの構築
    G = build_graph(nx, ny, dx, dy, k, thickness, Q, T_air, h_conv)

    # 熱ネットワークの解く
    T = solve_thermal_network(G, nx, ny)

    # 可視化
    plot_and_save(T)


if __name__ == "__main__":
    main()
    print("simulation finished, image saved: thermal_network_simulation.png")
