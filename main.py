import numpy as np
import matplotlib.pyplot as plt

# -----------------------
# パラメータ
# -----------------------
nx, ny = 51, 51
dx, dy = 0.001, 0.001  # ノード間距離 [m]
k = 236.0  # 熱伝導率 [W/mK] (...アルミニウム)
thickness = 0.001  # 板厚 [m]
area = thickness * dy  # 断面積 [m^2]
T_boundary = 300.0  # 境界温度 [K]

R = dx / (k * area)  # 熱抵抗 [K/W] = 距離 / (熱伝導率 * 断面積)

# 熱源 [W]
Q = np.zeros((nx, ny))

Q[20:30, 20:30] = 24.0 / 100  # 中央に大きな熱源(24Wを100分割)
Q[5:7, 18:20] = 12.0 / 4


# -----------------------
# 連立方程式を解く
# 方程式の本数: N = nx * ny
# -----------------------
N = nx * ny
A = np.zeros((N, N))
b = np.zeros(N)


def idx(i, j):
    return i * ny + j


for i in range(nx):
    for j in range(ny):
        p = idx(i, j)

        # 境界ノードであれば、固定温度のままで変動しない
        if i == 0 or i == nx - 1 or j == 0 or j == ny - 1:
            A[p, p] = 1.0
            b[p] = T_boundary
            continue

        # 内部ノード
        # 熱収支を維持するため、隣接ノードすべての熱抵抗の総和を引く
        A[p, p] = -(1 / R * 4)

        # 隣接するノードに、熱抵抗の逆数(熱コンダクタンス)をセットする
        for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            q = idx(i + di, j + dj)
            A[p, q] = 1.0 / R

        b[p] = -Q[i, j]
print(A)

# -----------------------
# 連立方程式を解く
# -----------------------
T = np.linalg.solve(A, b)
T = T.reshape((nx, ny))


# -----------------------
# 可視化
# -----------------------
plt.figure(figsize=(6, 5))
plt.imshow(T, origin="lower", cmap="coolwarm")
plt.colorbar(label="Temperature [K]")
plt.title("Thermal Network Simulation (10x10 Plate)")
plt.xlabel("x")
plt.ylabel("y")
plt.show()
plt.savefig("thermal_network_simulation.png")
