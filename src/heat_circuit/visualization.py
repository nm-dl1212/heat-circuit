import numpy as np
import matplotlib.pyplot as plt


def plot_and_save(T: np.ndarray, output_file: str = "thermal_network_simulation.png"):
    """
    温度分布を可視化して保存する。

    * param T: 温度分布 [K]
    * param output_file: 出力ファイル名
    """
    plt.figure(figsize=(6, 5))
    plt.imshow(T, origin="lower", cmap="coolwarm")
    plt.colorbar(label="Temperature [K]")
    plt.title("Thermal Network Simulation (Plate)")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.savefig(output_file, dpi=200)
    plt.close()
