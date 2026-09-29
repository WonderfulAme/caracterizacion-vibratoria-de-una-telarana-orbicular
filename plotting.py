import networkBuild as nb
import matplotlib.pyplot as plt
import numpy as np

'''Función para trazar la red de araña.'''
def plotNetwork(self):

    fig, ax = plt.subplots()

    # Dibujar hilos
    for i, j, tipo in self.threads:

        x = [self.X[i, 0], self.X[j, 0]]
        y = [self.X[i, 1], self.X[j, 1]]

        ax.plot(x, y, "k-", linewidth=0.5)

    # Nodos fijos del marco
    fixed_nodes = np.where(self.fixed)[0]

    ax.plot(
        self.X[fixed_nodes, 0],
        self.X[fixed_nodes, 1],
        "k^",
        markersize=2,
        label="Marco fijo"
    )

    ax.set_aspect("equal")

    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("Spiderweb Network")

    ax.legend()

    plt.savefig("informe/fig01_geometria.png", dpi=300, bbox_inches="tight")