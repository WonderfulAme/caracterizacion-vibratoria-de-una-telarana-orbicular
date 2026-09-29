'''
Este archivo construye la clase Network.
Contiene los parámetros básicos de la red.
'''
import numpy as np

# Factores de conversión
uN = 1e-6   # microNewton
mg = 1e-6   # miligramo
um = 1e-6   # micrómetro
cm = 1e-2   # centímetro


class Network:
    '''Clase que representa la red de araña'''
    def __init__(
        self,

        # Geometría
        N = 28,                # Número de radiales
        R_in=2.0*cm,           # Radio interno [m]
        Ds=0.4*cm,             # Separación entre anillos [m]
        R_anc=20.0*cm,         # Radio del marco [m]

        # Material
        rho=1270.0,            # Densidad [kg/m^3]
        d_R=3.93*um,           # Diámetro radial [m]
        E_R=10e9,              # Módulo de Young radial [Pa]
        d_S=2.4*um,            # Diámetro espiral [m]
        E_S=3e6,               # Módulo de Young espiral [Pa]

        # Pretensión
        T_R_medio=102.5*uN,    # Pretensión radial media [N]
        T_S=9.7*uN,             # Pretensión espiral [N]

        # Masa de la Araña
        m_spider = 50.0*mg,       # Masa de la araña en [kg]
    ):
        # Asignación
        self.N = N
        self.R_in = R_in
        self.Ds = Ds
        self.R_anc = R_anc
        self.rho = rho
        self.d_R = d_R
        self.E_R = E_R
        self.d_S = d_S
        self.E_S = E_S
        self.T_R_medio = T_R_medio
        self.T_S = T_S

        # Cálculo del número de anillos
        self.M = int(round((self.R_anc - self.Ds - self.R_in) / self.Ds)) + 1

        # Cálculo del radio externo en [cm]
        self.R_out = self.R_in + (self.M - 1)*self.Ds

    '''Muestra información sobre la red de araña'''
    def info(self):
        print("------------------------------------------------------")

        print("Spiderweb Information")
        print("\nGEOMETRY")
        print(f"Radial threads N      : {self.N}")
        print(f"Spiral rings M        : {self.M}")
        print(f"Inner radius          : {self.R_in/cm:.2f} cm")
        print(f"Outer spiral radius   : {self.R_out/cm:.2f} cm")
        print(f"Frame radius          : {self.R_anc/cm:.2f} cm")
        print(f"Spiral spacing        : {self.Ds/cm:.2f} cm")

        print("\nMATERIAL")
        print(f"Silk density          : {self.rho:.1f} kg/m^3")
        print(f"Radial diameter       : {self.d_R/um:.2f} um")
        print(f"Spiral diameter       : {self.d_S/um:.2f} um")
        print(f"Radial Young modulus  : {self.E_R/1e9:.3f} GPa")
        print(f"Spiral Young modulus  : {self.E_S/1e6:.3f} MPa")

        print("\nPRETENSION")
        print(f"Mean radial tension   : {self.T_R_medio/uN:.2f} uN")
        print(f"Spiral tension        : {self.T_S/uN:.2f} uN")

        print("------------------------------------------------------")


    def pretensions(self):

        # Pretensión de cada anillo
        T_S = np.full(self.M, self.T_S)

        # Incremento acumulado de tensión radial
        inc = np.concatenate([
            [0.0],
            np.cumsum(
                2*T_S*np.sin(np.pi/self.N)
            )
        ])

        # Pretensiones radiales
        T_R = np.zeros(self.M + 2)

        T_R[1:] = (
            self.T_R_medio
            - inc.mean()
            + inc
        )

        # Verificación física
        assert T_R[1] > 0, \
            "La pretensión radial interna resultó negativa"

        return T_R, T_S


    '''Construye los nodos de la red de araña'''
    def networkNodes(self):

        theta = 2*np.pi*np.arange(self.N)/self.N        # Ángulos de los radiales
        r = np.linspace(self.R_in, self.R_out, self.M)  # Radios de los anillos

        # Nodo central
        X = [[0.0, 0.0, 0.0]]
        id_nodo = {"hub": 0}
        fixed = [False]

        # Crear nodos de los anillos
        for m in range(1, self.M + 1):
            for n in range(self.N):

                # Cada nodo tiene un identificador único (n, m) de la forma (número de radial, número de anillo)
                id_nodo[(n, m)] = len(X)

                X.append([
                    r[m-1]*np.cos(theta[n]),
                    r[m-1]*np.sin(theta[n]),
                    0.0
                ])

                fixed.append(False)

        # crear nodos del marco
        for n in range(self.N):
            id_nodo[(n, "A")] = len(X)

            X.append([
                self.R_anc*np.cos(theta[n]),
                self.R_anc*np.sin(theta[n]),
                0.0
            ])

            fixed.append(True)

        self.X = np.array(X)
        self.id_nodo = id_nodo
        self.r = r
        self.theta = theta
        self.fixed = np.array(fixed)

    def networkThreads(self):
        threads = []

        # Hilos radiales
        for n in range(self.N):
            # Primeros hilos radiales desde el nodo central hasta el primer anillo
            threads.append((self.id_nodo["hub"], self.id_nodo[(n, 1)], "R"))

            # Entre anillos
            for m in range(2, self.M + 1):
                threads.append(
                    (
                        self.id_nodo[(n, m-1)],
                        self.id_nodo[(n, m)],
                        "R"
                    )
                )

            # Ultimos hilos radiales desde el último anillo hasta el marco
            threads.append((self.id_nodo[(n, self.M)], self.id_nodo[(n, "A")], "R"))

        # -----------------------------
        # Hilos espirales
        # -----------------------------
        for m in range(1, self.M + 1):
            for n in range(self.N):
                threads.append(
                    (
                        self.id_nodo[((n-1) % self.N, m)],
                        self.id_nodo[(n, m)],
                        "S"
                    )
                )

        self.threads = threads


    def threadProperties(self):
        # Area de la sección transversal de los hilos
        A_R = np.pi * self.d_R**2 / 4  # Radial 
        A_S = np.pi * self.d_S**2 / 4  # Espiral

        threadsData = []

        for i, j, tipo in self.threads:
            # Calcular la longitud y el vector unitario del hilo
            L = np.linalg.norm(self.X[i] - self.X[j])
            e = (self.X[i] - self.X[j]) / L

            if tipo == "R":
                EA = self.E_R * A_R

            else:
                EA = self.E_S * A_S

            threadsData.append({
                "i": i,
                "j": j,
                "L": L,
                "e": e,
                "EA": EA,
                "tipo": tipo
            })

        self.threadsData = threadsData
