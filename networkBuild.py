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
        R_in=0.4*cm,           # Radio interno [m]
        Ds=0.4*cm,             # Separación entre anillos [m]
        R_anc=20.0*cm,         # Radio del marco [m]

        # Material
        rho=1270.0,            # Densidad [kg/m^3]
        d_R=3.93*um,           # Diámetro radial [m]
        E_R=10e9,              # Módulo de Young radial [Pa]
        d_S=2.4*um,            # Diámetro espiral [m]
        E_S=3e6,               # Módulo de Young espiral [Pa]

        # Pretensión
        T_R_avg=102.5*uN,    # Pretensión radial media [N]
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
        self.T_R_avg = T_R_avg
        self.T_S = T_S
        self.m_spider = m_spider

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
        print(f"Mean radial tension   : {self.T_R_avg/uN:.2f} uN")
        print(f"Spiral tension        : {self.T_S/uN:.2f} uN")

        print("------------------------------------------------------")


    def pretensions(self):

        # Pretensión de cada hilo espiral
        T_S = np.full(self.M, self.T_S)

        # Incremento acumulado de tensión radial
        inc = np.concatenate([[0.0], np.cumsum(2*T_S*np.sin(np.pi/self.N))])

        # Pretensiones radiales
        T_R = np.zeros(self.M + 2)

        T_R[1:] = self.T_R_avg - inc.mean() + inc

        # Verificación física
        assert T_R[1] > 0, \
            "La pretensión radial interna resultó negativa"

        # Guardar en el objeto
        self.T_R = T_R
        self.T_S_array = T_S

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

                X.append([r[m-1]*np.cos(theta[n]), r[m-1]*np.sin(theta[n]), 0.0])

                fixed.append(False)

        # crear nodos del marco
        for n in range(self.N):
            id_nodo[(n, "A")] = len(X)

            X.append([self.R_anc*np.cos(theta[n]), self.R_anc*np.sin(theta[n]), 0.0])

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
            # Hub -> primer anillo
            threads.append((self.id_nodo["hub"], self.id_nodo[(n, 1)], "R", 1))

            # Entre anillos consecutivos
            for m in range(2, self.M + 1):
                threads.append((self.id_nodo[(n, m-1)], self.id_nodo[(n, m)], "R", m))

            # Último anillo -> marco
            threads.append((self.id_nodo[(n, self.M)], self.id_nodo[(n, "A")], "R", self.M + 1))

        # Hilos espirales
        for m in range(1, self.M + 1):
            for n in range(self.N):
                threads.append((self.id_nodo[((n-1) % self.N, m)], self.id_nodo[(n, m)], "S", m))

        self.threads = threads
    
    def threadProperties(self):
        # Área de la sección transversal
        A_R = np.pi * self.d_R**2 / 4   # Hilo radial
        A_S = np.pi * self.d_S**2 / 4   # Hilo espiral

        # Pretensiones
        T_R, T_S = self.pretensions()

        threadsData = []

        for i, j, tipo, m in self.threads:

            # Longitud del hilo
            L = np.linalg.norm(self.X[i] - self.X[j])

            # Vector unitario del hilo
            e = (self.X[i] - self.X[j]) / L

            # Propiedades según el tipo de hilo
            if tipo == "R":
                EA = self.E_R * A_R
                T0 = T_R[m]

            else:
                EA = self.E_S * A_S
                T0 = T_S[m - 1]

            # Rigidez axial del segmento
            kappa = EA / L

            threadsData.append({
                "i": i,
                "j": j,
                "L": L,
                "e": e,
                "A": A_R if tipo == "R" else A_S,
                "EA": EA,
                "kappa": kappa,
                "T0": T0,
                "tipo": tipo,
                "m": m
            })

        self.threadsData = threadsData

    '''Para cada hilo, calcular la masa calcula las masas nodales de la red de araña'''
    def nodalMasses(self):
        mu = np.zeros(len(self.X))
        for h in self.threadsData:
            A = h["A"]              # Área de la sección transversal
            lam = self.rho * A      # Masa por unidad de longitud
            mh = lam * h["L"]       # Masa total del hilo

            # Repartir la mitad en cada nodo
            mu[h["i"]] += mh / 2
            mu[h["j"]] += mh / 2

        # Masa de la araña en el nodo central
        mu[0] += self.m_spider * 0

        self.mu = mu

    def degreesOfFreedom(self):
        free = np.where(~self.fixed)[0] # Nodos libres
        dof = -np.ones((len(self.X), 3), dtype=int) # Inicialmente todos los grados de libertad son -1

        for k, i in enumerate(free):
            dof[i] = 3*k + np.arange(3) # asignar 3 grados de libertad cada uno de los nodos libres

        # Número total de grados de libertad
        ndof = 3 * len(free)

        self.free = free
        self.dof = dof
        self.ndof = ndof

    '''Matriz de rigidez para cada hilo'''
    def threadStiffnessMatrix(self, h):

        ke = (
            h["T0"] / h["L"] * np.eye(3)
            + (h["kappa"] - h["T0"] / h["L"])
            * np.outer(h["e"], h["e"])
        )

        return ke

    def globalStiffnessMatrix(self):

        K = np.zeros((self.ndof, self.ndof))

        for h in self.threadsData:

            # Matriz elemental 3x3
            ke = self.threadStiffnessMatrix(h)

            # Matriz elemental completa 6x6
            Ke = np.block([
                [ ke, -ke],
                [-ke,  ke]
            ])

            # Grados de libertad de los nodos i y j
            dof_i = self.dof[h["i"]]
            dof_j = self.dof[h["j"]]

            # GDL del elemento completo
            dof_h = np.concatenate([dof_i, dof_j])

            # Ensamblaje en la matriz global
            for a in range(6):
                for b in range(6):

                    if dof_h[a] != -1 and dof_h[b] != -1:
                        K[dof_h[a], dof_h[b]] += Ke[a, b]

        self.K = K

    def globalMassMatrix(self):

        Mdiag = np.repeat(
            self.mu[self.free],
            3
        )

        M = np.diag(Mdiag)

        self.Mdiag = Mdiag
        self.M = M

    def verification(self):

        # 1. Verificar que K sea simétrica
        symmetry_error = (np.linalg.norm(self.K - self.K.T)/ np.linalg.norm(self.K))

        # 2. Verificar equilibrio de fuerzas
        F = np.zeros_like(self.X)
        for h in self.threadsData:
            # Fuerza de pretensión sobre los extremos del hilo
            F[h["i"]] -= h["T0"] * h["e"]
            F[h["j"]] += h["T0"] * h["e"]

        # Solo revisar nodos libres
        residual = np.max(np.abs(F[self.free]))

        # Normalizar usando una tensión de referencia
        residual_rel = residual / self.T_R[1]

        # Verificaciones
        assert np.allclose(self.K, self.K.T), \
            f"La matriz de rigidez K no es simétrica. Error de Simetría: {symmetry_error:.2e}"
        
        assert residual_rel < 1e-6, \
            f"La red no está en equilibrio. Residual de Equilibrio: {residual_rel:.2e}"

        # 3. Verificar masas positivas
        assert np.all(np.diag(self.M) > 0), \
            f"La matriz de masa tiene elementos no positivos. Error: {np.diag(self.M)}"



            

        
