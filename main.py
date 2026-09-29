# =====================================================================
#  CICLO 2 — CARACTERIZACIÓN VIBRATORIA DE UNA TELARAÑA ORBICULAR
#  Especie de referencia: Araneus diadematus (araña de jardín europea)
# =====================================================================


# =====================================================================
# CELDA 1 — Importaciones y parámetros (con su fuente bibliográfica)
# =====================================================================
import os, time, shutil
import numpy as np
import scipy.linalg as sla
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 300, "font.size": 10,
                     "axes.grid": True, "grid.alpha": 0.3})
os.makedirs("figuras", exist_ok=True)

uN, mg, um, cm = 1e-6, 1e-6, 1e-6, 1e-2      # conversiones a SI

P_BASE = dict(
    # ---------------- geometría ----------------
    N=28,              # radiales: A. diadematus construye 25–30 (ángulo 12–15°)   [ADW]
    R_in=2.0*cm,       # radio del hub / primer anillo = 2 cm                        [Yavuz 2024]
    Ds=0.4*cm,         # separación entre vueltas de la espiral = 0.4 cm             [Yavuz 2024]
    R_anc=20.0*cm,     # radio de la red (marco, modelado rígido) = 20 cm            [Yavuz 2024]
    espaciado="lineal",
    anclajes=True,     # False reproduce la condición de frontera del anteproyecto
    # ---------------- araña en el hub ----------------
    hub_libre=True,    # hub = un solo nodo libre con la masa de la araña           [Yavuz 2024]
    m_arana=50*mg,     # 50 mg                                                       [Yavuz 2024]
    # ---------------- material ----------------
    rho=1270.0,        # densidad seda MA de Araneus = 1.27 g/cm^3                   [Stauffer 1994]
    d_R=3.93*um,       # diámetro radial                                             [Yavuz 2024, tabla 1]
    E_R=10e9,          # módulo inicial seda MA de A. diadematus = 10 GPa            [Gosline 1999]
    d_S=2.4*um,        # diámetro espiral de captura                                 [Yavuz 2024, tabla 1]
    E_S=3e6,           # módulo inicial seda viscosa de A. diadematus = 0.003 GPa    [Gosline 1999]
    factor_masa_S=1.0, # masa del pegamento: SIN DATO -> solo la fibra (supuesto)
    # ---------------- pretensiones ----------------
    T_R_medio=102.5*uN,  # pretensión radial media = 102.5 µN                        [Yavuz 2024, tabla 1]
    T_S=9.7*uN,          # pretensión de la espiral = 9.7 µN                          [Yavuz 2024, tabla 1]
    # ---------------- disipación ----------------
    zeta=0.05,         # estimado de la histéresis del 65 % de ambas sedas:
                       # eta = 0.65/(2*pi) ~ 0.10 -> zeta = eta/2 ~ 0.05              [Gosline 1999]
    # ---------------- excitación ----------------
    f_banda=(1.0, 1000.0),   # Hz: presas atrapadas 5–30 Hz [Landolfa 1996]; aleteo
                             # de mosca ~200 Hz usado como estímulo [Yavuz 2024]
    f_presa=(5.0, 30.0),     # Hz: banda de vibración de presas atrapadas            [Landolfa 1996]
    r_presa=8.0*cm,          # la presa se ubica a 8 cm del hub                     [Yavuz 2024]
)
P_BASE["M"] = int(round((P_BASE["R_anc"] - P_BASE["Ds"] - P_BASE["R_in"])/P_BASE["Ds"])) + 1
P_BASE["R_out"] = P_BASE["R_in"] + (P_BASE["M"] - 1)*P_BASE["Ds"]
print(f"N = {P_BASE['N']} radiales, M = {P_BASE['M']} anillos "
      f"({P_BASE['R_in']/cm:.1f}–{P_BASE['R_out']/cm:.1f} cm), marco a {P_BASE['R_anc']/cm:.0f} cm")


# =====================================================================
# CELDA 2 — Construcción de la red, pretensiones de equilibrio,
#            ensamblaje de M y K   (semana 7)
# =====================================================================
def pretensiones(P):
    """Equilibrio radial: T^R_{m+1} = T^R_m + 2 T^S_m sin(pi/N).
    Se elige T^R_1 para que la media de los radiales sea T_R_medio."""
    N, M = P["N"], P["M"]
    T_S = np.full(M, float(P["T_S"])) if np.isscalar(P["T_S"]) else np.asarray(P["T_S"], float)
    inc = np.concatenate([[0.0], np.cumsum(2*T_S*np.sin(np.pi/N))])   # m = 1..M+1
    T_R = np.zeros(M + 2)
    T_R[1:] = P["T_R_medio"] - inc.mean() + inc
    assert T_R[1] > 0, "La pretensión radial interna resultó negativa"
    return T_R, T_S

def construir_red(P):
    N, M = P["N"], P["M"]
    th = 2*np.pi*np.arange(N)/N
    r = (np.linspace if P["espaciado"] == "lineal" else np.geomspace)(P["R_in"], P["R_out"], M)

    X, fijo, id_nodo = [[0.0, 0.0, 0.0]], [not P["hub_libre"]], {"hub": 0}
    for m in range(1, M+1):
        for n in range(N):
            id_nodo[(n, m)] = len(X)
            X.append([r[m-1]*np.cos(th[n]), r[m-1]*np.sin(th[n]), 0.0]); fijo.append(False)
    if P["anclajes"]:
        for n in range(N):
            id_nodo[(n, "A")] = len(X)
            X.append([P["R_anc"]*np.cos(th[n]), P["R_anc"]*np.sin(th[n]), 0.0]); fijo.append(True)
    X, fijo = np.array(X), np.array(fijo)

    T_R, T_S = pretensiones(P)
    A_R, A_S = np.pi*P["d_R"]**2/4, np.pi*P["d_S"]**2/4
    lam = {"R": P["rho"]*A_R, "A": P["rho"]*A_R, "S": P["rho"]*A_S*P["factor_masa_S"]}
    EA = {"R": P["E_R"]*A_R, "A": P["E_R"]*A_R, "S": P["E_S"]*A_S}

    hilos = []
    def agregar(i, j, T0, tipo, m):
        L = np.linalg.norm(X[i]-X[j])
        hilos.append(dict(i=i, j=j, L=L, kap=EA[tipo]/L, T0=T0, tipo=tipo, m=m, e=(X[i]-X[j])/L))
    for n in range(N):
        agregar(id_nodo["hub"], id_nodo[(n, 1)], T_R[1], "R", 1)
        for m in range(2, M+1):
            agregar(id_nodo[(n, m-1)], id_nodo[(n, m)], T_R[m], "R", m)
        if P["anclajes"]:
            agregar(id_nodo[(n, M)], id_nodo[(n, "A")], T_R[M+1], "A", M+1)
    for m in range(1, M+1):
        for n in range(N):
            agregar(id_nodo[((n-1) % N, m)], id_nodo[(n, m)], T_S[m-1], "S", m)

    mu = np.zeros(len(X))                          # masas concentradas
    for h in hilos:
        mh = lam[h["tipo"]]*h["L"]; mu[h["i"]] += mh/2; mu[h["j"]] += mh/2
    mu[0] += P["m_arana"]

    libres = np.where(~fijo)[0]
    dof = -np.ones((len(X), 3), int)
    for k, i in enumerate(libres):
        dof[i] = 3*k + np.arange(3)
    ndof = 3*len(libres)

    K = np.zeros((ndof, ndof))
    for h in hilos:
        g = h["T0"]/h["L"]
        ke = g*np.eye(3) + (h["kap"] - g)*np.outer(h["e"], h["e"])
        di, dj = dof[h["i"]], dof[h["j"]]
        li, lj = di[0] >= 0, dj[0] >= 0
        if li: K[np.ix_(di, di)] += ke
        if lj: K[np.ix_(dj, dj)] += ke
        if li and lj:
            K[np.ix_(di, dj)] -= ke; K[np.ix_(dj, di)] -= ke
    Mdiag = np.repeat(mu[libres], 3)

    pos_libre = -np.ones(len(X), int); pos_libre[libres] = np.arange(len(libres))
    idx_anillo = np.array([[pos_libre[id_nodo[(n, m)]] for n in range(N)] for m in range(1, M+1)])
    return dict(P=P, N=N, M=M, r=r, th=th, X=X, fijo=fijo, id_nodo=id_nodo, hilos=hilos,
                mu=mu, libres=libres, dof=dof, ndof=ndof, K=K, Mdiag=Mdiag,
                T_R=T_R, T_S=T_S, idx_anillo=idx_anillo, pos_libre=pos_libre)

def fila_z(red, nodo):
    """Posición del nodo dentro del subsistema transversal (una fila por nodo libre)."""
    return red["pos_libre"][red["id_nodo"][nodo]]

def nodo_en_radio(red, radio, n=0):
    return (n, int(np.argmin(np.abs(red["r"] - radio))) + 1)

t0 = time.time()
red = construir_red(P_BASE)
print(f"Nodos libres: {len(red['libres'])}   gdl: {red['ndof']}   hilos: {len(red['hilos'])}"
      f"   ({time.time()-t0:.1f} s)")
print(f"T^R: interna = {red['T_R'][1]/uN:.1f} µN, media = {red['T_R'][1:].mean()/uN:.1f} µN, "
      f"junto al marco = {red['T_R'][-1]/uN:.1f} µN")


# =====================================================================
# CELDA 3 — Verificación del modelo  (semana 7)
# =====================================================================
def residuo_equilibrio(red):
    F = np.zeros_like(red["X"])
    for h in red["hilos"]:
        F[h["i"]] -= h["T0"]*h["e"]; F[h["j"]] += h["T0"]*h["e"]
    return np.abs(F[red["libres"]]).max()/red["T_R"][1]

def verificar(red):
    K = red["K"]; nd = red["ndof"]
    z = np.arange(2, nd, 3); xy = np.setdiff1d(np.arange(nd), z)
    Kz = K[np.ix_(z, z)]; Mz = red["Mdiag"][z]
    lam_min = sla.eigh(Kz, np.diag(Mz), eigvals_only=True, subset_by_index=[0, 0])[0]
    v = dict(simetria=np.linalg.norm(K-K.T)/np.linalg.norm(K),
             acople=np.abs(K[np.ix_(xy, z)]).max(),
             residuo=residuo_equilibrio(red), w2min=lam_min)
    print(f"  ||K-K^T||/||K||             = {v['simetria']:.1e}")
    print(f"  acople plano-transversal    = {v['acople']:.1e}  (debe ser 0)")
    print(f"  residuo de equilibrio/T^R_1 = {v['residuo']:.1e}  (debe ser ~0)")
    print(f"  omega_min^2 transversal     = {v['w2min']:.3e} s^-2 (>0)")
    return v

print("--- Verificación de la configuración base ---")
VERIF = verificar(red)
red_ante = construir_red(dict(P_BASE, anclajes=False))
VERIF["residuo_sin_anclaje"] = residuo_equilibrio(red_ante)
print(f"Sin anclajes (condición del anteproyecto): residuo = {VERIF['residuo_sin_anclaje']:.2f} T^R_1")
del red_ante

def benchmark_cuerda(Nb=6, Mb=8, D=0.4*cm):
    """Espiral apagada + hub fijo: cada radial es una cuerda con cuentas fija–fija.
    w_j = 2 sqrt(k/mu) sin(j pi/(2(M+1))), multiplicidad N."""
    Pb = dict(P_BASE, N=Nb, M=Mb, R_in=D, R_out=Mb*D, R_anc=(Mb+1)*D, hub_libre=False,
              m_arana=0.0, T_S=0.0, E_S=0.0, factor_masa_S=0.0, anclajes=True)
    rb = construir_red(Pb)
    w = np.sqrt(sla.eigh(rb["K"], np.diag(rb["Mdiag"]), eigvals_only=True))
    A = np.pi*Pb["d_R"]**2/4; mu = Pb["rho"]*A*D
    j = np.arange(1, Mb+1); S = np.sin(j*np.pi/(2*(Mb+1)))
    w_T = 2*np.sqrt(Pb["T_R_medio"]/D/mu)*S
    w_L = 2*np.sqrt(Pb["E_R"]*A/D/mu)*S
    w_ex = np.sort(np.concatenate([np.repeat(w_T, 2*Nb), np.repeat(w_L, Nb)]))
    err = (np.abs(np.sort(w) - w_ex)/w_ex).max()
    print(f"Benchmark cuerda con cuentas: error relativo máx = {err:.1e}")
    return j, w_T, w_L, np.sort(w), err

bench = benchmark_cuerda(); VERIF["benchmark"] = bench[-1]


# =====================================================================
# CELDA 4 — Valores propios y clasificación de modos  (semanas 8–9)
# =====================================================================
def analisis_modal(red):
    """Resuelve por separado las familias (desacople exacto). Devuelve los
    modos M-normalizados de cada familia en su propio subespacio."""
    K, Md, nd = red["K"], red["Mdiag"], red["ndof"]
    out = {}
    for fam, idx in [("T", np.arange(2, nd, 3)),
                     ("P", np.setdiff1d(np.arange(nd), np.arange(2, nd, 3)))]:
        w2, V = sla.eigh(K[np.ix_(idx, idx)], np.diag(Md[idx]))
        out[fam] = dict(idx=idx, w=np.sqrt(np.clip(w2, 0, None)), V=V)
    # número de onda azimutal p (DFT sobre el índice radial n)
    ia, th, N = red["idx_anillo"], red["th"], red["N"]
    uz = out["T"]["V"][ia]                                       # (M, N, nT)
    pot = (np.abs(np.fft.fft(uz, axis=1))**2).sum(0)
    p = np.argmax(pot, 0); out["T"]["p"] = np.minimum(p, N - p)
    VP = out["P"]["V"]; nl = len(red["libres"])
    U = np.zeros((nl, 2, VP.shape[1]))
    U[:, 0], U[:, 1] = VP[0::2], VP[1::2]
    ux, uy = U[ia, 0], U[ia, 1]
    ur = ux*np.cos(th)[None, :, None] + uy*np.sin(th)[None, :, None]
    ut = -ux*np.sin(th)[None, :, None] + uy*np.cos(th)[None, :, None]
    pot = (np.abs(np.fft.fft(ur, axis=1))**2 + np.abs(np.fft.fft(ut, axis=1))**2).sum(0)
    p = np.argmax(pot, 0); out["P"]["p"] = np.minimum(p, N - p)
    for f in out.values(): f["f"] = f["w"]/(2*np.pi)
    return out

t0 = time.time()
modal = analisis_modal(red)
print(f"Problema de valores propios resuelto en {time.time()-t0:.1f} s")
T_, P_ = modal["T"], modal["P"]
Mz = red["Mdiag"][T_["idx"]]; Kz = red["K"][np.ix_(T_["idx"], T_["idx"])]
VERIF["ortog"] = np.abs(T_["V"].T @ (Mz[:, None]*T_["V"]) - np.eye(len(Mz))).max()
VERIF["ortogK"] = np.abs(T_["V"].T @ Kz @ T_["V"] - np.diag(T_["w"]**2)).max()/T_["w"].max()**2
print(f"||Phi^T M Phi - I|| = {VERIF['ortog']:.1e},  ||Phi^T K Phi - W^2||/w_max^2 = {VERIF['ortogK']:.1e}")
for fam, nom in [("T", "transversales"), ("P", "en el plano")]:
    print(f"Modos {nom:13s}: {len(modal[fam]['f']):5d}   f = {modal[fam]['f'].min():10.2f} – "
          f"{modal[fam]['f'].max():10.1f} Hz")
fT = np.sort(T_["f"])
VERIF["pares"] = int(np.sum(np.isclose(fT[1:], fT[:-1], rtol=1e-7)))
print(f"Pares degenerados transversales: {VERIF['pares']} "
      f"(esperado por simetría C_N: {(red['N']//2 - 1)*red['M']})")


# =====================================================================
# CELDA 5 — Figuras de geometría y modos
# =====================================================================
def dibujar_red(ax, red, color="0.8", lw=0.4):
    X = red["X"]
    seg = [[X[h["i"], :2], X[h["j"], :2]] for h in red["hilos"]]
    ax.add_collection(LineCollection(seg, colors=color, linewidths=lw, zorder=1))
    ax.set_aspect("equal"); ax.autoscale(); ax.grid(False); ax.set_xticks([]); ax.set_yticks([])

def campo_z(red, v_libres):
    c = np.zeros(len(red["X"])); c[red["libres"]] = v_libres; return c

# Fig. 1 — geometría y pretensiones
fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.4))
X = red["X"]; seg = [[X[h["i"], :2]/cm, X[h["j"], :2]/cm] for h in red["hilos"]]
lc = LineCollection(seg, array=np.array([h["T0"]/uN for h in red["hilos"]]), cmap="viridis",
                    linewidths=0.6, norm=plt.matplotlib.colors.LogNorm())
ax[0].add_collection(lc); ax[0].set_aspect("equal"); ax[0].autoscale(); ax[0].grid(False)
ax[0].plot(0, 0, "ro", ms=5, label="hub con araña (50 mg)")
ax[0].plot(*(X[red["fijo"], :2].T/cm), "k^", ms=3, label="marco (fijo)")
ax[0].set_xlabel("x [cm]"); ax[0].set_ylabel("y [cm]"); ax[0].legend(loc="upper right", fontsize=7)
plt.colorbar(lc, ax=ax[0], label=r"$T^0$ [µN]")
ax[0].set_title(f"A. diadematus: N={red['N']}, M={red['M']}")
ax[1].plot(np.r_[red["r"], P_BASE["R_anc"]]/cm, red["T_R"][1:]/uN, "o-", ms=3, label=r"$T^R_m$ del modelo")
ax[1].axhline(102.5, color="C1", ls="--", label="media, modelo de Yavuz et al. (2024): 102.5 µN")
ax[1].axhline(132, color="C2", ls=":", label="junto al marco, Lin & Sobek (en Yavuz 2024): 132 µN")
ax[1].set_xlabel("r [cm]"); ax[1].set_ylabel("pretensión radial [µN]"); ax[1].legend(fontsize=7)
ax[1].set_title("Pretensión radial exigida por el equilibrio")
fig.tight_layout(); fig.savefig("figuras/fig01_geometria.png"); plt.show()

# Fig. 2 — espectro y clasificación
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
for fam, c, lab in [("T", "C0", "transversales"), ("P", "C3", "en el plano")]:
    d = modal[fam]
    ax[0].semilogy(np.arange(1, len(d["f"])+1), d["f"], ".", c=c, ms=2, label=lab)
    ax[1].semilogy(d["p"] + (0.15 if fam == "P" else -0.15), d["f"], "o", c=c, ms=2, alpha=0.5, label=lab)
fb = P_BASE["f_banda"]
for a in ax: a.axhspan(*fb, color="0.9", zorder=0)
ax[0].set_xlabel("índice de modo dentro de la familia"); ax[0].set_ylabel("f [Hz]"); ax[0].legend()
ax[0].set_title("Espectro (banda gris: 1–1000 Hz)")
ax[1].set_xlabel("número de onda azimutal $p$"); ax[1].set_ylabel("f [Hz]"); ax[1].legend()
ax[1].set_title("Clasificación por simetría cíclica")
fig.tight_layout(); fig.savefig("figuras/fig02_espectro.png"); plt.show()

# Fig. 3 — seis primeros modos transversales
fig, axs = plt.subplots(2, 3, figsize=(10, 6.8))
for ax, k in zip(axs.ravel(), range(6)):
    c = campo_z(red, T_["V"][:, k]); lim = np.abs(c).max()
    dibujar_red(ax, red)
    ax.scatter(*red["X"][:, :2].T, c=c, cmap="RdBu_r", vmin=-lim, vmax=lim, s=3, zorder=2)
    ax.set_title(f"modo {k+1}: f = {T_['f'][k]:.1f} Hz, p = {T_['p'][k]}", fontsize=9)
fig.suptitle(r"Modos transversales ($\phi_z$ en color)")
fig.tight_layout(); fig.savefig("figuras/fig03_modos_transversales.png"); plt.show()

# Fig. 4 — tres primeros modos en el plano
fig, axs = plt.subplots(1, 3, figsize=(11, 4))
for ax, k in zip(axs, range(3)):
    phx = np.zeros(len(X)); phy = np.zeros(len(X))
    phx[red["libres"]] = P_["V"][0::2, k]; phy[red["libres"]] = P_["V"][1::2, k]
    dibujar_red(ax, red)
    sel = np.arange(len(X))[::3]
    ax.quiver(X[sel, 0], X[sel, 1], phx[sel], phy[sel], color="C3", width=0.004, zorder=2)
    ax.set_title(f"modo {k+1}: f = {P_['f'][k]:.1f} Hz, p = {P_['p'][k]}", fontsize=9)
fig.suptitle(r"Modos en el plano (flechas: $(\phi_x,\phi_y)$)")
fig.tight_layout(); fig.savefig("figuras/fig04_modos_plano.png"); plt.show()

# Fig. 5 — benchmark
j, w_T, w_L, w_mod, _ = bench
fig, ax = plt.subplots(figsize=(5.5, 3.8))
ax.plot(j, w_T/(2*np.pi), "k-", label="exacta (transversal)")
ax.plot(j, w_L/(2*np.pi), "k--", label="exacta (longitudinal)")
ax.plot(j, np.unique(np.round(w_mod, 4))[:len(j)]/(2*np.pi), "o", c="C0", label="modelo (transversal)")
ax.plot(j, np.unique(np.round(w_mod, 4))[-len(j):]/(2*np.pi), "s", c="C3", mfc="none", label="modelo (longitudinal)")
ax.set_yscale("log"); ax.set_xlabel("orden $j$"); ax.set_ylabel("f [Hz]"); ax.legend(fontsize=8)
ax.set_title("Verificación: cuerda con cuentas")
fig.tight_layout(); fig.savefig("figuras/fig05_benchmark.png"); plt.show()


# =====================================================================
# CELDA 6 — Respuesta impulsional transversal (Duhamel, semana 10)
# =====================================================================
def respuesta_impulso(fam, fila_src, F0, t, zeta, filas_salida=None):
    """q(t) = F0 sum_k phi_k(src)/w_dk e^{-zeta w_k t} sin(w_dk t) phi_k  (subsistema de la familia)."""
    w, V = fam["w"], fam["V"]
    a = F0*V[fila_src]
    wd = w*np.sqrt(1 - zeta**2)
    Vs = V if filas_salida is None else V[filas_salida]
    q = np.empty((len(t), Vs.shape[0]))
    for s in range(0, len(t), 2000):
        tt = t[s:s+2000, None]
        q[s:s+2000] = ((a/wd)*np.exp(-zeta*w*tt)*np.sin(wd*tt)) @ Vs.T
    return q

N, M = red["N"], red["M"]
SRC = nodo_en_radio(red, P_BASE["r_presa"], n=0)            # presa a 8 cm
F0 = 5e-10   # impulso [N·s] escalado para un desplazamiento máximo ~0.2 mm en la fuente,
             # comparable al estímulo de 0.21 mm de Mortimer et al. (2018). El modelo es lineal.
OBS = {"fuente": SRC, "anillo 1, mismo radial": (0, 1),
       "anillo 1, a 90°": (N//4, 1), "anillo 1, radial opuesto": (N//2, 1)}
if P_BASE["hub_libre"]: OBS["hub (araña)"] = "hub"
fsrc = fila_z(red, SRC)
print(f"Fuente: nodo {SRC}, r = {red['r'][SRC[1]-1]/cm:.1f} cm")

t = np.linspace(0, 40e-3, 8001)                              # 40 ms
filas_obs = [fila_z(red, nd) for nd in OBS.values()]
q_obs = respuesta_impulso(T_, fsrc, F0, t, P_BASE["zeta"], filas_obs)
fig, ax = plt.subplots(figsize=(9, 4))
for (lab, nd), col in zip(OBS.items(), q_obs.T):
    ax.plot(t*1e3, col*1e6, lw=0.8, label=f"{lab} {nd}")
ax.set_xlabel("t [ms]"); ax.set_ylabel(r"$u_z$ [µm]"); ax.legend(fontsize=7)
ax.set_title(f"Respuesta a un impulso transversal de {F0*1e9:.1f} nN·s (ζ = {P_BASE['zeta']})")
fig.tight_layout(); fig.savefig("figuras/fig06_respuesta_impulsional.png"); plt.show()

# instantáneas
tiempos = np.array([0.3, 0.6, 1.0, 2.0])*1e-3
q_snap = respuesta_impulso(T_, fsrc, F0, tiempos, P_BASE["zeta"])
fig, axs = plt.subplots(1, 4, figsize=(13, 3.6))
for ax, tt, qs in zip(axs, tiempos, q_snap):
    c = campo_z(red, qs); lim = np.abs(c).max()
    dibujar_red(ax, red)
    ax.scatter(*red["X"][:, :2].T, c=c, cmap="RdBu_r", vmin=-lim, vmax=lim, s=3, zorder=2)
    ax.plot(*red["X"][red["id_nodo"][SRC], :2], "ko", ms=7, mfc="none", zorder=3)
    ax.set_title(f"t = {tt*1e3:.1f} ms", fontsize=9)
fig.suptitle(r"Propagación de la perturbación ($u_z$)")
fig.tight_layout(); fig.savefig("figuras/fig07_instantaneas.png"); plt.show()

# velocidad de la onda transversal a lo largo del radial de la fuente
t_f = np.linspace(0, 3e-3, 6001)
filas_rad = [fila_z(red, (SRC[0], m)) for m in range(1, M+1)]
q_rad = np.abs(respuesta_impulso(T_, fsrc, F0, t_f, 0.0, filas_rad))
lleg = np.array([t_f[np.argmax(c > 0.05*c.max())] for c in q_rad.T])
dist = np.abs(red["r"] - red["r"][SRC[1]-1]); sel = (dist > 0) & (red["r"] < red["r"][SRC[1]-1])
c_fit = np.polyfit(lleg[sel], dist[sel], 1)[0]
m_mid = SRC[1]//2
c_ref = np.sqrt(red["T_R"][m_mid]*P_BASE["Ds"]/red["mu"][red["id_nodo"][(0, m_mid)]])
print(f"Velocidad aparente de la onda transversal hacia el hub: {c_fit:.0f} m/s "
      f"(estimación sqrt(T/lambda_ef) = {c_ref:.0f} m/s)")


# =====================================================================
# CELDA 7 — Verificación: integración directa vs. superposición modal
# =====================================================================
Kz_s = Kz.copy()
v0 = np.zeros(len(Mz)); v0[fsrc] = F0/Mz[fsrc]
def rhs(tt, y):
    n = len(Mz); return np.concatenate([y[n:], -(Kz_s @ y[:n])/Mz])
t_ver = np.linspace(0, 5e-3, 801)
t0 = time.time()
sol = solve_ivp(rhs, (0, t_ver[-1]), np.concatenate([np.zeros(len(Mz)), v0]),
                t_eval=t_ver, method="DOP853", rtol=1e-10, atol=1e-20)
q_mod = respuesta_impulso(T_, fsrc, F0, t_ver, 0.0)
qd = sol.y[:len(Mz)].T; vd = sol.y[len(Mz):].T
VERIF["duhamel"] = np.abs(qd - q_mod).max()/np.abs(q_mod).max()
E = 0.5*np.sum(Mz*vd**2, 1) + 0.5*np.einsum("ti,ij,tj->t", qd, Kz_s, qd)
VERIF["energia"] = np.ptp(E)/E.mean()
print(f"Integración directa en {time.time()-t0:.1f} s. Error Duhamel vs directa = {VERIF['duhamel']:.1e};"
      f" variación de energía = {VERIF['energia']:.1e}")
fo = fila_z(red, (0, 1))
fig, ax = plt.subplots(figsize=(8, 3.5))
ax.plot(t_ver*1e3, qd[:, fo]*1e6, "k-", lw=2.2, alpha=0.35, label="integración directa (DOP853)")
ax.plot(t_ver*1e3, q_mod[:, fo]*1e6, "C0--", lw=1, label="superposición modal (Duhamel)")
ax.set_xlabel("t [ms]"); ax.set_ylabel(r"$u_z$ [µm]"); ax.legend(fontsize=8)
ax.set_title("Verificación temporal en el nodo (0,1), ζ = 0")
fig.tight_layout(); fig.savefig("figuras/fig08_verificacion_temporal.png"); plt.show()


# =====================================================================
# CELDA 8 — Función de transferencia e índice de transmisión (semana 11)
# =====================================================================
def malla_omega(w_modos, wa, wb, zeta, n_base=4000, n_pico=24):
    partes = [np.geomspace(wa, wb, n_base)]
    for wk in w_modos[(w_modos > wa) & (w_modos < wb)]:
        partes.append(np.linspace(wk*(1 - 5*zeta), wk*(1 + 5*zeta), n_pico))
    om = np.unique(np.concatenate(partes)); return om[(om >= wa) & (om <= wb)]

def transferencia(fam, fila_src, filas_obs, omega, zeta):
    w, V = fam["w"], fam["V"]
    a, B = V[fila_src], V[filas_obs]
    H = np.empty((len(filas_obs), len(omega)), complex)
    for s in range(0, len(omega), 1000):
        om = omega[s:s+1000]
        D = w[:, None]**2 - om[None, :]**2 + 2j*zeta*w[:, None]*om[None, :]
        H[:, s:s+1000] = B @ (a[:, None]/D)
    return H

def indice_transmision(fam, fila_src, omega, zeta, bloque=1000):
    """T_obs = int |H_obs,src|^2 dw para TODOS los observadores, sin guardar H completa."""
    w, V = fam["w"], fam["V"]; a = V[fila_src]
    T = np.zeros(V.shape[0]); prev = None
    for s in range(0, len(omega), bloque):
        om = omega[max(s-1, 0):s+bloque]
        D = w[:, None]**2 - om[None, :]**2 + 2j*zeta*w[:, None]*om[None, :]
        H2 = np.abs(V @ (a[:, None]/D))**2
        T += np.trapezoid(H2, om, axis=1)
    return T

zeta = P_BASE["zeta"]
wa, wb = 2*np.pi*np.array(P_BASE["f_banda"])
omega = malla_omega(T_["w"], wa, wb, zeta)
t0 = time.time()
Tidx = indice_transmision(T_, fsrc, omega, zeta)
omega2 = malla_omega(T_["w"], wa, wb, zeta, n_base=8000, n_pico=48)
Tidx2 = indice_transmision(T_, fsrc, omega2, zeta)
VERIF["malla"] = np.abs(Tidx2[filas_obs]/Tidx[filas_obs] - 1).max()
print(f"Índice calculado en {time.time()-t0:.1f} s; convergencia de la malla = {VERIF['malla']:.1e}")
T_rel = Tidx/Tidx[fsrc]
# índice en la banda de vibración de presas (5–30 Hz)
wpa, wpb = 2*np.pi*np.array(P_BASE["f_presa"])
Tidx_p = indice_transmision(T_, fsrc, malla_omega(T_["w"], wpa, wpb, zeta), zeta)
T_rel_p = Tidx_p/Tidx_p[fsrc]

H = transferencia(T_, fsrc, filas_obs, omega, zeta)
fig, ax = plt.subplots(figsize=(9, 4))
for (lab, nd), h in zip(OBS.items(), H):
    ax.loglog(omega/(2*np.pi), np.abs(h), lw=0.8, label=f"{lab} {nd}")
ax.set_xlabel("f [Hz]"); ax.set_ylabel(r"$|H_{obs,src}|$ [m/N]"); ax.legend(fontsize=7)
ax.set_title("Receptancia transversal entre la presa y los observadores")
fig.tight_layout(); fig.savefig("figuras/fig09_transferencia.png"); plt.show()

fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
dibujar_red(ax[0], red)
lib = red["libres"]
sc = ax[0].scatter(*red["X"][lib, :2].T, c=np.log10(T_rel + 1e-30), cmap="magma", s=4, zorder=2,
                   vmin=max(np.log10(T_rel.min()+1e-30), -12), vmax=0)
ax[0].plot(*red["X"][red["id_nodo"][SRC], :2], "c*", ms=10, zorder=3)
plt.colorbar(sc, ax=ax[0], label=r"$\log_{10}(\mathcal{T}_{obs,src}/\mathcal{T}_{src,src})$")
ax[0].set_title("Mapa de transmisión (1–1000 Hz)")
ia = red["idx_anillo"]
ax[1].semilogy(red["r"]/cm, T_rel[ia].mean(1), "o-", ms=3, label="promedio del anillo")
ax[1].semilogy(red["r"]/cm, T_rel[ia[:, SRC[0]]], "s--", ms=3, label="radial de la fuente")
ax[1].semilogy(red["r"]/cm, T_rel[ia[:, (SRC[0]+N//2) % N]], "^:", ms=3, label="radial opuesto")
ax[1].set_xlabel("r [cm]"); ax[1].set_ylabel(r"$\mathcal{T}_{obs,src}/\mathcal{T}_{src,src}$")
ax[1].legend(fontsize=8); ax[1].set_title("Transmisión por anillo")
fig.tight_layout(); fig.savefig("figuras/fig10_mapa_transmision.png"); plt.show()


fig, ax = plt.subplots(figsize=(8, 3.8))
et = [f"{k}\n{v}" for k, v in OBS.items() if k != "fuente"]
xb = np.arange(len(et))
vb = [T_rel[fila_z(red, v)] for k, v in OBS.items() if k != "fuente"]
vp = [T_rel_p[fila_z(red, v)] for k, v in OBS.items() if k != "fuente"]
ax.bar(xb - 0.2, vb, 0.4, label="banda 1–1000 Hz")
ax.bar(xb + 0.2, vp, 0.4, label="banda de presas 5–30 Hz")
ax.set_yscale("log"); ax.set_xticks(xb); ax.set_xticklabels(et, fontsize=7)
ax.set_ylabel(r"$\mathcal{T}_{obs,src}/\mathcal{T}_{src,src}$"); ax.legend(fontsize=8)
ax.set_title("Transmisión presa → observadores según la banda")
fig.tight_layout(); fig.savefig("figuras/fig12_transmision_bandas.png"); plt.show()


# =====================================================================
# CELDA 9 — Sensibilidad: amortiguamiento y papel de la araña
# =====================================================================
fo1 = fila_z(red, (0, 1))
zetas = [0.01, 0.02, 0.05, 0.1, 0.2]
Tz = [indice_transmision(T_, fsrc, malla_omega(T_["w"], wa, wb, zt), zt)[fo1] for zt in zetas]
fig, ax = plt.subplots(figsize=(5.5, 3.8))
ax.loglog(zetas, Tz, "o-", label="modelo")
ax.loglog(zetas, Tz[0]*zetas[0]/np.array(zetas), "k:", label=r"$\propto 1/\zeta$")
ax.axvline(P_BASE["zeta"], color="C1", ls="--", label="valor base")
ax.set_xlabel(r"$\zeta$"); ax.set_ylabel(r"$\mathcal{T}_{(0,1),src}$ [m$^2$/(N$^2$ s)]"); ax.legend(fontsize=8)
ax.set_title("El índice diverge si ζ → 0")
fig.tight_layout(); fig.savefig("figuras/fig11_sensibilidad_zeta.png"); plt.show()

# comparación hub fijo vs hub libre con la araña
def resumen_config(P):
    r_ = construir_red(P); m_ = analisis_modal(r_); T__ = m_["T"]
    s_ = nodo_en_radio(r_, P["r_presa"]); fs = fila_z(r_, s_)
    om = malla_omega(T__["w"], wa, wb, P["zeta"])
    Ti = indice_transmision(T__, fs, om, P["zeta"])
    wp = 2*np.pi*np.array(P["f_presa"])
    Tp = indice_transmision(T__, fs, malla_omega(T__["w"], *wp, P["zeta"]), P["zeta"])
    Nn = r_["N"]
    return dict(f1=T__["f"][0], n_banda=int(np.sum((T__["f"] > P["f_banda"][0]) & (T__["f"] < P["f_banda"][1]))),
                mismo=Ti[fila_z(r_, (0, 1))]/Ti[fs], noventa=Ti[fila_z(r_, (Nn//4, 1))]/Ti[fs],
                opuesto=Ti[fila_z(r_, (Nn//2, 1))]/Ti[fs],
                mismo_p=Tp[fila_z(r_, (0, 1))]/Tp[fs], noventa_p=Tp[fila_z(r_, (Nn//4, 1))]/Tp[fs],
                opuesto_p=Tp[fila_z(r_, (Nn//2, 1))]/Tp[fs])
COMP = {"hub libre con araña (50 mg)": resumen_config(P_BASE),
        "centro fijo (anteproyecto)": resumen_config(dict(P_BASE, hub_libre=False, m_arana=0.0))}
print(f"{'':30s}{'f1 [Hz]':>10s}{'modos 1-1000Hz':>16s}{'T mismo':>11s}{'T 90°':>11s}{'T opuesto':>11s}")
for k, v in COMP.items():
    print(f"{k:30s}{v['f1']:10.2f}{v['n_banda']:16d}{v['mismo']:11.2e}{v['noventa']:11.2e}{v['opuesto']:11.2e}")
    print(f"{'   (banda 5–30 Hz)':30s}{'':26s}{v['mismo_p']:11.2e}{v['noventa_p']:11.2e}{v['opuesto_p']:11.2e}")


# =====================================================================
# CELDA 10 — Resumen para el informe y descarga de figuras
# =====================================================================
print("\n=================== RESUMEN PARA EL INFORME ===================")
print(f"N = {N}, M = {M}, gdl = {red['ndof']} (transversales {len(T_['f'])}, plano {len(P_['f'])})")
print(f"T^R: {red['T_R'][1]/uN:.1f} (hub) – {red['T_R'][-1]/uN:.1f} (marco) µN, media {red['T_R'][1:].mean()/uN:.1f} µN")
print(f"Transversales: f1 = {T_['f'][0]:.2f} Hz, f2 = {T_['f'][1]:.2f} Hz, fmax = {T_['f'].max():.0f} Hz, "
      f"en banda: {int(np.sum((T_['f']>1)&(T_['f']<1000)))}")
print(f"En el plano:   f1 = {P_['f'][0]:.2f} Hz, fmax = {P_['f'].max():.0f} Hz")
print(f"Velocidad de onda transversal: {c_fit:.0f} m/s (estimada {c_ref:.0f} m/s)")
for lab, nd in OBS.items():
    print(f"  T_rel {lab:26s} {str(nd):9s} = {T_rel[fila_z(red, nd)]:.2e} (1–1000 Hz)   "
          f"{T_rel_p[fila_z(red, nd)]:.2e} (5–30 Hz)")
print("Verificación:", {k: (f"{v:.1e}" if isinstance(v, float) else v) for k, v in VERIF.items()})
print("================================================================")

shutil.make_archive("figuras_avance", "zip", "figuras")
try:
    from google.colab import files
    files.download("figuras_avance.zip")
except ImportError:
    print("Figuras en ./figuras y en figuras_avance.zip")