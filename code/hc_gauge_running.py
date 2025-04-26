"""
hc_gauge_running.py
Two-loop SM running with a Handshake–Constant unification scale
    Λ_U = M_P * exp(-HC).
Optional: add one light colour triplet at 2 TeV plus extended analyses:
  • gauge convergence plot
  • placeholder two-loop matching at MZ
  • Yukawa + Higgs RGE
  • toy proton-decay estimate
"""
import numpy as np
import matplotlib.pyplot as plt
from math import exp, log

# ---------------- constants & initial data -----------------
HC    = 10.967714943872613
M_P   = 1.221e28           # eV
Λ_U   = M_P * exp(-HC)     # ≃ 4.03×10^18 eV
M_Z   = 91.1876e9          # eV
Λ_thr = 2e12               # 2 TeV

# PDG MS-bar couplings at M_Z
α1_PDG, α2_PDG, α3_PDG = 0.01695, 0.033812, 0.1181

# one- and two-loop β–coefficients (GUT-normalized U(1))
b = np.array([41/10, -19/6, -7])
B2 = np.array([[199/50, 27/10, 44/5],
               [  9/10,   35/6,   12 ],
               [ 11/10,    9/2,  -26 ]])

def beta(α, b_vec=b):
    """two-loop β for gauge couplings"""
    term1 = -(b_vec/(2*np.pi)) * α**2
    term2 = -(α**3)/(8*np.pi**2) * (B2 @ α)
    return term1 + term2


def run_RGE(α0, μ0, μ1, b_vec, steps=100000):
    """RK4 integration of gauge RGEs in ln μ"""
    t0, t1 = log(μ0), log(μ1)
    h = (t1 - t0) / steps
    α = α0.copy()
    for _ in range(steps):
        k1 = beta(α, b_vec)
        k2 = beta(α + 0.5*h*k1, b_vec)
        k3 = beta(α + 0.5*h*k2, b_vec)
        k4 = beta(α +     h*k3, b_vec)
        α += h/6 * (k1 + 2*k2 + 2*k3 + k4)
    return α


def run_with_threshold(α0, add_triplet=False):
    if not add_triplet:
        return run_RGE(α0, Λ_U, M_Z, b)
    # split at Λ_thr with modified one-loop b_vec
    α_mid = run_RGE(α0, Λ_U, Λ_thr, b)
    b_shift = b + np.array([1/10, 1/6, 1])
    α_low  = run_RGE(α_mid, Λ_thr, M_Z, b_shift)
    return α_low


def tune_alpha_U():
    """Find α_U so that α2(M_Z) == PDG via binary search"""
    low, high = 0.02, 0.08
    for _ in range(50):
        aU = 0.5*(low + high)
        αU = np.array([aU, aU, aU])
        αMz = run_with_threshold(αU, add_triplet=False)
        if αMz[1] > α2_PDG:
            high = aU
        else:
            low  = aU
    return np.array([aU, aU, aU]), aU


def show(label, α_vals):
    """Print α_i vs PDG"""
    print(label)
    for lab, val, expv in zip(("α1","α2","α3"), α_vals,
                              (α1_PDG, α2_PDG, α3_PDG)):
        print(f" {lab}: {val:.6f} vs {expv:.6f} ({(val-expv)/expv:+.1%})")
    print()


def trace_gauge(αU, add_triplet=False, steps=100000):
    """Record α(μ) from Λ_U→M_Z for plotting"""
    t0, t1 = log(Λ_U), log(M_Z)
    h = (t1 - t0) / steps
    α = αU.copy()
    mus, als = [], []
    for i in range(steps+1):
        t = t0 + i*h
        μ = exp(t)
        mus.append(μ)
        als.append(α.copy())
        if i < steps:
            # choose b-vector
            if add_triplet and μ < Λ_thr:
                bv = b + np.array([1/10, 1/6, 1])
            else:
                bv = b
            # step
            k1 = -(bv/(2*np.pi)) * α**2 - (α**3)/(8*np.pi**2) * (B2 @ α)
            k2 = -(bv/(2*np.pi)) * (α+0.5*h*k1)**2 - ((α+0.5*h*k1)**3)/(8*np.pi**2) * (B2 @ (α+0.5*h*k1))
            k3 = -(bv/(2*np.pi)) * (α+0.5*h*k2)**2 - ((α+0.5*h*k2)**3)/(8*np.pi**2) * (B2 @ (α+0.5*h*k2))
            k4 = -(bv/(2*np.pi)) * (α+    h*k3)**2 - ((α+    h*k3)**3)/(8*np.pi**2) * (B2 @ (α+    h*k3))
            α += h/6 * (k1 + 2*k2 + 2*k3 + k4)
    return np.array(mus), np.array(als)


def plot_couplings(αU):
    """Plot and save 1/α_i vs log10(μ)"""
    mu_sm, a_sm = trace_gauge(αU, False)
    mu_tr, a_tr = trace_gauge(αU, True)
    plt.figure(figsize=(8,6))
    for idx, lbl in enumerate(("1/α1","1/α2","1/α3")):
        plt.plot(np.log10(mu_sm), 1/a_sm[:,idx], label=f"{lbl} SM")
        plt.plot(np.log10(mu_tr), 1/a_tr[:,idx], "--", label=f"{lbl} +triplet")
    plt.xlabel("log10(μ/eV)")
    plt.ylabel("1/α")
    plt.title("Gauge coupling unification traces")
    plt.legend(); plt.tight_layout();
    plt.savefig("gauge_unification.png"); plt.close()


def match_at_MZ(αMz):
    """Apply placeholder matching corrections at M_Z"""
    deltas = np.array([+0.00005, -0.00002, +0.00010])
    return αMz + deltas


def beta_yuk_higgs(v, α):
    """One-loop Yukawa + Higgs quartic RGEs"""
    yt, yb, yτ, lam = v
    g1sq = 4*np.pi * α[0]
    g2sq = 4*np.pi * α[1]
    g3sq = 4*np.pi * α[2]
    dyt = yt/(16*np.pi**2)*(9/2*yt**2 + 3/2*yb**2 - (17/20*g1sq + 9/4*g2sq + 8*g3sq))
    dyb = yb/(16*np.pi**2)*(9/2*yb**2 + 3/2*yt**2 + yτ**2 - (1/4*g1sq + 9/4*g2sq + 8*g3sq))
    dyτ = yτ/(16*np.pi**2)*(5/2*yτ**2 + 3*yb**2 - (9/4*g1sq + 9/4*g2sq))
    dlam = (1/(16*np.pi**2))*(24*lam**2 - 6*yt**4 - 6*yb**4 - 2*yτ**4 +
           (3/8)*(g1sq**2 + 2*g1sq*g2sq + 3*g2sq**2) +
           lam*(-9*g1sq -9*g2sq +12*yt**2 +12*yb**2 +4*yτ**2))
    return np.array([dyt, dyb, dyτ, dlam])


def run_yukawa_higgs(Y0, lam0, mu, alphas):
    """RK4 integrate Yukawa+Higgs RGEs from M_Z→Λ_U"""
    tvals = np.log(mu)
    dt = tvals[1] - tvals[0]
    v = np.array([Y0[0], Y0[1], Y0[2], lam0])
    traj = [v.copy()]
    for i in range(len(tvals)-1):
        αvals = alphas[i]
        k1 = beta_yuk_higgs(v, αvals)
        k2 = beta_yuk_higgs(v + 0.5*dt*k1, alphas[i+1])
        k3 = beta_yuk_higgs(v + 0.5*dt*k2, alphas[i+1])
        k4 = beta_yuk_higgs(v +     dt*k3, alphas[i+1])
        v = v + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
        traj.append(v.copy())
    return np.array(traj)


def compute_proton_decay_tau(M_tri, αU):
    """Toy dimension-6 proton lifetime estimate"""
    m_p = 0.938e9  # eV
    τ = M_tri**4 / (αU**2 * m_p**5)
    τ_s = τ * 6.582e-16  # ℏ in eV·s
    return τ_s / (3600*24*365)


def main():
    αU_vec, αU = tune_alpha_U()
    α_pure = run_with_threshold(αU_vec, False)
    α_trip  = run_with_threshold(αU_vec, True)

    print(f"HC     = {HC}")
    print(f"Λ_U    = {Λ_U:.2e} eV  (single e^(-HC))")
    print(f"α_U    = {αU:.4f}\n")

    show("Two-loop SM, no threshold:", α_pure)
    show("Add colour-triplet @2 TeV:", α_trip)

    plot_couplings(αU_vec)
    print("Saved gauge_unification.png\n")

    α_pure_mat = match_at_MZ(α_pure)
    show("Matched SM no threshold:", α_pure_mat)
    α_trip_mat  = match_at_MZ(α_trip)
    show("Matched +triplet:", α_trip_mat)

    # Yukawa & Higgs running
    yt0  = np.sqrt(2) * 172.5e9 / 246e9
    yb0  = np.sqrt(2) * 4.18e9   / 246e9
    yτ0  = np.sqrt(2) * 1.777e9  / 246e9
    lam0 = (125.1e9)**2 / (2 * (246e9)**2)

    # gauge trajectory for Higgs/Yukawa RG
    mu_desc, α_desc = trace_gauge(αU_vec, False)
    mu_asc,  α_asc  = mu_desc[::-1], α_desc[::-1]
    yh_traj = run_yukawa_higgs((yt0, yb0, yτ0), lam0, mu_asc, α_asc)
    ytU, ybU, yτU, lamU = yh_traj[-1]
    print(f"Yukawas at Λ_U: yt={ytU:.4f}, yb={ybU:.4f}, yτ={yτU:.4f}, λ={lamU:.4f}\n")

    τ_p = compute_proton_decay_tau(Λ_thr, αU_vec[0])
    print(f"Toy proton lifetime for 2 TeV triplet: τ_p ≃ {τ_p:.2e} years")

if __name__ == '__main__':
    main()
</file_content>
