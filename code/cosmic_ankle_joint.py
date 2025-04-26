#!/usr/bin/env python3
"""Joint Auger + Telescope-Array ankle analysis with improved statistics.

Features
--------
1.  Downloads Auger-2020 energy spectrum (plain-text table).
2.  Attempts to fetch a Telescope-Array 2019 spectrum table if available or
    loads a local TA file (TA_2019_energy_spectrum.txt) placed by the user.
3.  Performs a *joint* broken-power-law fit with one shared break energy.
4.  Computes χ² goodness of fit.
5.  Likelihood-ratio test (free break vs. break fixed at E_pred).
6.  Monte-Carlo null model that preserves slopes: draws (g1,g2) from the
    covariance of the free fit and adds Gaussian noise with σ=dJ.
7.  Propagates ±14 % Auger systematic energy-scale uncertainty.

Output
------
Prints key statistics and empirical/null p-values.
"""
import io
import os
import urllib.request, urllib.error
import numpy as np
import pandas as pd
import scipy.optimize as opt
from scipy.stats import chi2
import warnings
from scipy.optimize import OptimizeWarning
# Suppress optimization warnings and others
warnings.filterwarnings('ignore', category=OptimizeWarning)
warnings.filterwarnings('ignore')
from decimal import Decimal, getcontext

# -----------------------------------------------------------------------------
# High-precision Handshake Constant (100 decimal places)
# -----------------------------------------------------------------------------
# Set enough working precision (guard digits included)
getcontext().prec = 120

# Handshake constant
HC = Decimal("10.9677149438726125794913598226408314777631955170550152676341703163534016547475207221859489227825314671")
E_P_DEC = Decimal("1.221e28")  # Planck energy in eV
E_pred_dec = E_P_DEC * getcontext().exp(-HC * 2)  # exp(-2 HC) → ≈ 4×10^18 eV
E_pred = float(E_pred_dec)

print(f"Handshake-Constant prediction E_pred = {E_pred:.2e} eV (exp(-2 HC))\n")  # working ankle prediction

# --------------------------------------------------------------------------------------
# Fetch utilities
# --------------------------------------------------------------------------------------
AUGER_URLS = [
    "https://www.auger.org/static/document-centre/publications/2020-04-01_energy_spectrum_table.txt",
    "https://www.auger.org/document-centre/publications/2020-04-01_energy_spectrum_table.txt",
    # GitHub fallback (raw text table)
    "https://raw.githubusercontent.com/carmeloevoli/The_CR_Spectrum/master/data/allparticle/allparticle_AUGER_Etot.txt",
    # Pinned commit fallback (never disappears)
    "https://raw.githubusercontent.com/carmeloevoli/The_CR_Spectrum/d11dfa5d8d9c1c27579d2f06849ff4f37d366622/data/allparticle/allparticle_AUGER_Etot.txt",
]
TA_URLS = [
    # Official TA links (may fail)
    "https://www.telescopearray.org/static/documents/2019_TA_energy_spectrum.txt",
    "https://www.telescopearray.org/template/images/TA_SD_energy_spectrum_2019.txt",
    # GitHub fallback (moving head)
    "https://raw.githubusercontent.com/carmeloevoli/The_CR_Spectrum/master/data/allparticle/allparticle_TA_Etot.txt",
    # Pinned commit fallback
    "https://raw.githubusercontent.com/carmeloevoli/The_CR_Spectrum/d11dfa5d8d9c1c27579d2f06849ff4f37d366622/data/allparticle/allparticle_TA_Etot.txt",
]

def fetch_table(urls, local_fallback: str | None = None):
    # Prefer local fallback if present
    if local_fallback and os.path.exists(local_fallback):
        print(f"Loaded local file {local_fallback}")
        return open(local_fallback).read()
    txt = None
    for url in urls:
        try:
            txt = urllib.request.urlopen(url, timeout=15).read().decode()
            print(f"Fetched data from {url}")
            break
        except urllib.error.HTTPError as e:
            print(f"URL failed: {url} : {e.code}")
        except Exception as e:
            print(f"Error fetching {url}: {e}")
    if txt is None and local_fallback and os.path.exists(local_fallback):
        print(f"Loaded local file {local_fallback}")
        txt = open(local_fallback).read()
    return txt


def parse_table(txt):
    """Parse spectrum table.

    Supports two common formats:
    1. Three-column Auger text (log10(E/eV)  J  dJ).
    2. Six-column tables (E  J  dJ_stat  …) where *E* is in GeV or eV.
    """
    df = pd.read_csv(io.StringIO(txt), comment="#", delim_whitespace=True, header=None)

    # inspect header comments for units
    header_lines = [line for line in txt.splitlines() if line.strip().startswith('#')]
    unit_line = next((line for line in header_lines if 'Units:' in line), '')

    if df.shape[1] == 3:
        # logE format
        logE = df.iloc[:, 0].values
        E = 10 ** logE
        J = df.iloc[:, 1].values
        dJ = df.iloc[:, 2].values
    else:
        # E (possibly GeV) + J + dJ_stat...
        E = df.iloc[:, 0].values.astype(float)
        # If unit comment indicates eV, skip conversion; else if E in GeV, upconvert
        if 'E[eV]' in unit_line:
            pass
        elif np.median(E) < 1e13:
            E *= 1e9
        J = df.iloc[:, 1].values.astype(float)
        dJ = df.iloc[:, 2].values.astype(float)
    # Remove rows with non-positive or NaN uncertainties (curve_fit requires >0)
    m = (J > 0) & (dJ > 0) & np.isfinite(J) & np.isfinite(dJ)
    return E[m], J[m], dJ[m]


# --------------------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------------------

def broken(E, A, g1, g2, Eb):
    """Broken power-law (continuous at break)."""
    return np.where(E < Eb,
                    A * (E/Eb)**(-g1),
                    A * (E/Eb)**(-g2))

def triple(E, A, g1, g2, g3, Eb1, Eb2):
    # Three-slope broken power law: Eb1 < Eb2
    # < Eb1: slope g1; Eb1<=E<Eb2: slope g2; >=Eb2: slope g3 (continuous)
    slope1 = A * (E/Eb1)**(-g1)
    slope2 = A * (E/Eb1)**(-g2)
    norm3 = (Eb2/Eb1)**(-g2)
    slope3 = A * norm3 * (E/Eb2)**(-g3)
    return np.where(E < Eb1,
                    slope1,
                    np.where(E < Eb2,
                             slope2,
                             slope3))

def make_broken_with_shifts(tags):
    """Return a model *callable* that applies separate Δ energy-scale shifts to
    Auger and TA points before evaluating the broken power law.

    Parameters
    ----------
    tags : array-like of str
        Same length as *E* vector used in the fit.  Elements must be "A" (Auger)
        or "T" (TA) so the closure knows which δ to apply.
    """
    tags = np.asarray(tags)
    mask_A = tags == "A"

    def _model(E, A, g1, g2, Eb, delta_Au, delta_TA):
        E_shift = np.where(mask_A, E * (1 + delta_Au), E * (1 + delta_TA))
        return np.where(E_shift < Eb,
                        A * (E_shift / Eb) ** (-g1),
                        A * (E_shift / Eb) ** (-g2))

    return _model

def make_triple_with_shifts(tags):
    tags = np.asarray(tags)
    mask_A = tags == "A"
    def _model(E, A, g1, g2, g3, Eb1, Eb2, delta_Au, delta_TA):
        E_shift = np.where(mask_A, E*(1+delta_Au), E*(1+delta_TA))
        return triple(E_shift, A, g1, g2, g3, Eb1, Eb2)
    return _model

def fit_broken(E, J, dJ, Eb_fixed: float | None = None):
    if Eb_fixed is None:
        p0 = (1e-26, 3.3, 2.6, 5e18)
        popt, pcov = opt.curve_fit(broken, E, J, sigma=dJ, p0=p0, maxfev=20000)
    else:
        # three-parameter fit with Eb fixed
        def fixed(E, A, g1, g2):
            return broken(E, A, g1, g2, Eb_fixed)
        p0 = (1e-26, 3.3, 2.6)
        popt3, pcov3 = opt.curve_fit(fixed, E, J, sigma=dJ, p0=p0, maxfev=20000)
        popt = (*popt3, Eb_fixed)
        pcov = np.zeros((4,4))
        pcov[:3,:3] = pcov3
    return popt, pcov


def chi2_stat(E, J, dJ, params):
    return np.sum(((J - broken(E, *params)) / dJ)**2)

# --------------------------------------------------------------------------------------
# Null model preserving slopes
# --------------------------------------------------------------------------------------

def slope_preserving_null(E, J, dJ, params, pcov, n_sim=1000, rng=None):
    """Monte-Carlo p(null) with robust error handling."""
    rng = np.random.default_rng(rng)
    hits = 0
    for _ in range(n_sim):
        try:
            A_draw, g1_draw, g2_draw = rng.multivariate_normal(params[:3], pcov[:3, :3])
            Eb_draw = rng.uniform(1e18, 1e19)
            J_sim = broken(E, A_draw, g1_draw, g2_draw, Eb_draw)
            J_noisy = rng.normal(J_sim, dJ)
            popt, _ = fit_broken(E, J_noisy, dJ)
            if abs(np.log10(popt[3] / E_pred)) < 0.05:
                hits += 1
        except Exception:
            continue
    return hits / n_sim

# Monte-Carlo null for triple-slope model
def triple_preserving_null(E, J, dJ, params, pcov, model, n_sim=100000, rng=None):
    """Empirical p(null) for triple-slope fit with curvature- and shift-preserving draws."""
    rng = np.random.default_rng(rng)
    hits = 0
    for _ in range(n_sim):
        # draw all free parameters (A, g1,g2,g3, Eb1, Eb2, delta_Au, delta_TA)
        try:
            pars_draw = rng.multivariate_normal(params, pcov)
        except Exception:
            continue
        # simulate spectrum and noise
        J_sim = model(E, *pars_draw)
        J_noisy = rng.normal(J_sim, dJ)
        try:
            popt_sim, _ = opt.curve_fit(model, E, J_noisy, sigma=dJ, p0=params, maxfev=5000)
            # test if top break Eb2 aligns with E_pred
            if abs(np.log10(popt_sim[5] / E_pred)) < 0.05:
                hits += 1
        except Exception:
            continue
    return hits / n_sim

# --------------------------------------------------------------------------------------
# Main analysis
# --------------------------------------------------------------------------------------

def run():
    # Debug: check essential files present
    print("DEBUG: exists energy_spectrum_2020.txt =", os.path.exists("energy_spectrum_2020.txt"))
    print("DEBUG: exists TA_2019_energy_spectrum.txt =", os.path.exists("TA_2019_energy_spectrum.txt"))
    txt_files = sorted([f for f in os.listdir() if f.endswith(".txt")])
    print("DEBUG: txt files:", txt_files)
    print("DEBUG: run() entry")
    print()

    # 1. Load Auger
    print("DEBUG: fetching Auger")
    aug_txt = fetch_table(AUGER_URLS, "energy_spectrum_2020.txt")
    if aug_txt is None:
        print("DEBUG: aug_txt is None -> abort")
        print("Auger data unavailable → abort.")
        return
    print("DEBUG: parsing Auger data")
    E_aug, J_aug, dJ_aug = parse_table(aug_txt)

    # 2. Load Telescope Array (optional)
    print("DEBUG: checking local TA file")
    if os.path.exists("TA_2019_energy_spectrum.txt"):
        print("DEBUG: loaded local TA file TA_2019_energy_spectrum.txt")
        ta_txt = open("TA_2019_energy_spectrum.txt").read()
    else:
        print("DEBUG: fetching TA via URL")
        ta_txt = fetch_table(TA_URLS, None)
    if ta_txt is None:
        print("TA spectrum not found — proceeding with Auger only.\n")
        combined = False
        E, J, dJ = E_aug, J_aug, dJ_aug
    else:
        print("DEBUG: TA data loaded, parsing TA")
        E_ta, J_ta, dJ_ta = parse_table(ta_txt)
        print("DEBUG: TA parse success")
        # Concatenate arrays
        E   = np.concatenate([E_aug, E_ta])
        J   = np.concatenate([J_aug, J_ta])
        dJ  = np.concatenate([dJ_aug, dJ_ta])
        combined = True
        print("Using joint Auger + TA dataset.\n")
        # Build tag array once here so the closure below can capture it
        TAGS = np.array(["A"] * len(E_aug) + ["T"] * len(E_ta))
        model = make_broken_with_shifts(TAGS)
        p0 = (1e-26, 3.3, 2.6, 5e18, 0.0, 0.0)
        params_free, pcov_free = opt.curve_fit(model, E, J, sigma=dJ, p0=p0, maxfev=30000)
        Eb, Eb_err = params_free[3], np.sqrt(pcov_free[3, 3])
        # free-fit chi2 and DOF
        chi2_free = np.sum(((J - model(E, *params_free)) / dJ) ** 2)
        dof_free = len(E) - len(params_free)
        # energy-scale Gaussian penalty
        delta_Au, delta_TA = params_free[4], params_free[5]
        penalty_free = (delta_Au/0.14)**2 + (delta_TA/0.21)**2
        # penalized chi2 and p-value
        chi2_free_pen = chi2_free + penalty_free
        p_chi2 = 1 - chi2.cdf(chi2_free_pen, dof_free)

        # Fixed-break (E_pred) model – only 5 params free
        model_fixed = lambda E, A, g1, g2, delta_Au, delta_TA: model(E, A, g1, g2, E_pred, delta_Au, delta_TA)
        p0_fixed = (1e-26, 3.3, 2.6, 0.0, 0.0)
        params_fixed, _ = opt.curve_fit(model_fixed, E, J, sigma=dJ, p0=p0_fixed, maxfev=30000)
        # fixed-fit chi2
        chi2_fixed = np.sum(((J - model_fixed(E, *params_fixed)) / dJ) ** 2)
        # fixed-fit penalty
        delta_Au_fix, delta_TA_fix = params_fixed[3], params_fixed[4]
        penalty_fixed = (delta_Au_fix/0.14)**2 + (delta_TA_fix/0.21)**2
        chi2_fixed_pen = chi2_fixed + penalty_fixed
        # likelihood-ratio with penalized chi2
        delta_chi2 = chi2_fixed_pen - chi2_free_pen
        p_lr = 1 - chi2.cdf(delta_chi2, 1)

        # Null model (re-uses old four-param fit for speed)
        p_null = slope_preserving_null(E, J, dJ, params_free, pcov_free, n_sim=1000)

        delta_Au, delta_TA = params_free[4], params_free[5]
        in_band_plus  = abs(np.log10((Eb * (1 + 0.14)) / E_pred)) < 0.05
        in_band_minus = abs(np.log10((Eb * (1 - 0.14)) / E_pred)) < 0.05

        # --- Triple-slope fit (free) ---
        model_triple = make_triple_with_shifts(TAGS)
        p0_tri = (1e-26, 3.3, 2.6, 3.2, 1e18, E_pred, 0.0, 0.0)
        popt_tri, pcov_tri = opt.curve_fit(model_triple, E, J, sigma=dJ, p0=p0_tri, maxfev=50000)
        chi2_tri = np.sum(((J - model_triple(E, *popt_tri)) / dJ)**2)
        dof_tri = len(E) - len(popt_tri)
        # energy-scale penalty
        dAu_t, dTA_t = popt_tri[6], popt_tri[7]
        pen_tri = (dAu_t/0.14)**2 + (dTA_t/0.21)**2
        chi2_tri_pen = chi2_tri + pen_tri
        p_chi2_tri = 1 - chi2.cdf(chi2_tri_pen, dof_tri)

        # --- Triple-slope fit (Eb2 fixed) ---
        def model_tri_fix(E, A, g1, g2, g3, Eb1, delta_Au, delta_TA):
            return model_triple(E, A, g1, g2, g3, Eb1, E_pred, delta_Au, delta_TA)
        p0_tri_fix = (*popt_tri[:6], 0.0, 0.0)
        popt_tfix, _ = opt.curve_fit(model_tri_fix, E, J, sigma=dJ, p0=p0_tri_fix, maxfev=50000)
        chi2_tfix = np.sum(((J - model_tri_fix(E, *popt_tfix)) / dJ)**2)
        dAu_tf, dTA_tf = popt_tfix[6], popt_tfix[7]
        pen_tfix = (dAu_tf/0.14)**2 + (dTA_tf/0.21)**2
        chi2_tfix_pen = chi2_tfix + pen_tfix
        delta_chi2_tri = chi2_tfix_pen - chi2_tri_pen
        p_lr_tri = 1 - chi2.cdf(delta_chi2_tri, 1)

        # Report triple-slope results
        print("\n--- Triple-slope fit results ---")
        print(f"χ² p-value (tri free)     = {p_chi2_tri:.4f} (dof={dof_tri})")
        print(f"LR p (Eb2 fixed=E_pred)   = {p_lr_tri:.4f}")
        print()
        # triple-slope empirical null (100k trials)
        p_null_tri = triple_preserving_null(E, J, dJ, popt_tri, pcov_tri, model_triple)
        print(f"Empirical null p (tri)  = {p_null_tri:.4f}")
    # ------------------------------------------------------------------
    # Auger-only fallback (no TA data available) – keep previous 4-param fit
    # ------------------------------------------------------------------
    if not combined:
        # Fit broken power law for Auger-only data
        try:
            params_free, pcov_free = fit_broken(E, J, dJ)
        except Exception:
            print("Broken-power-law fit failed on Auger-only data; aborting.")
            return
        Eb, Eb_err = params_free[3], np.sqrt(pcov_free[3, 3])
        chi2_free = np.sum(((J - broken(E, *params_free)) / dJ) ** 2)
        dof_free = len(E) - len(params_free)
        params_fixed, _ = fit_broken(E, J, dJ, Eb_fixed=E_pred)
        chi2_fixed = np.sum(((J - broken(E, *params_fixed)) / dJ) ** 2)
        delta_chi2 = chi2_fixed - chi2_free
        p_lr = 1 - chi2.cdf(delta_chi2, 1)
        print(f"DEBUG: running 2-slope Monte-Carlo null (n=1000)")
        p_null = slope_preserving_null(E, J, dJ, params_free, pcov_free, n_sim=1000)
        print(f"DEBUG: 2-slope null p = {p_null:.4f}")
        in_band_plus  = abs(np.log10((Eb * 1.14) / E_pred)) < 0.05
        in_band_minus = abs(np.log10((Eb * 0.86) / E_pred)) < 0.05
        delta_Au = delta_TA = 0.0  # not fitted

    # ----------------------------- Report ----------------------------
    print("DEBUG: entering final report")
    print("================  RESULTS  ================")
    ds_label = "Auger+TA" if combined else "Auger"
    print(f"Dataset: {ds_label}")
    print(f"Fitted break Eb = {Eb:.2e} ± {Eb_err:.1e} eV")
    if combined:
        print(f"Δ_Au = {delta_Au:+.3f}   Δ_TA = {delta_TA:+.3f}  (energy-scale shifts)")
    # report penalized χ² p-value
    print(f"χ² p-value (free)   = {p_chi2:.4f}")
    print(f"Likelihood-ratio p  = {p_lr:.4f}")
    print(f"Empirical null p    = {p_null:.4f}")
    print(f"Systematic (+14%) window? {in_band_plus}")
    print(f"Systematic (−14%) window? {in_band_minus}")

if __name__ == "__main__":
    try:
        run()
    except Exception as e:
        import traceback; traceback.print_exc()
# End of script
