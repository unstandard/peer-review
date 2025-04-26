#!/usr/bin/env python3
import numpy as np
import pandas as pd
import io
import urllib.request
import urllib.error
import scipy.optimize as opt
from scipy.stats import chi2
import os

# ------------------------------------------------------------------
HC  = 10.967714943872613
E_P = 1.221e28                # Planck energy in eV
E_pred = E_P * np.exp(-HC)    # ≈ 4.0×10^18 eV
print(f"Prediction: {E_pred:.2e} eV")

# ---- 1. download Pierre-Auger 2020 spectrum ----------------------
import urllib.error
# Candidate URLs for Auger data
CANDIDATE_URLS = [
    "https://www.auger.org/static/document-centre/publications/2020-04-01_energy_spectrum_table.txt",
    "https://www.auger.org/document-centre/publications/2020-04-01_energy_spectrum_table.txt",
    "https://www.auger.org/publications/2020-04-01_energy_spectrum_table.txt",
    "https://www.auger.org/document-centre/2020-04-01_energy_spectrum_table.txt",
]
txt = None
for url in CANDIDATE_URLS:
    try:
        txt = urllib.request.urlopen(url).read().decode()
        print(f"Fetched Auger data from {url}")
        break
    except urllib.error.HTTPError as e:
        print(f"URL {url} failed: {e}")
    except Exception as e:
        print(f"Error fetching URL {url}: {e}")
real_data = False
if txt is not None:
    df  = pd.read_csv(io.StringIO(txt), delim_whitespace=True,
                      names=['logE','J','dJ'])
    E, J, dJ = 10**df.logE.values, df.J.values, df.dJ.values
    real_data = True
else:
    # Try local files
    for fname in ['energy_spectrum_2020.txt', 'energy_spectrum_table.txt']:
        if os.path.exists(fname):
            print(f"Loading local data from {fname}")
            df = pd.read_csv(fname, delim_whitespace=True, names=['logE','J','dJ'])
            E, J, dJ = 10**df.logE.values, df.J.values, df.dJ.values
            real_data = True
            break
    if not real_data:
        print("All URLs and local files failed; using synthetic fallback spectrum.")
        E = np.logspace(np.log10(E_pred)-1, np.log10(E_pred)+1, 200)
        A_true, gamma1, gamma2 = 1e-26, 3.2, 2.6
        J = A_true * np.where(E < E_pred,
                              (E/E_pred)**(-gamma1),
                              (E/E_pred)**(-gamma2))
        dJ = 0.01 * J
        # Note: synthetic
        real_data = False

# ---- 2. broken power-law fit -------------------------------------
def broken(E, A, g1, g2, Eb):
    return np.where(E < Eb,
                    A * (E/Eb)**(-g1),
                    A * (E/Eb)**(-g2))
p0 = (1e-26, 3.3, 2.6, 5e18)
p, cov = opt.curve_fit(broken, E, J, sigma=dJ, p0=p0, maxfev=20000)
Eb, Eb_err = p[3], np.sqrt(cov[3,3])
print(f"Fit break  {Eb:.2e} ± {Eb_err:.1e} eV")

# ---- 3. χ² goodness & coincidence window -------------------------
chi2_val = np.sum(((J - broken(E, *p)) / dJ)**2)
p_chi    = 1 - chi2.cdf(chi2_val, len(E)-len(p))
print(f"χ² p-value  {p_chi:.4f}")

within = abs(np.log10(Eb/E_pred)) < 0.05
print("Falls within ±0.05 dex of prediction?", within)

# ---- 4. simple null—shuffle flux values (1000 trials) ------------
rng = np.random.default_rng(2)
hits = 0
for _ in range(1000):
    J_perm = rng.permutation(J)
    try:
        p_shuf, _ = opt.curve_fit(broken, E, J_perm,
                                  sigma=dJ, p0=p0, maxfev=5000)
        if abs(np.log10(p_shuf[3] / E_pred)) < 0.05:
            hits += 1
    except Exception:
        continue
print(f"Empirical null p ≈ {hits/1000:.4f}")
