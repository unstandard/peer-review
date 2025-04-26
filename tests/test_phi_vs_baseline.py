import statistics as st
import sys, pathlib
root = pathlib.Path(__file__).resolve().parents[1]
sys.path.append(str(root))
from golden_ratio_nn_demo import run_once

ACC_B, ACC_PHI = [], []
for seed in range(5):
    res = run_once(seed)
    ACC_B.append(res['acc_b'])
    ACC_PHI.append(res['acc_phi'])
    params_b = res['params_b']
    params_phi = res['params_phi']

# test 1 – Φ-net not worse by >0.01 absolute on average
assert (st.mean(ACC_B) - st.mean(ACC_PHI)) < 0.01
# test 2 – Φ-net params at least 15 % smaller
assert params_phi / params_b <= 0.85
