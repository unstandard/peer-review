# golden_ratio_nn_demo.py
import time
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

PHI = (1 + 5 ** 0.5) / 2          # 1.618…

def run_once(seed=0):
    X, y = load_digits(return_X_y=True)
    X /= 16.0
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=seed
    )
    baseline = MLPClassifier(hidden_layer_sizes=(64,), activation='relu', solver='adam', max_iter=200, random_state=seed)
    t0 = time.time()
    baseline.fit(X_train, y_train)
    t_b = time.time() - t0
    y_pred = baseline.predict(X_test)
    acc_b = accuracy_score(y_test, y_pred)
    params_b = sum(np.prod(w.shape) for w in baseline.coefs_) + sum(b.size for b in baseline.intercepts_)

    h1 = int(np.ceil(64 / PHI))
    h2 = int(np.ceil(h1 / PHI))
    phi_net = MLPClassifier(hidden_layer_sizes=(h1, h2), activation='relu', solver='adam', max_iter=200, random_state=seed)
    t0 = time.time()
    phi_net.fit(X_train, y_train)
    t_phi = time.time() - t0
    y_pred_phi = phi_net.predict(X_test)
    acc_phi = accuracy_score(y_test, y_pred_phi)
    params_phi = sum(np.prod(w.shape) for w in phi_net.coefs_) + sum(b.size for b in phi_net.intercepts_)

    return {
        'acc_b': acc_b,
        'acc_phi': acc_phi,
        'params_b': params_b,
        'params_phi': params_phi,
        't_b': t_b,
        't_phi': t_phi,
    }

def run_experiment(random_state: int = 0):
    res = run_once(random_state)
    return (
        res['acc_b'],
        res['params_b'],
        res['t_b'],
        res['acc_phi'],
        res['params_phi'],
        res['t_phi'],
    )

if __name__ == "__main__":
    import statistics
    n_seeds = 20
    results = []
    print("seed  |  acc_baseline  acc_phi  params_b  params_phi  t_b(s)  t_phi(s)")
    print("------|--------------------------------------------------------------")
    for seed in range(n_seeds):
        acc_b, params_b, t_b, acc_phi, params_phi, t_phi = run_experiment(seed)
        results.append((acc_b, params_b, t_b, acc_phi, params_phi, t_phi))
        print(f"{seed:>4}  |  {acc_b:.4f}     {acc_phi:.4f}   {params_b:6}   {params_phi:6}   {t_b:.2f}   {t_phi:.2f}")
    acc_bs = [r[0] for r in results]
    acc_phis = [r[3] for r in results]
    mean_b = statistics.mean(acc_bs)
    std_b = statistics.stdev(acc_bs)
    mean_phi = statistics.mean(acc_phis)
    std_phi = statistics.stdev(acc_phis)
    delta_mu = mean_phi - mean_b
    pooled_std = (std_b**2 + std_phi**2)**0.5
    if pooled_std > 0:
        z = delta_mu / pooled_std
    else:
        z = float('nan')
    print("\nSummary over 20 seeds:")
    print(f"Baseline acc: {mean_b:.4f} ± {std_b:.4f}")
    print(f"Phi-net  acc: {mean_phi:.4f} ± {std_phi:.4f}")
    print(f"Δμ / σ = {z:.2f}")
    print(f"Mean params: Baseline={statistics.mean([r[1] for r in results]):.1f}, Phi-net={statistics.mean([r[4] for r in results]):.1f}")
    print(f"Mean time (s): Baseline={statistics.mean([r[2] for r in results]):.2f}, Phi-net={statistics.mean([r[5] for r in results]):.2f}")

if __name__ == "__main__":
    import statistics
    n_seeds = 20
    results = []
    print("seed  |  acc_baseline  acc_phi  params_b  params_phi  t_b(s)  t_phi(s)")
    print("------|--------------------------------------------------------------")
    for seed in range(n_seeds):
        acc_b, params_b, t_b, acc_phi, params_phi, t_phi = run_experiment(seed)
        results.append((acc_b, params_b, t_b, acc_phi, params_phi, t_phi))
        print(f"{seed:>4}  |  {acc_b:.4f}     {acc_phi:.4f}   {params_b:6}   {params_phi:6}   {t_b:.2f}   {t_phi:.2f}")
    acc_bs = [r[0] for r in results]
    acc_phis = [r[3] for r in results]
    mean_b = statistics.mean(acc_bs)
    std_b = statistics.stdev(acc_bs)
    mean_phi = statistics.mean(acc_phis)
    std_phi = statistics.stdev(acc_phis)
    delta_mu = mean_phi - mean_b
    pooled_std = (std_b**2 + std_phi**2)**0.5
    if pooled_std > 0:
        z = delta_mu / pooled_std
    else:
        z = float('nan')
    print("\nSummary over 20 seeds:")
    print(f"Baseline acc: {mean_b:.4f} ± {std_b:.4f}")
    print(f"Phi-net  acc: {mean_phi:.4f} ± {std_phi:.4f}")
    print(f"Δμ / σ = {z:.2f}")
    print(f"Mean params: Baseline={statistics.mean([r[1] for r in results]):.1f}, Phi-net={statistics.mean([r[4] for r in results]):.1f}")
    print(f"Mean time (s): Baseline={statistics.mean([r[2] for r in results]):.2f}, Phi-net={statistics.mean([r[5] for r in results]):.2f}")
