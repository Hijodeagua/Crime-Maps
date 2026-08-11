"""Step 0 smoke test: simulate a 3-node multivariate Hawkes with exponential
kernels and recover its parameters with tick's learners (docs example shapes)."""
import time

import numpy as np

t0 = time.time()
from tick.hawkes import SimuHawkesExpKernels, HawkesExpKern, HawkesADM4

n_nodes = 3
baseline = np.array([0.12, 0.07, 0.05])
adjacency = np.array([
    [0.30, 0.00, 0.10],
    [0.00, 0.30, 0.00],
    [0.15, 0.10, 0.20],
])
decay = 1.0

simu = SimuHawkesExpKernels(
    adjacency=adjacency,
    decays=decay,
    baseline=baseline,
    end_time=20000,
    seed=42,
    verbose=False,
)
simu.simulate()
events = simu.timestamps
print(f"simulated events per node: {[len(t) for t in events]}")

learner = HawkesExpKern(decays=decay, penalty="l2", C=1000)
learner.fit(events)
print("HawkesExpKern baseline_hat:", np.round(learner.baseline, 3))
print("HawkesExpKern adjacency_hat:\n", np.round(learner.adjacency, 3))

adm4 = HawkesADM4(decay=decay)
adm4.fit(events)
print("HawkesADM4 baseline_hat:", np.round(adm4.baseline, 3))
print("HawkesADM4 adjacency_hat:\n", np.round(adm4.adjacency, 3))

base_err = np.max(np.abs(adm4.baseline - baseline))
adj_err = np.max(np.abs(adm4.adjacency - adjacency))
print(f"max abs error — baseline: {base_err:.3f}, adjacency: {adj_err:.3f}")
assert base_err < 0.05 and adj_err < 0.15, "parameter recovery outside tolerance"
print(f"SMOKE TEST PASSED in {time.time() - t0:.1f}s")
