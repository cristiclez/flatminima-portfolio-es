"""One-off cross-validation of flatminima's GARCH/GJR MLE against the `arch` package.

`arch` is NOT a dependency of the project; install it only to reproduce this check:

    pip install arch
    python scripts/validate_against_arch.py

Data are SIMULATED (known truth), so the comparison isolates the estimator.
"""

from __future__ import annotations

import warnings

import numpy as np
from arch import arch_model

from flatminima.data.synthetic import simulate_garch_t
from flatminima.risk.garch import fit_garch

warnings.filterwarnings("ignore")

CASES = {
    "garch": dict(omega=2e-6, alpha=0.08, beta=0.90, nu=6.0, mu=3e-4, gamma=0.0),
    "gjr": dict(omega=2e-6, alpha=0.03, beta=0.90, nu=6.0, mu=3e-4, gamma=0.08),
}


def main() -> None:
    for model, truth in CASES.items():
        r, _ = simulate_garch_t(8000, rng=np.random.default_rng(11), **truth)
        mine = fit_garch(r, model, compute_se=False).params
        o = 1 if model == "gjr" else 0
        res = arch_model(r * 100, mean="Constant", vol="GARCH", p=1, o=o, q=1, dist="t").fit(
            disp="off"
        )
        p = res.params
        ref = {
            "mu": p["mu"] / 100,
            "omega": p["omega"] / 1e4,
            "alpha": p["alpha[1]"],
            "gamma": p["gamma[1]"] if o else 0.0,
            "beta": p["beta[1]"],
            "nu": p["nu"],
        }
        print(f"\n{model.upper()}  (n=8000, simulated)")
        print(f"{'param':<8}{'true':>12}{'flatminima':>14}{'arch':>14}{'|diff|':>12}")
        worst = 0.0
        for k in ("mu", "omega", "alpha", "gamma", "beta", "nu"):
            a, b = getattr(mine, k), ref[k]
            scale = max(abs(b), 1e-12)
            worst = max(worst, abs(a - b) / scale if k in ("omega", "mu") else abs(a - b))
            print(f"{k:<8}{truth[k]:>12.6f}{a:>14.6f}{b:>14.6f}{abs(a - b):>12.2e}")
        print(f"max discrepancy (relative for mu/omega, absolute otherwise): {worst:.2e}")


if __name__ == "__main__":
    main()
