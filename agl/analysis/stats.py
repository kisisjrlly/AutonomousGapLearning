"""Statistics for the paper: bootstrap CIs and two-proportion tests.

Episodes are the independent sampling unit (each = one task instance).
Attempt-level rates are resampled by episode cluster to respect dependence
between attempts of the same episode.
"""
import numpy as np


def bootstrap_ci(values, n_boot=10000, alpha=0.05, seed=0, stat=np.mean):
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        return (float("nan"), float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(n_boot, len(values)))
    boots = stat(values[idx], axis=1)
    return (float(stat(values)), float(np.percentile(boots, 100 * alpha / 2)),
            float(np.percentile(boots, 100 * (1 - alpha / 2))))


def cluster_bootstrap_rate(records, key, cluster_key="env", n_boot=10000,
                           alpha=0.05, seed=0):
    """Rate CI where records (list of dicts, e.g. attempts) cluster by episode."""
    clusters = {}
    for r in records:
        clusters.setdefault(r[cluster_key], []).append(float(r[key]))
    groups = [np.array(v) for v in clusters.values()]
    if not groups:
        return (float("nan"),) * 3
    rng = np.random.default_rng(seed)
    m = len(groups)
    point = float(np.concatenate(groups).mean())
    boots = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.integers(0, m, size=m)
        allv = np.concatenate([groups[i] for i in pick])
        boots[b] = allv.mean()
    return (point, float(np.percentile(boots, 100 * alpha / 2)),
            float(np.percentile(boots, 100 * (1 - alpha / 2))))


def two_proportion_z(x1, n1, x2, n2):
    """Two-sided two-proportion z-test. Returns (z, p)."""
    from math import erf, sqrt
    if min(n1, n2) == 0:
        return float("nan"), float("nan")
    p1, p2 = x1 / n1, x2 / n2
    p = (x1 + x2) / (n1 + n2)
    se = sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    if se == 0:
        return 0.0, 1.0
    z = (p1 - p2) / se
    pval = 2 * (1 - 0.5 * (1 + erf(abs(z) / sqrt(2))))
    return float(z), float(pval)


def paired_bootstrap_diff(before, after, n_boot=10000, alpha=0.05, seed=0):
    """CI on mean(after - before) for paired samples; also sign-test p."""
    before, after = np.asarray(before, float), np.asarray(after, float)
    d = after - before
    mean, lo, hi = bootstrap_ci(d, n_boot=n_boot, alpha=alpha, seed=seed)
    # binomial sign test (two-sided, normal approx)
    from math import erf, sqrt
    nz = d[d != 0]
    if len(nz):
        k = (nz < 0).sum()
        n = len(nz)
        z = (k - n / 2) / sqrt(n / 4)
        p_sign = 2 * (1 - 0.5 * (1 + erf(abs(z) / sqrt(2))))
    else:
        p_sign = 1.0
    return {"mean_diff": mean, "ci": [lo, hi], "p_sign": float(p_sign), "n": int(len(d))}
