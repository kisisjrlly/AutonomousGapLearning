"""Three-question offline attempt pilot using NumPy ridge baselines.

1. Does a common candidate bank have reproducibly condition-dependent rankings?
2. Do recovered probes' sensor transitions improve held-out prediction?
3. Does that improvement improve candidate choice on the same held-out settings?

This is an independent-fixture diagnostic, NOT a flown retry experiment, a
safety certificate, an ICL-emergence result, or a publication acceptance test.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import tempfile
import sys
from typing import Sequence

import numpy as np

from ..attempt.outcome import TARGET_FIELDS
from ..data.attempt_dataset import AttemptDataset, split_task_ids

PILOT_VERSION = "common_bank_sensor_ridge_v1"
PREDICTED_FIELDS = ("utility",) + tuple(TARGET_FIELDS)
INVALID = ("invalid_state", "invalid_start")


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json_exclusive(path, value):
    """Publish valid JSON atomically without overwriting an existing report."""
    path = Path(path)
    encoded = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix="." + path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(encoded + "\n")
            f.flush()
            os.fsync(f.fileno())
        os.link(tmp, path)
    finally:
        os.unlink(tmp)


def evidence_summary(evidence: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """72 numeric features from sensor transitions; no outcomes or simulator truth.

    Complete transition order is irrelevant for this deliberately simple
    baseline. Action-response cross moments are NOT a causal-identification
    theorem. dt belongs to each recorded transition, not the sampling interval.
    """
    x = np.asarray(evidence, dtype=np.float64)[np.asarray(mask, dtype=bool)]
    if not len(x):
        raise ValueError("empty recovered context")
    if x.shape[1] != 40 or not np.isfinite(x).all() or np.any(x[:, 38] <= 0):
        raise ValueError("invalid sensor transition/dt")
    pre, action, post = x[:, :17], x[:, 17:21], x[:, 21:38]
    delta = (post[:, :13] - pre[:, :13]) / x[:, 38, None]
    z = np.concatenate([pre[:, :13], action, delta], axis=1)
    cross = (action[:, :, None] * delta[:, None, 9:12]).mean(axis=0).ravel()
    return np.concatenate([z.mean(0), z.std(0), cross])


def decision_inputs(ds: AttemptDataset, group: str):
    """Return one common decision observation, never candidate-specific readings.

    Context is only the predeclared PROBE prefix. A failed probe stops that
    prefix: later independent-fixture rollouts must not masquerade as retries.
    The caller reports failed-prefix cases rather than silently dropping them.
    """
    rows = np.flatnonzero(ds.group_id == group)
    rows = rows[np.argsort(ds.attempt_index[rows])]
    probes = rows[ds.role[rows] == "probe"]
    candidates = rows[ds.role[rows] == "candidate"]
    if not len(probes) or not len(candidates):
        raise ValueError("each group requires a probe prefix and candidates")
    if ds.attempt_index[probes].max() >= ds.attempt_index[candidates].min():
        raise ValueError("probes must precede all candidates")
    if not ds.context_eligible[probes].all():
        return None
    # All candidates receive the SAME observation already available after the
    # final recovered probe. Never read candidates' initial_observation/evidence.
    last = probes[-1]
    valid = np.flatnonzero(ds.evidence_mask[last])
    current = ds.evidence[last, valid[-1], 21:38].astype(np.float64)
    summary = evidence_summary(ds.evidence[probes], ds.evidence_mask[probes])
    return candidates, current.copy(), summary


def design_matrix(spec, current, summary=None):
    """Candidate interactions allow the sensor baseline to change rankings."""
    spec = np.asarray(spec, dtype=np.float64)
    current = np.asarray(current, dtype=np.float64)
    if current.ndim == 1:
        current = np.broadcast_to(current, (len(spec), len(current)))
    current_interactions = (spec[:, :, None] * current[:, None, :]).reshape(len(spec), -1)
    base = np.concatenate([spec, spec**2, current, current**2, current_interactions], axis=1)
    if summary is None:
        return base
    summary = np.asarray(summary, dtype=np.float64)
    if summary.ndim == 1:
        summary = np.broadcast_to(summary, (len(spec), len(summary)))
    interactions = (spec[:, :, None] * summary[:, None, :]).reshape(len(spec), -1)
    return np.concatenate([base, summary, interactions], axis=1)


def observed_utility(ds: AttemptDataset, rows, time_cost: float):
    """Controller/protocol-conditional utility, never intrinsic feasibility.

    +1 success, 0 recovered, -1 contact/other valid termination, minus elapsed
    simulated time * time_cost. Invalid numerical/initial states remain masked.
    All constituent outcomes and costs are reported separately.
    """
    rows = np.asarray(rows)
    status = ds.status[rows]
    valid = ~np.isin(status, INVALID)
    dt = float(ds.meta["sim_config"]["sim"]["dt_ctrl"])
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("invalid control timestep")
    elapsed = ds.steps[rows] * dt
    score = np.where(status == "success", 1., np.where(status == "recovered", 0., -1.))
    return score - time_cost * elapsed, valid, elapsed


def grouped_interval(values, group_ids, *, seed=0, draws=1000):
    """Paired bootstrap over PHYSICAL SETTINGS, not correlated candidate rows."""
    values, group_ids = np.asarray(values, float), np.asarray(group_ids)
    valid = np.isfinite(values)
    group_means = np.array([values[(group_ids == g) & valid].mean()
                           for g in np.unique(group_ids[valid])])
    if not len(group_means):
        return {"mean": None, "ci95": None, "physical_groups": 0}
    result = {"mean": float(group_means.mean()), "ci95": None,
              "physical_groups": int(len(group_means))}
    if len(group_means) >= 2:
        rng = np.random.default_rng(seed)
        boot = group_means[rng.integers(len(group_means), size=(draws, len(group_means)))].mean(1)
        result["ci95"] = np.quantile(boot, [.025, .975]).tolist()
    return result


@dataclass
class MaskedRidge:
    """Train-only standardization and group-balanced masked regression."""
    x_mean: np.ndarray
    x_scale: np.ndarray
    y_mean: np.ndarray
    y_scale: np.ndarray
    coef: np.ndarray
    available: np.ndarray

    @classmethod
    def fit(cls, x, y, mask, groups, alpha):
        x, y = np.asarray(x, float), np.asarray(y, float)
        mask = np.asarray(mask, bool)
        if not len(x) or not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError("invalid training data")
        if not np.isfinite(alpha) or alpha <= 0:
            raise ValueError("positive ridge penalty required")
        xm, xs = x.mean(0), x.std(0)
        xs = np.where(xs > 1e-8, xs, 1.)
        z = (x - xm) / xs
        ym, ys = np.zeros(y.shape[1]), np.ones(y.shape[1])
        coef = np.zeros((x.shape[1], y.shape[1]))
        available = np.zeros(y.shape[1], bool)
        groups = np.asarray(groups)
        for j in range(y.shape[1]):
            ok = mask[:, j]
            if ok.sum() < 2 or len(np.unique(groups[ok])) < 2:
                continue
            available[j] = True
            # Floors prevent a constant training target from exploding the validation scale.
            floors = (.25, .05, .05, .05, .05, .1, .1, .25, .25, .25, .25)
            floor = floors[j] if y.shape[1] == len(floors) else 1e-6
            ym[j], ys[j] = y[ok, j].mean(), max(y[ok, j].std(), floor)
            _, inverse, counts = np.unique(groups[ok], return_inverse=True, return_counts=True)
            weight = 1. / counts[inverse]
            weight *= len(weight) / weight.sum()
            zz = z[ok] * np.sqrt(weight[:, None])
            yy = (y[ok, j] - ym[j]) / ys[j] * np.sqrt(weight)
            # Dual form avoids allocating a large feature-feature matrix.
            if zz.shape[0] <= zz.shape[1]:
                coef[:, j] = zz.T @ np.linalg.solve(zz @ zz.T + alpha * np.eye(len(zz)), yy)
            else:
                coef[:, j] = np.linalg.solve(zz.T @ zz + alpha * np.eye(zz.shape[1]), zz.T @ yy)
        return cls(xm, xs, ym, ys, coef, available)

    def predict(self, x):
        return (((np.asarray(x) - self.x_mean) / self.x_scale) @ self.coef) * self.y_scale + self.y_mean


def _mse_rows(y, mask, prediction, model):
    use = mask & model.available[None, :]
    error = ((y - prediction) / model.y_scale)**2
    return np.divide((error * use).sum(1), use.sum(1),
                     out=np.full(len(y), np.nan), where=use.sum(1) > 0)


def _choose_ridge(x, y, mask, groups, train, val, penalties):
    options = []
    for alpha in penalties:
        model = MaskedRidge.fit(x[train], y[train], mask[train], groups[train], alpha)
        error = _mse_rows(y[val], mask[val], model.predict(x[val]), model)
        means = [np.nanmean(error[groups[val] == g]) for g in np.unique(groups[val])]
        options.append((float(np.mean(means)), float(alpha), model))
    options.sort(key=lambda t: (t[0], t[1]))
    if not np.isfinite(options[0][0]) or not options[0][2].available[0]:
        raise ValueError("insufficient validation/utility supervision")
    loss, alpha, model = options[0]
    # Do not refit on validation: both normalizers remain training-only.
    return model, {"alpha": alpha, "validation_normalized_mse": loss}


def check_common_bank(datasets: Sequence[AttemptDataset]):
    if len(datasets) < 2:
        raise ValueError("at least two independent rollout repetitions required")
    for ds in datasets:
        for name in ('group_id', 'status', 'role'):
            if getattr(ds, name).dtype.kind == 'S':
                value = getattr(ds, name).astype('U')
                setattr(ds, name, value); ds.arrays[name] = value
    first = datasets[0]
    groups = np.unique(first.group_id)
    if len(groups) < 6:
        raise ValueError("at least six physical groups required for a pilot split")
    fingerprint = first.meta.get("fixed_specs")
    if fingerprint is None:
        raise ValueError("random candidate datasets are not a common-bank experiment")
    for ds in datasets:
        if not np.array_equal(groups, np.unique(ds.group_id)):
            raise ValueError("repetitions must contain the same physical settings")
        for k in ("fixed_specs", "controller_version", "controller", "executor_options",
                  "distribution", "sampling_mode", "batch_tasks", "gust_disabled", "wind_model", "task_mode",
                  "probes_per_task", "source_sha256"):
            if ds.meta.get(k) != first.meta.get(k):
                raise ValueError(f"mixed execution protocol: {k}")
        if ds.meta["sim_config"] != first.meta["sim_config"]:
            raise ValueError("mixed simulation settings")
        for group in groups:
            rows = np.flatnonzero(ds.group_id == group)
            rows = rows[np.argsort(ds.attempt_index[rows])]
            if not np.array_equal(ds.spec[rows], np.asarray(fingerprint, np.float32)):
                raise ValueError("candidate bank is not identical across physical groups")
            expected = np.arange(len(rows)) < int(ds.meta["probes_per_task"])
            if not np.array_equal(ds.role[rows] == "probe", expected):
                raise ValueError("probe prefix mismatch")
    seeds = [ds.meta.get("rollout_seed") for ds in datasets]
    if any(s is None for s in seeds) or len(set(seeds)) != len(seeds):
        raise ValueError("distinct recorded rollout seeds required; repeated files are not replication")
    return groups


def evaluate(paths, *, split_seed=17, time_cost=.01, penalties=(1., 10., 100.), bootstrap_draws=1000):
    paths = [Path(p) for p in paths]
    if not np.isfinite(time_cost) or time_cost < 0:
        raise ValueError("time_cost must be nonnegative finite")
    if isinstance(bootstrap_draws, bool) or not isinstance(bootstrap_draws, int) or bootstrap_draws < 1:
        raise ValueError("positive bootstrap draw count required")
    if not penalties or any(not np.isfinite(a) or a <= 0 for a in penalties):
        raise ValueError('positive ridge penalty candidates required')
    checksums = [file_sha256(p) for p in paths]
    if len(set(checksums)) != len(checksums):
        raise ValueError("duplicate input files cannot be counted as repetitions")
    datasets = [AttemptDataset(p) for p in paths]
    groups = check_common_bank(datasets)
    parts = split_task_ids(groups, train_frac=.6, val_frac=.2, seed=split_seed)
    n_probe = int(datasets[0].meta["probes_per_task"])
    spec_bank = np.asarray(datasets[0].meta["fixed_specs"], float)[n_probe:]
    gindex = {g: i for i, g in enumerate(groups)}
    utility = np.full((len(groups), len(datasets), len(spec_bank)), np.nan)
    all_status = []
    decisions = []
    blocked = []
    for repetition, ds in enumerate(datasets):
        for group in groups:
            rows = np.flatnonzero(ds.group_id == group)
            rows = rows[np.argsort(ds.attempt_index[rows])]
            probes, candidates = rows[:n_probe], rows[n_probe:]
            u, valid, elapsed = observed_utility(ds, candidates, time_cost)
            utility[gindex[group], repetition, valid] = u[valid]
            all_status.extend(ds.status[rows].tolist())
            shared = decision_inputs(ds, group)
            if shared is None:
                blocked.append({"group": str(group), "repetition": repetition,
                                "reason": "probe_prefix_not_fully_recovered",
                                "probe_status": ds.status[probes].tolist()})
                continue
            if not valid.all():
                blocked.append({"group": str(group), "repetition": repetition,
                                "reason": "invalid_candidate_matrix"})
                continue
            _, current, summary = shared
            pu, _, pt = observed_utility(ds, probes, time_cost)
            decisions.append({"group": str(group), "rep": repetition,
                "base": design_matrix(ds.spec[candidates], current),
                "context": design_matrix(ds.spec[candidates], current, summary),
                "current": current, "summary": summary,
                "y": np.column_stack([u, ds.target[candidates]]),
                "mask": np.column_stack([valid, ds.target_mask[candidates]]),
                "status": ds.status[candidates].copy(), "seconds": elapsed,
                "probe_seconds": float(pt.sum()), "probe_utility": float(pu.sum())})

    is_synthetic = any(str(ds.meta.get('sampling_mode', '')).startswith('synthetic') for ds in datasets)
    report = {"version": PILOT_VERSION,
        "scope": "synthetic_logic_test_NOT_drone_result" if is_synthetic else "offline_independent_fixture_pilot",
        "synthetic_data": is_synthetic,
        "evaluator_sha256": file_sha256(Path(__file__)),
        "runtime": {"python": sys.version.split()[0], "numpy": np.__version__},
        "sources": [{"path": str(p), "sha256": s} for p, s in zip(paths, checksums)],
        "split_seed": split_seed, "split_groups": {k: v.tolist() for k, v in parts.items()},
        "physical_groups": len(groups), "rollout_repetitions": len(datasets),
        "candidates_per_group": len(spec_bank), "probe_count": n_probe,
        "utility_definition": {"success": 1., "recovered": 0., "contact_other_terminal": -1.,
                               "time_cost_per_simulated_second": time_cost,
                               "invalid": "masked, never converted to safe/failure outcome"},
        "status_counts": {s: all_status.count(s) for s in sorted(set(all_status))},
        "blocked_decisions": blocked, "eligible_decisions": len(decisions),
        "limitations": ["Fixed geometry; privileged pose/geometry executor, not an onboard policy.",
          "Independent fixture trials, not physically flown retries; relocation cost is unmeasured.",
          "Probe contacts/failures remain in status counts; predictor comparisons condition on recovered prefixes.",
          "Simple ridge summaries are baselines, not a neural world model or active experiment selector.",
          "Intervals resample physical groups; small pilots are not confirmatory experiments.",
          "Held-out settings are sampled from this generator; this is not out-of-mechanism generalization.",
          "Only one executor is evaluated here; stronger controllers/system-identification baselines remain necessary.",
          "No learned score is used as a safety certificate."], "flight_approval": False}

    # Gate 1: choose each condition's optimum on one repetition and evaluate it
    # on another. Never use the same stochastic outcome both to select and score.
    train_idx = [gindex[g] for g in parts["train"]]
    test_idx = [gindex[g] for g in parts["test"]]
    complete_train = [i for i in train_idx if np.isfinite(utility[i]).all()]
    complete_test = [i for i in test_idx if np.isfinite(utility[i]).all()]
    gate1 = {"complete_train_groups": len(complete_train), "complete_test_groups": len(complete_test),
             "incomplete_groups": groups[~np.isfinite(utility).all(axis=(1, 2))].tolist()}
    fixed_choice = None
    if complete_train and complete_test:
        fixed_choice = int(np.argmax(utility[complete_train].mean(axis=(0, 1))))
        gains, labels, chosen = [], [], []
        for gi in complete_test:
            for r in range(len(datasets)):
                selector, scorer = r, (r + 1) % len(datasets)
                best = int(np.argmax(utility[gi, selector]))
                gains.append(utility[gi, scorer, best] - utility[gi, scorer, fixed_choice])
                labels.append(groups[gi]); chosen.append(best)
        gate1.update(train_selected_fixed_candidate=fixed_choice,
                     condition_selected_candidates=sorted(set(chosen)),
                     cross_repeat_choice_gain=grouped_interval(gains, labels, seed=split_seed, draws=bootstrap_draws),
                     interpretation="Privileged diagnostic of candidate-ranking variation; not a deployable oracle.")
    else:
        gate1["unavailable_reason"] = "no complete train/test candidate matrices"
    report["question_1_strategy_necessity"] = gate1

    # A failed/unsafe prefix is NOT replaced by history from later fixture trials.
    # Keep its denominator above; refuse to make a learning claim without splits.
    decision_groups = np.array([d["group"] for d in decisions])
    split_counts = {k: int(np.isin(decision_groups, ids).sum()) for k, ids in parts.items()}
    report["eligible_decisions_by_split"] = split_counts
    physical_counts = {k: len(np.unique(decision_groups[np.isin(decision_groups, ids)]))
                       for k, ids in parts.items()}
    report["eligible_physical_groups_by_split"] = physical_counts
    if any(v == 0 for v in split_counts.values()) or physical_counts['train'] < 2:
        report["question_2_sensor_value"] = {"unavailable_reason": "insufficient eligible physical groups in the fixed split"}
        report["question_3_decision_value"] = {"unavailable_reason": "insufficient eligible physical groups in the fixed split"}
        return report
    x0 = np.concatenate([d["base"] for d in decisions])
    xc = np.concatenate([d["context"] for d in decisions])
    y = np.concatenate([d["y"] for d in decisions])
    mask = np.concatenate([d["mask"] for d in decisions])
    row_groups = np.repeat(decision_groups, len(spec_bank))
    train = np.isin(row_groups, parts["train"])
    val = np.isin(row_groups, parts["val"])
    m0, fit0 = _choose_ridge(x0, y, mask, row_groups, train, val, penalties)
    mc, fitc = _choose_ridge(xc, y, mask, row_groups, train, val, penalties)
    test_decisions = [d for d in decisions if d["group"] in parts["test"]]
    # Swaps are within held-out settings, same fixed probe protocol, and never
    # selected using hidden parameter labels or candidate outcomes.
    summary_by_group = {}
    for d in test_decisions:
        summary_by_group.setdefault((d['group'], d['rep']), d['summary'])
    predicted = []
    for d in test_decisions:
        peers = sorted(g for (g, r) in summary_by_group if r == d['rep'])
        other = peers[(peers.index(d['group']) + 1) % len(peers)]
        swap = summary_by_group[(other, d['rep'])] if len(peers) > 1 else None
        p0, pc = m0.predict(d["base"]), mc.predict(d["context"])
        ps = mc.predict(design_matrix(spec_bank, d['current'], swap)) if swap is not None else None
        predicted.append((d, p0, pc, ps))
    labels, errors0, errorsc, errorss = [], [], [], []
    for d, p0, pc, ps in predicted:
        # Use one training-derived output scale for the paired error comparison.
        common = d["mask"] & (m0.available & mc.available)[None, :]
        for p, dest in ((p0, errors0), (pc, errorsc), (ps, errorss)):
            if p is None:
                dest.append(np.nan)
            else:
                per_row = _mse_rows(d["y"], common, p, m0)
                dest.append(float(np.nanmean(per_row)))
        labels.append(d['group'])
    details = {}
    for j, name in enumerate(PREDICTED_FIELDS):
        e0, ec, gids = [], [], []
        if not (m0.available[j] and mc.available[j]):
            details[name] = {"unavailable_reason": "not enough training supervision"}; continue
        for d, p0, pc, _ in predicted:
            valid = d['mask'][:, j]
            if valid.any():
                e0.append(float(np.mean((d['y'][valid, j] - p0[valid, j])**2)))
                ec.append(float(np.mean((d['y'][valid, j] - pc[valid, j])**2))); gids.append(d['group'])
        details[name] = {"no_context_mse": grouped_interval(e0, gids, seed=split_seed, draws=bootstrap_draws),
                         "sensor_context_mse": grouped_interval(ec, gids, seed=split_seed, draws=bootstrap_draws)}
    report["question_2_sensor_value"] = {"no_context_fit": fit0, "sensor_context_fit": fitc,
        "normalized_mse_reduction": grouped_interval(np.array(errors0)-errorsc, labels, seed=split_seed, draws=bootstrap_draws),
        "swapped_minus_correct_mse": grouped_interval(np.array(errorss)-errorsc, labels, seed=split_seed, draws=bootstrap_draws),
        "by_target": details,
        "interpretation": "Positive error reduction supports this baseline only; null results do not prove information is absent."}

    choices = []
    for d, p0, pc, ps in predicted:
        chosen = {"no_context": int(np.argmax(p0[:, 0])), "sensor_context": int(np.argmax(pc[:, 0]))}
        if fixed_choice is not None: chosen["train_fixed"] = fixed_choice
        if ps is not None: chosen["swapped_context"] = int(np.argmax(ps[:, 0]))
        method = {}
        for name, c in chosen.items():
            method[name] = {"candidate": c, "utility": float(d['y'][c, 0]),
                            "status": str(d['status'][c]), "candidate_seconds": float(d['seconds'][c]),
                            "candidate_plus_probe_seconds": float(d['seconds'][c]+d['probe_seconds'])}
        choices.append({"group": d['group'], "repetition": d['rep'], "methods": method,
                        "probe_seconds": d['probe_seconds']})
    differences = [r['methods']['sensor_context']['utility']-r['methods']['no_context']['utility'] for r in choices]
    contacts = [float(r['methods']['sensor_context']['status']=='contact')-
                float(r['methods']['no_context']['status']=='contact') for r in choices]
    time_charged_gain = [r['methods']['sensor_context']['utility'] - time_cost*r['probe_seconds']
                         - r['methods']['train_fixed']['utility']
                         for r in choices if 'train_fixed' in r['methods']]
    time_charged_groups = [r['group'] for r in choices if 'train_fixed' in r['methods']]
    report["question_3_decision_value"] = {
        "conditional_gain_charging_probe_time_vs_fixed_without_probe": grouped_interval(
            time_charged_gain, time_charged_groups, seed=split_seed, draws=bootstrap_draws),
        "context_minus_no_context_utility": grouped_interval(differences, labels, seed=split_seed, draws=bootstrap_draws),
        "context_minus_no_context_contact_rate": grouped_interval(contacts, labels, seed=split_seed, draws=bootstrap_draws),
        "choices": choices,
        "budget_rule": "No-context/context comparisons share the same recovered probes and candidate menu.",
        "net_benefit_status": "NOT_ESTABLISHED: failed-probe branches and real relocation/replanning costs need continuous execution.",
        "note": "Greater average prediction accuracy does not imply greater decision value; compare both questions separately."}
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', nargs='+', required=True, help='V1 common-bank repeated datasets')
    p.add_argument('--out', required=True)
    p.add_argument('--split-seed', type=int, default=17)
    p.add_argument('--time-cost', type=float, default=.01)
    args = p.parse_args()
    if Path(args.out).exists(): raise FileExistsError(args.out)
    result = evaluate(args.input, split_seed=args.split_seed, time_cost=args.time_cost)
    write_json_exclusive(args.out, result)
    print(json.dumps({k: result[k] for k in ('scope', 'physical_groups', 'eligible_decisions', 'flight_approval')}, indent=2))


if __name__ == '__main__':
    main()
