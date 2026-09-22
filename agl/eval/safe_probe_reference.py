"""CPU-only, oracle-geometry protocol harness; NOT a drone/vision controller.

Axis-aligned, rest-to-rest holonomic segments use a conservative inflated-body
test. An abstract view-gated sensor reveals a hidden rear aperture. This tests
experimental plumbing, not learned adaptation or real-flight safety.
"""
import argparse
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import random


@dataclass(frozen=True)
class Safety:
    radius: float = 0.16
    margin: float = 0.03
    position_error: float = 0.01
    speed: float = 0.2
    latency: float = 0.05
    brake_accel: float = 1.0

    def inflation(self):
        if min(self.radius, self.margin, self.position_error,
               self.speed, self.latency) < 0 or self.brake_accel <= 0:
            raise ValueError("Invalid safety parameters")
        return (self.radius + self.margin + self.position_error
                + self.speed * self.latency + self.speed**2 / (2 * self.brake_accel))


@dataclass(frozen=True)
class Task:
    center: float
    width: float

    def obstacles(self):
        # Front opening is identical for every task; rear opening is latent.
        return [(2.0, 2.15, -5., -.7), (2.0, 2.15, .7, 5.),
                (2.15, 2.4, -5., self.center - self.width / 2),
                (2.15, 2.4, self.center + self.width / 2, 5.)]


def clearance(a, b, task):
    """Exact center-to-obstacle distance for an axis-aligned segment."""
    if a[0] != b[0] and a[1] != b[1]:
        raise ValueError("Reference supports axis-aligned segments only")
    x0, x1 = sorted((a[0], b[0]))
    y0, y1 = sorted((a[1], b[1]))
    return min(math.hypot(max(rx0 - x1, x0 - rx1, 0.),
                          max(ry0 - y1, y0 - ry1, 0.))
               for rx0, rx1, ry0, ry1 in task.obstacles())


def execute(task, waypoints, safety):
    """Oracle rejects an entire segment before motion; no dynamic simulation.

    Every accepted segment is reversible in this static holonomic reference.
    Rejection means remain at the last rest state, not instant mid-flight stop.
    """
    threshold = safety.inflation()
    position = (0., 0.)
    trace = [position]
    minimum = math.inf
    distance = 0.
    intervention = False
    for target in waypoints:
        d = clearance(position, target, task)
        if d <= threshold:
            intervention = True
            break
        minimum = min(minimum, d - safety.radius)
        distance += math.dist(position, target)
        position = target
        trace.append(position)
    return {"trace": trace, "intervention": intervention,
            "min_clearance": None if math.isinf(minimum) else minimum,
            "contact": minimum <= 0., "distance": distance,
            "success": position[0] >= 3.}


def observe(task, position, rng, error=.005):
    """ABSTRACT sensor, not ray tracing. Noise bound is known by construction."""
    if not (1.2 <= position[0] <= 1.5 and abs(position[1]) >= .35):
        return None
    return {"center": task.center + rng.uniform(-error, error),
            "width": task.width, "center_error": error,
            "source": "abstract_view_gated_sensor"}


def plan(evidence, safety):
    # Deliberately simple scripted reference, no trained policy.
    if evidence is None:
        return [], "insufficient_evidence"
    if evidence["width"] / 2 - evidence["center_error"] <= safety.inflation():
        return [], "insufficient_certified_margin"
    y = evidence["center"]
    return [(0., y), (3., y)], "commit"


def run_bank(n=24, seed=20260918, safety=None):
    if n <= 0:
        raise ValueError("n must be positive")
    safety = safety or Safety()
    safety.inflation()
    rng = random.Random(seed)
    rows = []
    for i in range(n):
        task = Task((-.22 if i % 2 == 0 else .22), .25 if i % 3 == 0 else .52)
        probe = execute(task, [(1.4, 0.), (1.4, .4)], safety)
        evidence = observe(task, probe["trace"][-1], rng)
        # Explicitly validate return along the accepted path, not teleportation.
        reverse = list(reversed(probe["trace"]))
        retreat_ok = all(clearance(a, b, task) > safety.inflation()
                         for a, b in zip(reverse, reverse[1:]))
        if probe["intervention"] or not retreat_ok:
            rows.append({"task_id": i, "task": asdict(task), "probe": probe,
                         "retreat_ok": retreat_ok, "eligible": False})
            continue
        swapped = dict(evidence, center=-evidence["center"])
        # All branches start from the same returned rest state (0,0).
        branches = {}
        for name, history in (("real", evidence), ("removed", None),
                              ("swapped", swapped)):
            path, decision = plan(history, safety)
            branches[name] = dict(execute(task, path, safety), decision=decision,
                                  start=[0., 0.], evidence=history)
        rows.append({"task_id": i, "task": asdict(task), "probe": probe,
                     "retreat_trace": reverse, "retreat_ok": retreat_ok,
                     "eligible": True, "branches": branches})
    eligible = [r for r in rows if r["eligible"]]
    summary = {}
    for name in ("real", "removed", "swapped"):
        bs = [r["branches"][name] for r in eligible]
        summary[name] = {"n": len(bs), "successes": sum(b["success"] for b in bs),
                         "contacts": sum(b["contact"] for b in bs),
                         "interventions": sum(b["intervention"] for b in bs),
                         "abstentions": sum(b["decision"] != "commit" for b in bs)}
    return {"scope": "abstract_oracle_geometry_reference_NOT_flight_validation",
            "seed": seed, "safety": asdict(safety), "n_tasks": n,
            "n_eligible": len(eligible), "summary": summary, "tasks": rows}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n", type=int, default=24)
    p.add_argument("--seed", type=int, default=20260918)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    result = run_bank(args.n, args.seed)
    # Do not silently overwrite prior evidence.
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as f:
        json.dump(result, f, indent=2, allow_nan=False)
    print(json.dumps({k: v for k, v in result.items() if k != "tasks"}, indent=2))


if __name__ == "__main__":
    main()
