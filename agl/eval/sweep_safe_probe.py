"""Sweep conservative safety assumptions for the abstract probe protocol.

This is a robustness check of the reference safety gate, not a flight result.
"""
import argparse
import json
from pathlib import Path

from .safe_probe_reference import Safety, run_bank


def sweep(seed=20260918, n=24):
    rows = []
    for latency in (0.0, 0.05, 0.10, 0.20):
        for speed in (0.1, 0.2, 0.4):
            for error in (0.005, 0.02, 0.05):
                safety = Safety(speed=speed, latency=latency,
                                position_error=error)
                result = run_bank(n=n, seed=seed, safety=safety)
                real = result['summary']['real']
                swapped = result['summary']['swapped']
                rows.append({
                    'latency': latency, 'speed': speed,
                    'position_error': error,
                    'inflation': safety.inflation(),
                    'eligible': result['n_eligible'],
                    'real_successes': real['successes'],
                    'real_contacts': real['contacts'],
                    'real_interventions': real['interventions'],
                    'swapped_contacts': swapped['contacts'],
                    'swapped_interventions': swapped['interventions'],
                })
    return {'scope': 'abstract_oracle_geometry_robustness_NOT_flight_validation',
            'n_tasks': n, 'seed': seed, 'rows': rows}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--n', type=int, default=24)
    p.add_argument('--seed', type=int, default=20260918)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    result = sweep(args.seed, args.n)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x') as f:
        json.dump(result, f, indent=2, allow_nan=False)
    contacts = [r['real_contacts'] + r['swapped_contacts'] for r in result['rows']]
    print(f"cases={len(result['rows'])} max_contacts={max(contacts)}")


if __name__ == '__main__':
    main()
