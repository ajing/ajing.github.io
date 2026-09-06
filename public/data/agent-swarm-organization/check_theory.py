"""Recheck small examples in the agent-swarm organization article.

Python 3 standard library only. No model calls, network, or source corpus.
This companion checks a subset of the original study's exact constructions.
It is not a reproduction of the full original enumeration or LLM collection.
"""

from collections import defaultdict
from fractions import Fraction as Q
from itertools import combinations, product
from pathlib import Path
import json


def coverage(actions):
    return len(set().union(*actions))


def contribution(actions, i):
    return coverage(actions) - coverage(actions[:i] + actions[i + 1 :])


def check_coverage():
    subsets = [frozenset(i for i in range(3) if mask & (1 << i)) for mask in range(8)]
    menus = list(combinations(subsets, 2))
    games = equilibria = moves = 0
    for menu in product(menus, repeat=2):
        games += 1
        profiles = list(product(*menu))
        optimum = max(map(coverage, profiles))
        for a in profiles:
            equilibrium = True
            for i in range(2):
                for b in menu[i]:
                    if b == a[i]:
                        continue
                    changed = a[:i] + (b,) + a[i + 1 :]
                    gain = contribution(changed, i) - contribution(a, i)
                    assert gain == coverage(changed) - coverage(a)
                    equilibrium &= gain <= 0
                    moves += 1
            if equilibrium:
                equilibria += 1
                assert 2 * coverage(a) >= optimum
    # A tight equilibrium: no strictly profitable unilateral move.
    a = (frozenset(), frozenset({0}))
    assert coverage(a) == 1
    assert contribution((frozenset({0}), a[1]), 0) == contribution(a, 0)
    assert contribution((a[0], frozenset({1})), 1) == contribution(a, 1)
    return {"unit_weight_games": games, "unilateral_changes": moves, "equilibria": equilibria}


def check_cycle():
    cycle = [(1, 0, 0), (1, 1, 0), (0, 1, 0), (0, 1, 1),
             (0, 0, 1), (1, 0, 1), (1, 0, 0)]
    for old, new in zip(cycle, cycle[1:]):
        changed = [i for i in range(3) if old[i] != new[i]]
        assert len(changed) == 1
        i = changed[0]
        assert old[i] == old[(i + 1) % 3]
        assert new[i] != new[(i + 1) % 3]
    for a in product((0, 1), repeat=3):
        assert any(a[i] == a[(i + 1) % 3] for i in range(3))
    return {"strict_improvement_cycle_length": len(cycle) - 1}


def check_certificates(records):
    """Per-resource inequalities imply the stated equilibrium lower bounds.

    a: users choosing this resource only in equilibrium A
    x: users choosing it in both A and comparison O
    b: users choosing it only in O
    D is the summed utility difference for unilateral deviations toward O.
    At equilibrium its weighted sum is nonnegative, hence W(O) <= mu W(A).
    This verifies lower-bound certificates; it does not rebuild tight games.
    """
    patterns = 0
    for record in records:
        n, mu = record["agents"], Q(record["mu"])
        f = [Q(0)] + [Q(x) for x in record["reward_table"]] + [Q(0)]
        assert Q(record["bound"]) * mu == 1
        for a, x, b in product(range(n + 1), repeat=3):
            if not 1 <= a + x + b <= n:
                continue
            difference = a * f[a + x] - b * f[a + x + 1]
            assert int(x + b > 0) + difference <= mu * int(a + x > 0)
            patterns += 1
    return {"certificates": len(records), "resource_patterns": patterns}


def states(m, delta):
    return [(y, x, (y - x) % m, y if reveal else -1, p / (m * m))
            for y in range(m) for x in range(m)
            for reveal, p in [(True, delta), (False, 1 - delta)] if p]


def decision_mass(group):
    masses = defaultdict(Q)
    for y, _, _, _, p in group:
        masses[y] += p
    return max(masses.values(), default=Q(0))


def split(group, action):
    groups = defaultdict(list)
    for row in group:
        groups[row[action]].append(row)
    return list(groups.values())


def solve(group, actions, budget, greedy=False):
    if budget == 0 or not actions:
        return decision_mass(group)
    choices = actions
    if greedy:
        scores = {a: sum(decision_mass(g) for g in split(group, a)) for a in actions}
        best = max(scores.values())
        choices = [a for a in actions if scores[a] == best]
    # Give greedy its most favorable tie-breaking, including downstream value.
    return max(sum(solve(g, [a for a in actions if a != action], budget - 1, greedy)
                   for g in split(group, action)) for action in choices)


def check_information():
    count = 0
    for m in (2, 3, 4, 5, 8, 10, 16, 20):
        for delta in (Q(1, 2), Q(1, 10), Q(1, m * m)):
            group = states(m, delta)
            expected = delta + (1 - delta) / m
            assert sum(row[-1] for row in group) == 1
            assert decision_mass(group) == Q(1, m)
            assert solve(group, [1, 2, 3], 2) == 1
            assert solve(group, [1, 2, 3], 2, greedy=True) == expected
            count += 1
    p = Q(3, 4)
    assert 3 * p * p - 2 * p ** 3 == Q(27, 32)
    assert Q(1, 400) + Q(399, 400) / 20 == Q(419, 8000)
    return {"exact_bayesian_instances": count, "example_greedy_accuracy": "419/8000"}


def check_measurements(data):
    """Check arithmetic of exported outcomes, not independent truth of labels."""
    total = 0
    for stage in ("visibility", "last_action"):
        rows = data[stage]["trajectories"]
        for arm, aggregate in data[stage]["aggregates"].items():
            selected = [r for r in rows if r["arm"] == arm]
            assert len(selected) == aggregate["trajectories"]
            for outcome in ("correct", "wrong", "unknown"):
                assert sum(r["outcome"] == outcome for r in selected) == aggregate[outcome]
            tokens = sum(r["complete_policy_tokens"] for r in selected)
            assert tokens == aggregate["sum_complete_policy_tokens"]
            assert Q(tokens, len(selected)) == Q(str(aggregate["mean_complete_policy_tokens"]))
            assert Q(sum(r["tool_requests"] for r in selected), len(selected)) == Q(str(aggregate["mean_tool_requests"]))
        total += len(rows)
    return {"exported_trajectories_checked": total}


if __name__ == "__main__":
    data = json.loads(Path(__file__).with_name("results.json").read_text())
    results = {
        "coverage": check_coverage(),
        "visibility": check_cycle(),
        "utility_certificates": check_certificates(data["utility_certificates"]),
        "information": check_information(),
        "measurement_arithmetic": check_measurements(data),
    }
    print(json.dumps({"status": "all assertions passed", **results}, indent=2))
