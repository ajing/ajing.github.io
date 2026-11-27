"""Compare two acceptance rules on a fully enumerated trace space."""

import argparse
import math


STATES = ("A", "B", "C", "D")
BASE = (0.24, 0.18, 0.12, 0.46)
EXPERT = (0.0, 0.0, 1.0, 0.0)
PROPOSAL = (0.30, 0.40, 0.20, 0.10)
VALID = (0, 1, 2)

Distribution = tuple[float, ...]
Matrix = tuple[Distribution, ...]


def condition(distribution: Distribution, support: tuple[int, ...]) -> Distribution:
    mass = sum(distribution[index] for index in support)
    return tuple(
        probability / mass if index in support else 0.0
        for index, probability in enumerate(distribution)
    )


def transition_matrix(*, greedy: bool) -> Matrix:
    rows = []
    for current in VALID:
        row = [0.0] * len(STATES)
        for candidate in VALID:
            if candidate == current:
                continue
            if greedy:
                acceptance = float(BASE[candidate] > BASE[current])
            else:
                # The independence proposal gives q(candidate | current) = q(candidate).
                ratio = (BASE[candidate] * PROPOSAL[current]) / (
                    BASE[current] * PROPOSAL[candidate]
                )
                acceptance = min(1.0, ratio)
            row[candidate] = PROPOSAL[candidate] * acceptance
        # Rejected and self proposals leave the trace unchanged.
        row[current] = 1.0 - sum(row)
        rows.append(tuple(row))
    return tuple(rows)


def advance(distribution: Distribution, kernel: Matrix) -> Distribution:
    """Advance a distribution supported on VALID by one transition."""
    return tuple(
        sum(distribution[state] * row[destination] for state, row in zip(VALID, kernel))
        for destination in range(len(STATES))
    )


def kl_divergence(distribution: Distribution, reference: Distribution) -> float:
    return sum(
        probability * math.log(probability / target)
        for probability, target in zip(distribution, reference)
        if probability > 0.0
    )


def expected_log_probability(distribution: Distribution) -> float:
    return sum(
        probability * math.log(base_probability)
        for probability, base_probability in zip(distribution, BASE)
    )


def formatted(distribution: Distribution) -> str:
    return " ".join(f"{state}={probability:.4f}" for state, probability in zip(STATES, distribution))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=12, help="transitions to propagate (default: 12)")
    args = parser.parse_args()
    if args.steps < 0:
        parser.error("--steps must be nonnegative")

    target = condition(BASE, VALID)
    exact_kernel = transition_matrix(greedy=False)
    greedy_kernel = transition_matrix(greedy=True)
    exact = greedy = EXPERT

    print("Valid traces: A, B, C")
    print(f"Base p:      {formatted(BASE)}")
    print(f"Expert:      {formatted(EXPERT)}")
    print(f"Proposal q:  {formatted(PROPOSAL)}")
    print(f"Target p_C:  {formatted(target)}")
    print("\n          KL to p_C (nats)      E[log p(x)] (nats)")
    print(" step    exact MH    greedy     exact MH    greedy")
    for step in range(args.steps + 1):
        print(
            f"{step:5d}{kl_divergence(exact, target):12.4f}{kl_divergence(greedy, target):10.4f}"
            f"{expected_log_probability(exact):13.4f}{expected_log_probability(greedy):10.4f}"
        )
        if step < args.steps:
            exact = advance(exact, exact_kernel)
            greedy = advance(greedy, greedy_kernel)

    print(f"\nFinal exact MH: {formatted(exact)}")
    print(f"Final greedy:   {formatted(greedy)}")


if __name__ == "__main__":
    main()
