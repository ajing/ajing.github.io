import math
import unittest

from experiment import (cll, entropy, frontier, make_data, make_examples,
                        correct_prob, sample_weight, sft_gradient, reward_gradient)
from llm_adapter import build_weights, validate_record


class NumericalChecks(unittest.TestCase):
    def test_cll_centering_and_external_distribution(self):
        p, q = [0.8, 0.15, 0.05], [0.2, 0.3, 0.5]
        self.assertAlmostEqual(sum(pi*cll(p, i) for i, pi in enumerate(p)), 0)
        kl = sum(qi*math.log(qi/pi) for qi, pi in zip(q, p))
        self.assertAlmostEqual(sum(qi*cll(p, i) for i, qi in enumerate(q)),
                               entropy(p)-entropy(q)-kl)
        self.assertGreater(cll(p, 0), 0)  # q=delta on the modal action is off-policy.

    def test_uniform_distribution_is_blind(self):
        for i in range(3):
            self.assertAlmostEqual(cll([1/3]*3, i), 0)

    def test_prefix_reachability(self):
        a, b = make_examples()['paths']
        self.assertEqual(a['final_action_cll'], b['final_action_cll'])
        self.assertAlmostEqual(a['prefix_probability']/b['prefix_probability'], 40)

    def test_weight_floor_and_peak(self):
        self.assertEqual(frontier(-0.6), 1)
        self.assertAlmostEqual(frontier(-100), 0.1)

    def test_normalized_detached_sft_gradient(self):
        batch = make_data(42, 16)
        w = [0.3, -0.1, 0.2]
        eps = 1e-5
        for method in ['uniform', 'probability', 'cll_monotone', 'cll_frontier']:
            raw = [sample_weight(r, method) for r in batch]
            def loss(params):
                return -sum(a*math.log(correct_prob(params, r)) for a, r in zip(raw, batch))/sum(raw)
            analytical, ess = sft_gradient(w, batch, method)
            self.assertGreaterEqual(ess, 1-1e-10)
            self.assertLessEqual(ess, len(batch)+1e-10)
            for j in range(3):
                plus, minus = w[:], w[:]
                plus[j] += eps
                minus[j] -= eps
                numerical = (loss(plus)-loss(minus))/(2*eps)
                self.assertAlmostEqual(analytical[j], numerical, places=8)

    def test_exact_reward_gradient(self):
        batch = make_data(123, 16)
        w, eps = [0.2, 0.4, -0.1], 1e-5
        analytical = reward_gradient(w, batch)
        for j in range(3):
            plus, minus = w[:], w[:]
            plus[j] += eps
            minus[j] -= eps
            numerical = sum(correct_prob(plus,r)-correct_prob(minus,r) for r in batch)/(2*eps*len(batch))
            self.assertAlmostEqual(analytical[j], numerical, places=8)

    def test_adapter_weights_preserve_mask_and_step_alignment(self):
        row = dict(input_ids=[10, 11, 12, 13, 14], loss_mask=[0, 0, 1, 0, 1],
                   step_ids=[-1, -1, 0, -1, 0], quality=1,
                   tokens=[dict(position=2, step_id=0, logp=-2, cll=-1),
                           dict(position=4, step_id=0, logp=-3, cll=-2)])
        validate_record(row)
        weights = build_weights(row)
        self.assertEqual(weights[0], 0)
        self.assertEqual(weights[2], 0)
        self.assertEqual(weights[1], weights[3])
        self.assertAlmostEqual(weights[1], frontier(-1.5, -1.3, 0.6, 0.1))
        row['quality'] = 0
        self.assertEqual(build_weights(row), [0]*4)

    def test_sequence_weights_are_not_normalized_away(self):
        def record(z):
            return dict(input_ids=[0, 1, 2], tokens=[
                dict(position=1, step_id=0, cll=z, logp=-2),
                dict(position=2, step_id=1, cll=z, logp=-2)])
        near = build_weights(record(-1.3), level='sequence')
        far = build_weights(record(-10), level='sequence')
        self.assertGreater(sum(near), sum(far)*9)


if __name__ == '__main__':
    unittest.main()
