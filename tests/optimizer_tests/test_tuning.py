import torch
import unittest

from megatron.training import optimizer_overlap_tuning_suggest_ratio


class TestTuning(unittest.TestCase):
    def test_suggestion(self):
        current_ratio = .100000
        target_time_ratio = .85
        time_pairs_all = torch.tensor([
            (10., 30.),
            (0., 0.),
            (0., 0.),
            (12., 32.),
            (12., 32.),
            (10., 32.),
        ], device="cuda")

        PP = 4
        capacity_ratio, suggested_ratio, msg = \
            optimizer_overlap_tuning_suggest_ratio(current_ratio, target_time_ratio, time_pairs_all, PP)
        self.assertAlmostEqual(capacity_ratio, .212500)  # 12ms = 10%   -->   30ms * .85 = 21.25%
        self.assertAlmostEqual(suggested_ratio, .212500)

        PP = 8
        capacity_ratio, suggested_ratio, msg = \
            optimizer_overlap_tuning_suggest_ratio(current_ratio, target_time_ratio, time_pairs_all, PP)
        self.assertAlmostEqual(capacity_ratio, .212500)
        self.assertAlmostEqual(suggested_ratio, .200000)

        time_pairs_all.zero_()
        capacity_ratio, suggested_ratio, msg = \
            optimizer_overlap_tuning_suggest_ratio(current_ratio, target_time_ratio, time_pairs_all, PP)
        self.assertIs(capacity_ratio, None)
        self.assertIs(suggested_ratio, None)


if __name__ == "__main__":
    unittest.main()
