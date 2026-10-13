"""Regression checks for image pairing and exact McNemar edge cases."""
import unittest
import pandas as pd
from statsmodels.stats.contingency_tables import mcnemar
from src.analyze import paired_correctness


class PairingTests(unittest.TestCase):
    def setUp(self):
        self.a = pd.DataFrame({'image_id': ['a', 'b', 'c', 'd'],
                              'true': [0, 1, 2, 3], 'pred': [0, 1, 0, 0]})
        self.b = pd.DataFrame({'image_id': ['a', 'b', 'c', 'd'],
                              'true': [0, 1, 2, 3], 'pred': [0, 0, 2, 0]})

    def test_order_independent_pairing(self):
        self.assertEqual(paired_correctness(self.a, self.b.iloc[::-1]), [[1, 1], [1, 1]])

    def test_no_discordant_pairs(self):
        self.assertEqual(mcnemar(paired_correctness(self.a, self.a), exact=True).pvalue, 1)

    def test_exact_binomial(self):
        self.assertAlmostEqual(mcnemar([[0, 8], [0, 0]], exact=True).pvalue, 2 / 256)

    def test_reject_missing_and_duplicate_ids(self):
        for b in [self.b.iloc[:-1], pd.concat([self.b, self.b.iloc[:1]])]:
            with self.assertRaises(ValueError):
                paired_correctness(self.a, b)

    def test_reject_label_mismatch(self):
        self.b.loc[0, 'true'] = 1
        with self.assertRaises(ValueError):
            paired_correctness(self.a, self.b)


if __name__ == '__main__':
    unittest.main()
