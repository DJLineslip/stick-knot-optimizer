"""Known public Wirt_Hm row, independently reconstructed as a Wirtinger map."""
import ast
import unittest

from equistick.coxeter import _d4_matrix, verify_map


S5_GAUSS = str([1, -2, 3, -1, -4, -5, 6, 7, 2, -3, -8, 9, -10, 4,
                11, -6, -12, 13, 14, -15, 5, -11, -7, 12, -13, 8,
                -9, -14, 15, 10])
S5_SEEDS = str({'(-12, 13, 14, -15)': '(2, 4)',
                '(-5, 6, 7, 2, -3)': '(3, 5)',
                '(-7, 12, -13)': '(1, 2)',
                '(-2, 3, -1)': '(1, 3)'})
D4_GAUSS = str([1, -2, 3, -1, -4, -5, 6, 7, 2, -3, -8, 9, -10, 4,
                11, -6, -12, 8, -9, -13, 14, 12, -7, -11, 5, -15,
                13, -14, 15, 10])
D4_SEEDS = str({
    '(-9, -13)': '(0, -1, 1, 1)|(-1, 0, 1, 1)|(0, 0, 1, 0)|(0, 0, 0, 1)',
    '(-2, 3, -1)': '(-1, 1, 0, 0)|(0, 1, 0, 0)|(0, 0, 1, 0)|(0, 0, 0, 1)',
    '(-11, 5, -15)': '(0, 0, -1, 1)|(-1, 1, -1, 1)|(-1, 0, 0, 1)|(0, 0, 0, 1)',
    '(-8, 9, -10)': '(1, -1, 0, 0)|(0, -1, 0, 0)|(0, -1, 1, 0)|(0, -1, 0, 1)'})


class CoxeterMapTests(unittest.TestCase):
    def test_d4_involution_that_is_not_a_root_reflection_is_rejected(self):
        minus_identity = '(-1, 0, 0, 0)|(0, -1, 0, 0)|(0, 0, -1, 0)|(0, 0, 0, -1)'
        with self.assertRaisesRegex(ValueError, 'root reflection'):
            _d4_matrix(minus_identity)

    def test_s5_map_propagates_all_strands_checks_relations_and_generates(self):
        proof = verify_map(S5_GAUSS, S5_SEEDS, 'S5')
        self.assertTrue(proof['passed'], proof)
        self.assertEqual(proof['strand_count'], 15)
        self.assertEqual(len(proof['strand_images']), 15)
        self.assertEqual(proof['relations_checked'], 15)
        self.assertEqual(proof['group_order'], 120)

    def test_d4_map_propagates_and_generates_canonical_weyl_group(self):
        proof = verify_map(D4_GAUSS, D4_SEEDS, 'D4')
        self.assertTrue(proof['passed'], proof)
        self.assertEqual(proof['strand_count'], 15)
        self.assertEqual(len(proof['strand_images']), 15)
        self.assertEqual(proof['relations_checked'], 15)
        self.assertEqual(proof['group_order'], 192)

    def test_relations_alone_do_not_prove_surjectivity(self):
        seeds = {strand: '(1, 2)' for strand in ast.literal_eval(S5_SEEDS)}
        proof = verify_map(S5_GAUSS, str(seeds), 'S5')
        self.assertFalse(proof['passed'])
        self.assertEqual(proof['relations_checked'], 15)
        self.assertEqual(proof['failed_relations'], [])
        self.assertEqual(proof['group_order'], 2)
        self.assertEqual(proof['status'], 'not_surjective')
        self.assertFalse(proof['generation_passed'])


if __name__ == '__main__':
    unittest.main()
