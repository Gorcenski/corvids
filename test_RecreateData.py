import unittest
from sympy import Matrix

from RecreateData import RecreateData, multiprocessGetManipBases, multiprocess_get_solution_space

class TestRecreateData(unittest.TestCase):
    def setUp(self):
        self.rd = RecreateData(min_score=1, max_score=5, num_samples=5, mean=3.0, variance=2.5)

    def test_integration_recreate_data(self):
        expected_result = {(3.0, 2.5): [[1, 1, 1, 1, 1]]}
        self.assertDictEqual(expected_result, self.rd.recreateData())

    def test_integration_get_data_simple(self):
        expected_key = (3.0, 2.5)
        expected_result = [[1, 2, 3, 4, 5]]
        self.rd.recreateData()
        self.assertEqual(expected_result, self.rd.getDataSimple()[expected_key])
   
    def test_integration_get_data_simple_with_multiple_results(self):
        expected_key = (3.0, 2.0)
        expected_result_first = [1, 2, 4, 4, 4]
        expected_result_second = [2, 2, 2, 4, 5]
        self.rd = RecreateData(min_score=1, max_score=5, num_samples=5, mean=3.0, variance=2.0)
        self.rd.recreateData()
        self.assertIn(expected_result_first, self.rd.getDataSimple()[expected_key])
        self.assertIn(expected_result_second, self.rd.getDataSimple()[expected_key])
    
    def test_integration_with_no_multiprocess(self):
        expected_key = (3.0, 2.5)
        expected_result_first = [1, 2, 3, 4, 5]
        expected_result_second = [2, 2, 2, 4, 5]
        self.rd.recreateData(multiprocess=False)
        self.assertIn(expected_result_first, self.rd.getDataSimple()[expected_key])
        self.assertNotIn(expected_result_second, self.rd.getDataSimple()[expected_key])
    
    # # Eventually we only need this one
    # # def test_integration_with_find_first(self):
    # #     expected_key = (3.0, 2.5)
    # #     expected_result_first = [1, 2, 3, 4, 5]
    # #     expected_result_second = [2, 2, 2, 4, 5]
    # #     self.rd.recreateData(find_first=True)
    # #     self.assertIn(expected_result_first, self.rd.getDataSimple()[expected_key])
    # #     self.assertNotIn(expected_result_second, self.rd.getDataSimple()[expected_key])
    
    # def test_integration_with_check_val(self):
    #     expected_key = (3.0, 2.0)
    #     expected_result_first = [1, 2, 4, 4, 4]
    #     expected_result_second = [2, 2, 2, 4, 5]
    #     self.rd = RecreateData(min_score=1, max_score=5, num_samples=5, mean=3.0, variance=2.0)
    #     self.rd.recreateData(check_val=1)
    #     self.assertIn(expected_result_first, self.rd.getDataSimple()[expected_key])
    #     self.assertNotIn(expected_result_second, self.rd.getDataSimple()[expected_key])
    
    def test_integration_with_poss_vals(self):
        expected_key = (3.0, 2.0)
        expected_result_first = [1, 2, 4, 4, 4]
        expected_result_second = [2, 2, 2, 4, 5]
        self.rd = RecreateData(min_score=1, max_score=5, num_samples=5, mean=3.0, variance=2.0)
        self.rd.recreateData(poss_vals=[2, 3, 4, 5])
        self.assertNotIn(expected_result_first, self.rd.getDataSimple()[expected_key])
        self.assertIn(expected_result_second, self.rd.getDataSimple()[expected_key])

    # def test_integration_with_check_val_and_find_first(self):
    #     expected_key = (3.0, 2.0)
    #     expected_result_first = [1, 2, 4, 4, 4]
    #     expected_result_second = [2, 2, 2, 4, 5]
    #     self.rd = RecreateData(min_score=1, max_score=5, num_samples=5, mean=3.0, variance=2.0)
    #     self.rd.recreateData(check_val=5, find_first=True)
    #     self.assertNotIn(expected_result_first, self.rd.getDataSimple()[expected_key])
    #     self.assertIn(expected_result_second, self.rd.getDataSimple()[expected_key])

    def test_integration_with_check_val_and_poss_vals(self):
        expected_key = (3.0, 2.0)
        expected_result_first = [1, 2, 4, 4, 4]
        expected_result_second = [2, 2, 2, 4, 5]
        self.rd = RecreateData(min_score=1, max_score=5, num_samples=5, mean=3.0, variance=2.0)
        self.rd.recreateData(check_val=5, poss_vals=[2, 3, 4, 5])
        self.assertNotIn(expected_result_first, self.rd.getDataSimple()[expected_key])
        self.assertIn(expected_result_second, self.rd.getDataSimple()[expected_key])

    # _compute_valid_means_variances
    def test__compute_valid_means_variances(self):
        self.rd = RecreateData(min_score=1, max_score=5, num_samples=5, mean=3.0, variance=2.0)
        self.rd.poss_vals = range(self.rd.absolute_min, self.rd.absolute_max+1)
        expected_result = [(3.0, 2.0)]
        self.assertEqual(expected_result, self.rd._compute_valid_means_variances(False))

    def test__compute_valid_means_variances_with_precision(self):
        self.rd = RecreateData(min_score=1, max_score=5, num_samples=50, mean=3.0, variance=2.0, mean_precision=0.1, variance_precision=0.01)
        self.rd.poss_vals = range(self.rd.absolute_min, self.rd.absolute_max+1)
        expected_result = (3.0, 2.0)
        self.assertIn(expected_result, self.rd._compute_valid_means_variances(False))

    def test__compute_valid_means_variances_with_precision_and_find_first_true_with_none_result(self):
        self.rd = RecreateData(min_score=1, max_score=50, num_samples=50, mean=3.0, variance=2.5, mean_precision=0.1, variance_precision=0.01)
        self.rd.poss_vals = range(self.rd.absolute_min, self.rd.absolute_max+1)
        self.assertIsNone(self.rd._compute_valid_means_variances(True))

    #_initial_adjusted_var_from_mean
    def test_compute_initial_adjusted_var_from_mean(self):
        self.rd = RecreateData(min_score=1, max_score=50, num_samples=50, mean=3.0, variance=2.5, mean_precision=0.1, variance_precision=0.01)
        expected_input = 2.98
        expected_result = 2450.0
        self.assertEquals(expected_result, self.rd._initial_adjusted_var_from_mean(expected_input))

    # _initial_mean_valid_from_mean
    def test_compute_initial_mean_valid_from_mean(self):
        self.rd = RecreateData(min_score=1, max_score=50, num_samples=50, mean=3.0, variance=2.5, mean_precision=0.1, variance_precision=0.01)
        expected_input = 2.98
        expected_result = [2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]
        self.assertEquals(expected_result, self.rd._initial_mean_valid_from_mean(expected_input))

    # initialization
    def test_poss_vals_autoset_on_init(self):
        self.rd = RecreateData(min_score=1, max_score=5, num_samples=50, mean=3.0, variance=2.0, mean_precision=0.1, variance_precision=0.01)
        expected_poss_vals = [1, 2, 3, 4, 5]
        self.assertEquals(expected_poss_vals, self.rd.poss_vals)

    # set_ranges_with_possible_values
    def test_set_ranges_with_possible_values(self):
        self.rd = RecreateData(min_score=1, max_score=5, num_samples=50, mean=3.0, variance=2.0, mean_precision=0.1, variance_precision=0.01)
        expected_poss_vals = [1, 2, 3, 4]
        self.rd.set_ranges_with_possible_values(expected_poss_vals)
        self.assertEquals(expected_poss_vals, self.rd.poss_vals)

    # _build_solution_space
    def test__build_solution_space_with_multiprocess(self):
        self.rd = RecreateData(min_score=1, max_score=5, num_samples=5, mean=3.0, variance=2.0)
        mvp = self.rd._compute_valid_means_variances(False)
        expected_result = [(Matrix([[1, 1, 0, 3, 0]]), 
                            Matrix([[-1, 3, -3,  1, 0],
                                    [-1, 2,  0, -2, 1]]),
                            Matrix([[  1,  1, 1,  1,   1],
                                    [  1,  2, 3,  4,   5],
                                    [100, 25, 0, 25, 100]]),
                            Matrix([[  5],
                                    [ 15],
                                    [200]]),
                            (3.0, 2.0))]
        self.assertEquals(expected_result, self.rd._build_solution_space(mvp, multiprocess=True))

    def test__build_solution_space_without_multiprocess(self):
        self.rd = RecreateData(min_score=1, max_score=5, num_samples=5, mean=3.0, variance=2.0)
        mvp = self.rd._compute_valid_means_variances(False)
        expected_result = [(Matrix([[1, 1, 0, 3, 0]]), 
                            Matrix([[-1, 3, -3,  1, 0],
                                    [-1, 2,  0, -2, 1]]),
                            Matrix([[  1,  1, 1,  1,   1],
                                    [  1,  2, 3,  4,   5],
                                    [100, 25, 0, 25, 100]]),
                            Matrix([[  5],
                                    [ 15],
                                    [200]]),
                            (3.0, 2.0))]
        self.assertEquals(expected_result, self.rd._build_solution_space(mvp, multiprocess=False))

    def test__build_solution_space_should_be_identical_with_or_without_multiprocess(self):
        rd1 = RecreateData(min_score=1, max_score=5, num_samples=5, mean=3.0, variance=2.0)
        rd2 = RecreateData(min_score=1, max_score=5, num_samples=5, mean=3.0, variance=2.0)
        mvp1 = rd1._compute_valid_means_variances(False)
        mvp2 = rd2._compute_valid_means_variances(False)
        self.assertEquals(rd1._build_solution_space(mvp1, multiprocess=True), rd2._build_solution_space(mvp2, multiprocess=False))

    # multiprocess_get_solution_space
    def test_multiprocess_get_solution_space(self):
        expected_result = (Matrix([[1, 1, 0, 3, 0]]), 
                           Matrix([[-1, 3, -3,  1, 0],
                                   [-1, 2,  0, -2, 1]]),
                           Matrix([[  1,  1, 1,  1,   1],
                                   [  1,  2, 3,  4,   5],
                                   [100, 25, 0, 25, 100]]),
                           Matrix([[  5],
                                   [ 15],
                                   [200]]),
                           (3.0, 2.0))
        self.assertEquals(expected_result, multiprocess_get_solution_space(1, 5, 5, (3.0, 2.0)))

    def test_multiprocess_get_solution_space_with_int_check_val(self):
        expected_result = (Matrix([[0, 3, 0, 1, 0]]), 
                           Matrix([[-1, 3, -3,  1, 0],
                                   [-1, 2,  0, -2, 1]]),
                           Matrix([[  1,  1, 1,  1,   1],
                                   [  1,  2, 3,  4,   5],
                                   [100, 25, 0, 25, 100]]),
                           Matrix([[  4],
                                   [ 10],
                                   [100]]),
                           (3.0, 2.0))
        self.assertEquals(expected_result, multiprocess_get_solution_space(1, 5, 5, (3.0, 2.0), check_val=5))

    def test_multiprocess_get_solution_space_with_single_list_check_val_same_as_scalar(self):
        self.assertEquals(multiprocess_get_solution_space(1, 5, 5, (3.0, 2.0), check_val=5),
                          multiprocess_get_solution_space(1, 5, 5, (3.0, 2.0), check_val=[5]))

    def test_multiprocess_get_solution_space_with_single_dict_check_val_same_as_scalar(self):
        self.assertEquals(multiprocess_get_solution_space(1, 5, 5, (3.0, 2.0), check_val=5),
                          multiprocess_get_solution_space(1, 5, 5, (3.0, 2.0), check_val={5: 1}))

    def test_multiprocess_get_solution_space_with_multiple_list_check_val(self):
        self.assertIsNone(multiprocess_get_solution_space(1, 5, 100, (2.3, 0.45), check_val=[1,5]))

    def test_multiprocess_get_solution_space_with_multiple_dict_check_val(self):
        self.assertIsNone(multiprocess_get_solution_space(1, 5, 100, (2.3, 0.45), check_val={5: 1, 1: 2}))

    def test_multiprocess_get_solution_space_with_poss_vals(self):
        expected_result = (Matrix([[1, 1, 0, 3, 0]]), 
                           Matrix([[-1, 3, -3,  1, 0],
                                   [-1, 2,  0, -2, 1]]),
                           Matrix([[  1,  1, 1,  1,   1],
                                   [  1,  2, 3,  4,   5],
                                   [100, 25, 0, 25, 100]]),
                           Matrix([[  5],
                                   [ 15],
                                   [200]]),
                           (3.0, 2.0))
        self.assertEquals(expected_result, multiprocess_get_solution_space(1, 5, 5, (3.0, 2.0), poss_vals=[1,2,3,4,5]))

    def test_multiprocess_get_solution_space_with_poss_vals_and_int_check_val(self):
        expected_result = (Matrix([[0, 3, 0, 1, 0]]), 
                           Matrix([[-1, 3, -3,  1, 0],
                                   [-1, 2,  0, -2, 1]]),
                           Matrix([[  1,  1, 1,  1,   1],
                                   [  1,  2, 3,  4,   5],
                                   [100, 25, 0, 25, 100]]),
                           Matrix([[  4],
                                   [ 10],
                                   [100]]),
                           (3.0, 2.0))
        self.assertEquals(expected_result, multiprocess_get_solution_space(1, 5, 5, (3.0, 2.0), check_val=5, poss_vals=[1,2,3,4,5]))

    def test_multiprocess_get_solution_space_with_poss_vals_and_list_check_val(self):
        self.assertIsNone(multiprocess_get_solution_space(1, 5, 5, (3.0, 2.0), check_val=[4,5], poss_vals=[1,2,3,4,5]))

    # _find_first_solution
    def test__find_first_solution_without_check_val_or_poss_vals(self):
        find_first = True
        mean_var_pairs = self.rd._compute_valid_means_variances(find_first)
        solution_spaces = self.rd._build_solution_space(mean_var_pairs)
        expected_result = {'_': [[1, 1, 1, 1, 1]]}
        self.assertDictEqual(expected_result, self.rd._find_first_solution(solution_spaces))
    
    def test__find_first_solution_independent_of_multiprocess(self):
        find_first = True
        mean_var_pairs = self.rd._compute_valid_means_variances(find_first)
        solution_spaces = self.rd._build_solution_space(mean_var_pairs)
        self.assertDictEqual(self.rd._find_first_solution(solution_spaces, multiprocess=True), 
                             self.rd._find_first_solution(solution_spaces, multiprocess=False))

    def test__find_first_solution_without_check_val_but_with_poss_vals(self):
        find_first = True
        poss_vals = [1, 2, 3, 4, 5]
        mean_var_pairs = self.rd._compute_valid_means_variances(find_first)
        solution_spaces = self.rd._build_solution_space(mean_var_pairs, poss_vals=poss_vals)
        expected_result = {'_': [[1, 1, 1, 1, 1]]}
        self.assertDictEqual(expected_result, self.rd._find_first_solution(solution_spaces))

    def test__find_first_solution_with_check_val_but_without_poss_vals(self):
        find_first = True
        check_val = [4, 5]
        mean_var_pairs = self.rd._compute_valid_means_variances(find_first)
        solution_spaces = self.rd._build_solution_space(mean_var_pairs, check_val=check_val)
        expected_result = {'_': [[1, 1, 1, 1, 1]]}
        self.assertIsNone(self.rd._find_first_solution(solution_spaces, check_val=check_val))

    def test__find_first_solution_with_check_val_and_poss_vals(self):
        find_first = True
        poss_vals = [1, 2, 3, 4, 5]
        check_val = [4, 5]
        mean_var_pairs = self.rd._compute_valid_means_variances(find_first)
        solution_spaces = self.rd._build_solution_space(mean_var_pairs, check_val=check_val, poss_vals=poss_vals)
        expected_result = {'_': [[1, 1, 1, 1, 1]]}
        self.assertIsNone(self.rd._find_first_solution(solution_spaces, check_val=check_val, poss_vals=poss_vals))

    # Refactoring checks
    def test_reduce_lambda_scalar(self):
        check_val = 5
        variance = 2
        mean = 3
        num_samples = 5
        check_vals = [(check_val, 1)]
        x = reduce(lambda x, y: [x[0] - y[1],
                                 x[1] - y[0] * y[1],
                                 x[2] - y[1] * (x[0] * y[0] - x[1]) ** 2],
                   check_vals,
                   [num_samples, mean, variance])
        variance -= (num_samples * check_val - mean) ** 2
        mean -= check_val
        num_samples -= 1
        self.assertEquals(x, [num_samples, mean, variance])

    def test_reduce_lambda_list(self):
        check_vals = [4, 5]
        variance = 2
        mean = 3
        num_samples = 5
        x = reduce(lambda x, y: [x[0] - y[1],
                                 x[1] - y[0] * y[1],
                                 x[2] - y[1] * (x[0] * y[0] - x[1]) ** 2],
                   zip(check_vals, [1] * len(check_vals)),
                   [num_samples, mean, variance])
        for val in check_vals:
            variance -= (num_samples * val - mean) ** 2
            mean -= val
            num_samples -= 1
        self.assertEquals(x, [num_samples, mean, variance])

    def test_reduce_lambda_dict(self):
        check_vals = {4: 1, 5: 2}
        variance = 2
        mean = 3
        num_samples = 5
        x = reduce(lambda x, y: [x[0] - y[1],
                                 x[1] - y[0] * y[1],
                                 x[2] - y[1] * (x[0] * y[0] - x[1]) ** 2],
                   [(a, b) for a, b in check_vals.iteritems()],
                   [num_samples, mean, variance])
        for val, num in check_vals.iteritems():
            variance -= num * (num_samples * val - mean) ** 2
            mean -= num * val
            num_samples -= num
        self.assertEquals(x, [num_samples, mean, variance])

    def test_integration_recreate_data_with_find_first_true_with_none_result(self):
        self.rd = RecreateData(min_score=1, max_score=50, num_samples=50, mean=3.0, variance=2.5, mean_precision=0.1, variance_precision=0.01)
        self.assertIsNone(self.rd.recreateData(find_first=True))

    def test_integration_get_data_simple_should_return_valueerror_if_no_data(self):
        self.assertRaises(ValueError, self.rd.getDataSimple)
