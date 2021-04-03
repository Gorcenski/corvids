import unittest

from RecreateData import RecreateData, multiprocessGetManipBases, multiprocessGetSolutionSpace

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
    
    # # def test_integration_with_check_val(self):
    # #     expected_key = (3.0, 2.0)
    # #     expected_result_first = [1, 2, 4, 4, 4]
    # #     expected_result_second = [2, 2, 2, 4, 5]
    # #     self.rd = RecreateData(min_score=1, max_score=5, num_samples=5, mean=3.0, variance=2.0)
    # #     self.rd.recreateData(check_val=1)
    # #     self.assertIn(expected_result_first, self.rd.getDataSimple()[expected_key])
    # #     self.assertNotIn(expected_result_second, self.rd.getDataSimple()[expected_key])
    
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

    def test_compute_valid_means_variances(self):
        self.rd = RecreateData(min_score=1, max_score=5, num_samples=5, mean=3.0, variance=2.0)
        self.rd.poss_vals = range(self.rd.absolute_min, self.rd.absolute_max+1)
        expected_result = [(3.0, 2.0)]
        self.assertEqual(expected_result, self.rd.compute_valid_means_variances(False))

    def test_compute_valid_means_variances_with_precision(self):
        self.rd = RecreateData(min_score=1, max_score=5, num_samples=50, mean=3.0, variance=2.0, mean_precision=0.1, variance_precision=0.01)
        self.rd.poss_vals = range(self.rd.absolute_min, self.rd.absolute_max+1)
        expected_result = (3.0, 2.0)
        self.assertIn(expected_result, self.rd.compute_valid_means_variances(False))

    def test_compute_valid_means_variances_with_precision_and_find_first_true_with_none_result(self):
        self.rd = RecreateData(min_score=1, max_score=50, num_samples=50, mean=3.0, variance=2.5, mean_precision=0.1, variance_precision=0.01)
        self.rd.poss_vals = range(self.rd.absolute_min, self.rd.absolute_max+1)
        self.assertIsNone(self.rd.compute_valid_means_variances(True))

    def test_compute_initial_adjusted_var_from_mean(self):
        self.rd = RecreateData(min_score=1, max_score=50, num_samples=50, mean=3.0, variance=2.5, mean_precision=0.1, variance_precision=0.01)
        expected_input = 2.98
        expected_result = 2450.0
        self.assertEquals(expected_result, self.rd._initial_adjusted_var_from_mean(expected_input))

    def test_compute_initial_mean_valid_from_mean(self):
        self.rd = RecreateData(min_score=1, max_score=50, num_samples=50, mean=3.0, variance=2.5, mean_precision=0.1, variance_precision=0.01)
        expected_input = 2.98
        expected_result = [2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]
        self.assertEquals(expected_result, self.rd._initial_mean_valid_from_mean(expected_input))

    def test_integration_recreate_data_with_find_first_true_with_none_result(self):
        self.rd = RecreateData(min_score=1, max_score=50, num_samples=50, mean=3.0, variance=2.5, mean_precision=0.1, variance_precision=0.01)
        self.assertIsNone(self.rd.recreateData(find_first=True))

    def test_integration_get_data_simple_should_return_valueerror_if_no_data(self):
        self.assertRaises(ValueError, self.rd.getDataSimple)
