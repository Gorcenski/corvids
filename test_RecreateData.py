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

    def test_integration_get_data_simple_should_return_valueerror_if_no_data(self):
        self.assertRaises(ValueError, self.rd.getDataSimple)
