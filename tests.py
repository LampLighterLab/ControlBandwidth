import unittest
import environment
from value_estimator import MonteCarlo

class TestMonteCarlo(unittest.TestCase):
    
    def test_one(self):
        self.assertEqual(1, 1)