import math
import unittest

class TestFactorial(unittest.TestCase):

    def test_positive(self):
        self.assertEqual(math.factorial(5),120)
    
    def test_zero(self):
        self.assertEqual(math.factorial())
    
