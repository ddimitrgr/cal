import unittest
from cal.base import IntegerQuotientRing


class TestRing(unittest.TestCase):

    def test_integer_quotient_ring(self):
        # Test notation
        R = IntegerQuotientRing.create(5**2)
        R10, R22 = R(10), R(22)
        R10 * R.one
        R10.one * R10
        R10 * R.zero
        R10.zero * R10
        2 * R22 * R10 * 3
        R10 ** 2 - R22
        -R22
