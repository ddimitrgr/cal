import unittest
from fractions import Fraction
from cal.poly import Poly


class TestPoly(unittest.TestCase):

    def test_poly_w_fraction_coeff(self):
        # Test notation
        P = Poly.create(Fraction)
        p1 = P([Fraction(1, 1), Fraction(3, 1), Fraction(4, 1)])
        p2 = P([Fraction(1, 1), Fraction(-3, 1), Fraction(3, 1)])
        self.assertTrue(type(p1.one) == P)
        self.assertTrue(type(P.one) == P)
        self.assertTrue(type(p1.zero) == P)
        self.assertTrue(type(P.zero) == P)

        p1.make_monic()
        divmod(p1, p2)
        p1 * p2
        p1 + p2 + p1*p2
        p1 - p2 + p1*p2
        p2*Fraction(1, 5)
        Fraction(1, 3)*p2
        p2**5 + 2*p1 +p1*3
        p2 - 3*p2.zero - 5*p2.one
        p2 + 4*P.zero + P.zero*p1
        p2 + 4 * P.one + P.one * p1
        -p2
        (-5)*p1
        self.assertEqual((-5)*p1, -(5*p1))

