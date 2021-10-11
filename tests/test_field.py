import unittest
from typing import List
from fractions import Fraction
from cal.base import Q
from cal.poly import Poly
from cal.field import PolyField, PrimeField


class TestField(unittest.TestCase):

    def test_prime_field(self):
        F = PrimeField.create(37)
        p1, p2 = F(20), F(31)
        x, xi = (p2 * p2 + p1 ** 3) ** 5, (p2 * p2 + p1 ** 3) ** -5
        self.assertEqual(x*xi, F.one)

    def test_q_class(self):
        f1 = Q(3, 4)
        f2 = Q(Fraction(5, 2))
        self.assertTrue(isinstance(Q.zero, Q))
        self.assertTrue(isinstance(Q.one, Q))
        self.assertTrue(isinstance(f1.one, Q))
        self.assertTrue(isinstance(f2.zero, Q))
        self.assertEqual(f1+f2, Q(13, 4))
        self.assertEqual(f1*f2, Q(15, 8))
        self.assertEqual(
            f1*f2 - f1 + f2 + Q.one + 2*f1.one,
            Q(Fraction(15, 8) - Fraction(3, 4) + Fraction(5, 2) + 3)
        )

    def test_poly_field_w_fraction_coeff(self):
        # Test notation
        R = Poly.create()
        _p2: List[Fraction] = [Fraction(1, 1), Fraction(1, 1), Fraction(1, 1), Fraction(1, 1)]
        r2 = R(_p2)
        F1 = PolyField.create(r2)

        f2 = F1([Fraction(11, 2), Fraction(1, 3), Fraction(5, 2), Fraction(1, 2)])
        g2 = F1([Fraction(1, 7), Fraction(2, 5), Fraction(1, 6), Fraction(7, 5), Fraction(2, 1)])

        self.assertTrue(isinstance(f2.one, F1))
        self.assertTrue(isinstance(f2.zero, F1))
        self.assertTrue(isinstance(F1.one, F1))
        self.assertTrue(isinstance(F1.zero, F1))

        4 * g2 + f2* 3 + f2*Fraction(3, 4) + Fraction(1, 2)*g2
        f2i = f2 ** -1
        self.assertEqual(f2*f2i, f2.one)
