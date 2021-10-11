import unittest
from fractions import Fraction
from cal.field import PrimeField
from cal.poly import Poly
from cal.cyclo import calc_cyclotomic_field
from cal.factor import square_free_poly_decomposition_in_q, \
    square_free_poly_decomposition_in_finite_field, \
    factor_poly_in_prime_field, \
    factor, \
    square_root_of_poly_in_q


class TestFactorization(unittest.TestCase):

    def test_square_root_of_poly_in_q(self):
        P = Poly.create()
        f1 = P([Fraction(1, 1), Fraction(1, 1), Fraction(4, 1)])
        f2 = P([Fraction(-1, 1), Fraction(1,1)])
        p1 = f1 * f2**2
        r1 = square_root_of_poly_in_q(p1)
        p2 = f1**4 * f2**2
        r2 = square_root_of_poly_in_q(p2)
        self.assertIsNone(r1)
        self.assertEqual(r2, f1**2 * f2)

    def test_square_free_decomp_in_q(self):
        P1 = Poly.create(Fraction)
        f1 = P1([Fraction(-3, 2), Fraction(1, 1)])
        f2 = P1([Fraction(1, 1), Fraction(1, 1), Fraction(1, 1)])
        f3 = P1([Fraction(1, 2), Fraction(0, 1), Fraction(1, 1)])
        f4 = P1([Fraction(0, 1), Fraction(1, 1)])
        f5 = P1([Fraction(-4, 1), Fraction(0, 1), Fraction(1, 1)])

        p0 = f1 ** 3 * f2 * f3 ** 2
        answer = {f1: 3, f2: 1, f3: 2}
        fa = square_free_poly_decomposition_in_q(p0)
        self.assertEqual(fa, answer)

        p1 = f4 * f5 ** 2
        answer = {f4: 1, f5: 2}
        fa = square_free_poly_decomposition_in_q(p1)
        self.assertEqual(fa, answer)

    def test_square_free_comp_in_finite_field(self):
        p = 5
        F = PrimeField.create(p=p)
        P = Poly.create(domain=F)
        f1 = P([F(2), F(4), F(1)])
        f2 = P([F(1), F(1), F(1)])
        f3 = P([F(4), F.zero, F(1)])
        answer = {f1: 5, f2: 2, f3: 1}
        po = f1**5 * f2**2 * f3
        f_map = square_free_poly_decomposition_in_finite_field(po)
        self.assertEqual(f_map, answer)

    def test_factorization_in_finite_field(self):
        # Test using cyclotomic polynomials Qn(x) that are known to be irreducible.
        n1, n2, n3, p = 4, 5, 10, 7
        P1, _ = calc_cyclotomic_field(n=n1, p=p)
        P2, _ = calc_cyclotomic_field(n=n2, p=p)
        P3, _ = calc_cyclotomic_field(n=n3, p=p)
        f1, f2, f3 = P1.irreducible_poly, P2.irreducible_poly, P3.irreducible_poly
        F = P1.coeff_type
        # We need type casting !
        f2 = type(f1)([F(c.value) for c in f2.coefficient_list])
        f3 = type(f1)([F(c.value) for c in f3.coefficient_list])
        f4 = type(f1)([F(3), F.one])
        f5 = type(f1)([F(4), F.one])

        #  Factoring with unit multiplicities in F7: Q4(x)*Q5(x)*Q10(x)
        factor_to_mult_map = factor_poly_in_prime_field(f1*f2*f3)
        for fa, mu in ((f1, 1), (f2, 1), (f3, 1)):
            self.assertEqual(factor_to_mult_map[fa], mu)
            del factor_to_mult_map[fa]
        self.assertFalse(factor_to_mult_map)

        # Full factoring in F7: : Q4(x)**2 * Q5(x) * (x+3)**2 * (x+4)**2
        factor_to_mult_map = factor(f1**2 * f2 * f4**2 * f5**2)
        for fa, mu in ((f1, 2), (f2, 1), (f4, 2), (f5, 2)):
            self.assertEqual(factor_to_mult_map[fa], mu)
            del factor_to_mult_map[fa]
        self.assertFalse(factor_to_mult_map)

        # Factor (x**n - x) * (x - 1)**2 to test for dealing with x factor
        n, p = 8, 5
        F = PrimeField.create(p)
        coe = [F.zero]*(n+1)
        coe[1], coe[-1] = -F.one, F.one
        P = Poly.create(domain=F)
        p1 = P(coe)
        coe2 = [F.one, -2*F.one, F.one]
        p2 = P(coe2)
        p3 = p1*p2
        answer = {
            P([F.one, F.one, F.one, F.one, F.one, F.one, F.one]) : 1,
            P([-F.one, F.one]) : 3,
            P([F.zero, F.one]): 1
        }
        factor_to_mult_map = factor(p3)
        self.assertEqual(factor_to_mult_map, answer)

        # In F(5): (x**2 + x + 1)(x + 3) = x**3 + 4*x**2 + 4*x + 3
        F = PrimeField.create(5)
        P = Poly.create(domain=F)
        F0, F1, F2, F3, F4 = F.zero, F.one, F(2), F(3), F(4)
        p = P([F3, F4, F4, F1])
        answer = {
            P([F.one, F.one, F.one]) : 1,
            P([F(3), F.one]) : 1
        }
        factor_to_mult_map = factor_poly_in_prime_field(p)
        self.assertEqual(factor_to_mult_map, answer)

        # In F(5): (x**2 + 2*x + 3)(x**2 + x + 1) = x**4 + 3*x**3 + 1*x**2 + 0*x + 1
        p = P([F3, F0, F1, F3, F1])
        answer = {
            P([F.one, F.one, F.one]) : 1,
            P([F(3), F(2), F.one]) : 1
        }
        factor_to_mult_map = factor_poly_in_prime_field(p)
        self.assertEqual(factor_to_mult_map, answer)
