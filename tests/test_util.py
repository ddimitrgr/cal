import unittest
from fractions import Fraction
from random import randint, seed
from cal.util import int_root, fraction_root, \
    find_prime, factor_int, \
    int_list_gcd_and_lcd, euler_phi, \
    find_prime_field_gen, log, \
    find_prime_field_element_with_order, \
    int_root_modulo
from cal.cyclo import calc_cyclotomic_field
from cal.field import is_poly_root

seed(33)


class TestUtils(unittest.TestCase):

    def test_int_root(self):
        n = 0
        self.assertEqual(int_root(n), 0)
        n = 1
        self.assertEqual(int_root(n), 1)
        n = 2
        self.assertEqual(int_root(n), -1)
        n = 13
        self.assertEqual(int_root(n), -1)
        n = 4
        self.assertEqual(int_root(n), 2)
        n = 3 ** 6 * 5 ** 2 * (7 ** 4)
        self.assertEqual(int_root(n)**2, n)
        n = 3 ** 6 * 5 ** 2 * (7 ** 5)
        self.assertEqual(int_root(n), -1)

    def test_fraction_root(self):
        f = Fraction((5 ** 2) * 41 ** 2, 3 ** 2 * (7 ** 2))
        self.assertEqual(fraction_root(f)**2, f)
        f = Fraction((5 ** 0) * 41 ** 2, 3 ** 2 * (7 ** 2))
        self.assertEqual(fraction_root(f)**2, f)
        f = Fraction((5 ** 3) * 41 ** 2, 3 ** 2 * (7 ** 2))
        self.assertEqual(fraction_root(f), Fraction(-1, 1))
        f = Fraction((5 ** 0) * 41 ** 2, 3 ** 2 * 7 ** 3)
        self.assertEqual(fraction_root(f), Fraction(-1, 1))

    def test_find_prime(self):
        self.assertEqual(find_prime(20), 23)
        self.assertEqual(find_prime(54), 59)

    def test_factor_int(self):
        for i in range(20):
            n = randint(1, 2000)
            fa = factor_int(n)
            n_recon = 1
            for f, mult in fa.items():
                n_recon *= f**mult
            s_fa = ' * '.join([f'{k}**{v}' for (k, v) in fa.items()])
            self.assertEqual(n, n_recon)

    def test_gcd_and_lcd(self):
        li = [25, 15, 5, 45]
        gcd, lcd = int_list_gcd_and_lcd(li)
        self.assertEqual(gcd, 5)
        self.assertEqual(lcd, 5**2 * 3**2)

    def test_euler_phi(self):
        n = 2**2 * 3
        # 1, 5, 7, 11
        answer = 4
        self.assertEqual(euler_phi(n), answer)
        n = 2 ** 2 * 5
        # 1, 3, 7, 9, 11, 13, 17, 19
        answer = 8
        self.assertEqual(euler_phi(n), answer)

    def test_find_prime_field_element_with_order(self):
        seed(100)
        p = find_prime(p_min=30)
        for i_e in range(10):
            while True:
                e = randint(2, p-1)
                if (p-1) % e == 0:
                    break
            fe = find_prime_field_element_with_order(e=e, p=p)
            i, pro = 1, fe
            while i < e - 1:
                pro = (pro*fe) % p
                self.assertNotEqual(pro, 1)
                i += 1
            pro = (pro * fe) % p
            self.assertEqual(pro, 1)

    def test_discrete_log(self):
        for i in range(10):
            p_min = randint(2, 10000)
            p = find_prime(p_min)
            g = find_prime_field_gen(p)
            y = randint(1, p-1)
            x = log(y, base=g, p=p)
            y0 = (g**x) % p
            self.assertEqual(y0, y)

    def test_int_root_modulo(self):
        seed(5)
        for i in range(50):
            p_min = randint(2, 40)
            p = find_prime(p_min)
            x = randint(1, p-1)
            r = 2
            y = x**r % p
            x0 = int_root_modulo(y, p=p)
            self.assertEqual(y, x0**r % p)


class TestCyclo(unittest.TestCase):

    def test_unity_roots(self):
        n, p = 9, 5
        P, roots = calc_cyclotomic_field(n, p)
        cnt_of_roots = 0
        for i, r in enumerate(roots):
            is_root = is_poly_root(P.irreducible_poly, r)
            if is_root:
                cnt_of_roots += 1
        self.assertEqual(P.irreducible_poly.rank() - 1, cnt_of_roots)
        self.assertEqual(P.irreducible_poly.rank() - 1, euler_phi(n))
