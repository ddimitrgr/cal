from typing import List, Tuple, Dict
from fractions import Fraction
from cal.util import factor_int, fun_moebius, euler_phi
from cal.poly import Poly
from cal.field import PrimeField, PolyField


def calc_cyclotomic_field(n: int, p: int) -> Tuple[type, List[PolyField]]:
    """
    Extend a field of characteristic p (prime or 0 for Q) with the roots of x**n = 1.
    The extension if a vector space of dimension φ(n) where φ is the Euler function when
    either:
    - p = 0
    - p > 0 and the minimum d for which p**d = 1 % n is φ(n)

    * When p > 0 the cyclotomic polynomial is reducible to φ(n)/d factors (of the same
    degree) where d is the minimum exponent for which p**d = 1 % n.

    :param n: maximum root order
    :param p: base field characteristic
    :return: returns the field class and of the field elements that represent the roots
    """

    if p > 0 and n % p == 0:
        raise Exception("Error: the order of the polynomial divides characteristic !")
    if p > 0:
        for d in range(1, euler_phi(n)):
            if p**d % n == 1:
                raise Exception("Error: cyclotomic polynomial is reducible for p = {p} and n = {n} !")

    f = factor_int(n)

    # Find all distinct divisors of the order and their prime factorization
    d_map_to_f: Dict[int, Dict[int, int]] = dict()
    li_f: List[int] = []
    for fa, mu in f.items():
        li_f.extend([fa]*mu)
    m = len(li_f)
    for i in range(0, 2**m):
        d0, f0 = 1, dict()
        for b in range(0, m):
            if (i >> b) & 1 == 1:
                d0 *= li_f[b]
                if li_f[b] not in f0:
                    f0[li_f[b]] = 0
                f0[li_f[b]] += 1
        d_map_to_f[d0] = f0

    if p == 0:
        R = Poly.create()
        nu, de = R.one, R.one
        for d in d_map_to_f:
            li_c: List[Fraction] = [Fraction(0, 1)] * (1 + d)
            li_c[0], li_c[-1] = Fraction(-1, 1), Fraction(1, 1)
            if fun_moebius(d_map_to_f[n//d]) > 0:
                nu *= R(li_c)
            elif fun_moebius(d_map_to_f[n//d]) < 0:
                de *= R(li_c)
        P = PolyField.create(irreducible=nu//de)
        roots: List[PolyField] = []
        for i in range(n):
            li_c: List[Fraction] = [Fraction(0, 1)]*(i+1)
            li_c[i] = Fraction(1, 1)
            roots.append(P(li_c))
        return P, roots
    else:
        F = PrimeField.create(p)
        R = Poly.create(domain=F)
        nu, de = R.one, R.one
        for d in d_map_to_f:
            li_c: List[Fraction] = [F.zero] * (1 + d)
            li_c[0], li_c[-1] = -F.one, F.one
            if fun_moebius(d_map_to_f[n//d]) > 0:
                nu *= R(li_c)
            elif fun_moebius(d_map_to_f[n//d]) < 0:
                de *= R(li_c)
        P = PolyField.create(irreducible=nu//de)
        roots: List[PolyField] = []
        for i in range(n):
            li_c: List[Fraction] = [F.zero]*(i+1)
            li_c[i] = F.one
            roots.append(P(li_c))
        return P, roots
