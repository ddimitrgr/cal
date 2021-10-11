from typing import List, Dict, Optional
from fractions import Fraction
from cal.util import fraction_root
from cal.base import ext_euclid_algo
from cal.poly import Poly
from cal.field import PrimeField
from cal.tensor import Tensor, eigen_vector_calc


def square_free_poly_decomposition_in_q(p: Poly) -> Dict[Poly, int]:
    """
    Square-free decomposition of a monic polynomial with rational coefficients.

    :param p:
    :return: dictionary of factors with their multiplicities
    """
    poly_index: Dict[Poly, int] = dict()
    if len(p) < 2:
        poly_index[p] = 1
        return poly_index
    aa: List[Poly] = []
    bb: List[Poly] = []
    cc: List[Poly] = []
    dd: List[Poly] = []
    i = 0
    pd = p.derivative()
    _, _, a = ext_euclid_algo(p, pd)
    b = p // a
    c = pd // a
    aa.append(a)
    bb.append(b)
    cc.append(c)
    dd.append(c - b.derivative())
    while True:
        i += 1
        _, _, a = ext_euclid_algo(bb[-1], dd[-1])
        b = bb[-1]//a
        c = dd[-1]//a
        aa.append(a)
        bb.append(b)
        cc.append(c)
        if bb[-1] == bb[-1].one:
            break
        else:
            dd.append(c - b.derivative())
    for e_min1, f in enumerate(aa[1:]):
        if len(f) > 1:
            f.make_monic()
            poly_index[f] = e_min1+1
    return poly_index


def square_free_poly_decomposition_in_finite_field(p: Poly) -> Dict[Poly, int]:
    factor_to_mult_map: Dict[Poly, int] = dict()
    p.derivative()
    _, _, c = ext_euclid_algo(p, p.derivative())
    c.make_monic()
    w: Poly = p//c

    i = 1
    while len(w) > 1:
        _, _, y = ext_euclid_algo(w, c)
        y.make_monic()
        f, w, c = w//y, y, c//y
        factor_to_mult_map[f] = i
        i += 1

    if c.rank() > 1:
        F: PrimeField = p.domain_of_coeff
        q = F.prime
        highest_pow = (len(c.coefficient_list) - 1)//q
        li: List[F] = [F.zero]*(1+highest_pow)
        for i in range(1+highest_pow):
            li[i] = c.coefficient_list[i*q]
        c_root = type(p)(li)
        more_factors = square_free_poly_decomposition_in_finite_field(c_root)
        for fa, mu in more_factors.items():
            if fa not in factor_to_mult_map:
                factor_to_mult_map[fa] = q*mu
            else:
                factor_to_mult_map[fa] += q*mu
    return factor_to_mult_map


def square_root_of_poly_in_q(p: Poly) -> Optional[Poly]:
    """
    Find polynomial that when squared gives p(x).

    :param p: Polynomial with rational coefficients
    :return: the root polynomial or None if the root does not exist
    """
    lead_coeff = p.coefficient_list[-1]
    lead_coeff_root = fraction_root(lead_coeff)
    if lead_coeff_root == Fraction(-1, 1):
        return None
    pmo = type(p)(p.coefficient_list)
    poly_index = square_free_poly_decomposition_in_q(pmo)
    o = type(p)([lead_coeff_root])
    f: Dict[Poly, int] = {}
    for po, e in poly_index.items():
        e_div2, m = divmod(e, 2)
        if m == 1:
            return None
        f[po] = e_div2
    for po, e in f.items():
        o *= po**e
    return o


def factor_poly_in_prime_field(p: Poly) -> Dict[Poly, int]:
    """
    Factor a polynomial in into irreducible factors. The coefficients must be in
    a finite field and all factors must have multiplicity 1.
    :param p:
    :return: a dictionary of the factors with their multiplicity with must be 1.
    """
    F: PrimeField = p.domain_of_coeff
    P: Poly = type(p)
    factors: Dict[Poly, int] = dict()
    p_target = p

    # Remove any x**1 factor !
    if p.coefficient_list[0] == F.zero:
        f = P([F.zero, F.one])
        factors[f] = 1
        p_target = p // f
        if p_target.rank() == 1:
            return factors

    # Deal with the rest of the factors
    FL = [F(i) for i in range(F.prime)]
    n, q = p_target.rank(), F.prime
    B = Tensor(shape=(n, n), entry_type=F)
    B.s((0, 0), F.one)
    for i in range(1, n):
        coeff = [F.zero]*(i*q)
        coeff.append(F.one)
        b: P = P(coeff) % p_target
        if len(b) > 1:
            B[:len(b), i] = Tensor(b.coefficient_list, shape=(len(b), ), entry_type=F)
        else:
            B[0, i] = b.coefficient_list[0]

    E = eigen_vector_calc(B, eigen_values=[F.one])
    p_remain = p_target
    if E is None:
        factors[p_target] = 1
        return factors
    if E.dim == 1:
        coeff = [E.g(i) for i in E.idx()]
        f = P(coeff)
        for c in FL:
            ftrial = f - P([c])
            if not ftrial.is_zero():
                _, _, gcd = ext_euclid_algo(p_remain, ftrial)
                gcd.make_monic()
                if len(gcd) > 1:
                    p_remain = p_remain // gcd
                    if gcd not in factors:
                        factors[gcd] = 1
    else:
        for i in range(E.shape[1]):
            coeff = [E[j + (i,)] for j in E[:, i].idx()]
            f = P(coeff)
            for c in FL:
                ftrial = f - P([c])
                if not ftrial.is_zero():
                    _, _, gcd = ext_euclid_algo(p_remain, ftrial)
                    gcd.make_monic()
                    if len(gcd) > 1:
                        p_remain = p_remain // gcd
                        if gcd not in factors:
                            factors[gcd] = 1

    if len(p_remain) > 1:
        if p_remain not in factors:
            factors[p_remain] = 1

    # Polynomial is irreducible
    if len(factors) == 1:
        return factors

    # More than one factor: REDO for each
    # to determine if they're irreducible.
    more_factors: Dict[Poly, int] = dict()
    for fa, mu in factors.items():
        new_factors = factor_poly_in_prime_field(fa)
        for new_fa, new_mu in new_factors.items():
            if new_fa not in more_factors:
                more_factors[new_fa] = mu*new_mu
    return more_factors


def factor(p: Poly) -> Dict[Poly, int]:
    """
    Factor polynomial that has coefficients in a finite field.
    The polynomial factors can have multiplicity > 1.

    :param p:
    :return: dictionary of factors and their multiplicity
    """
    poly_index: Dict[Poly, int] = dict()
    di = square_free_poly_decomposition_in_finite_field(p)
    for p2, e2 in di.items():
        if len(p2) == 2:
            if p2 not in poly_index:
                poly_index[p2] = e2
            else:
                poly_index[p2] += e2
        elif len(p2) > 2:
            di2 = factor_poly_in_prime_field(p2)
            for p3, e3 in di2.items():
                if p3 not in poly_index:
                    poly_index[p3] = e2*e3
                else:
                    poly_index[p3] += e2*e3
        else:
            pass
    return poly_index
