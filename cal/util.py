from typing import Dict, List, Tuple, Union, Optional, Callable
from fractions import Fraction
from random import randint
from math import ceil
from cal.base import efficient_int_pow_modulo, ext_euclid_algo_for_ints


def factor_int(n: int) -> Dict[int, int]:
    i, remain_n = 2, n
    o: Dict[int, int] = dict()
    while i <= remain_n:
        power = 0
        while True:
            d, m = divmod(remain_n, i)
            if m > 0:
                break
            else:
                remain_n = d
                power += 1
        if power > 0:
            o[i] = power
        i += 1
    return o


def int_list_gcd_and_lcd(li: List[int]) -> Tuple[int, int]:
    _, _, gcd = ext_euclid_algo_for_ints(li[0], li[1])
    lcd = (li[0]*li[1])//gcd
    for e in li[2:]:
        _, _, gcd = ext_euclid_algo_for_ints(gcd, e)
        _, _, c = ext_euclid_algo_for_ints(lcd, e)
        lcd = max(lcd, (lcd*e)//c)
    return gcd, lcd


def euler_phi(x: Union[int, Dict[int, int]]) -> int:
    d: Dict[int, int] = factor_int(x) if isinstance(x, int) else x
    phi = 1
    for fa, mu in d.items():
        phi *= (fa - 1) * fa**(mu - 1)
    return phi


def fun_moebius(x: Union[int, Dict[int, int]]) -> int:
    d: Dict[int, int] = factor_int(x) if isinstance(x, int) else x
    if len(d) == 0:
        return 1
    elif any([(e > 1) for f, e in d.items()]):
        return 0
    else:
        return (-1)**len(d)


def int_root(n: int) -> int:
    i, remain_n, roo = 2, n, 1
    while i <= remain_n:
        power = 0
        while True:
            d, m = divmod(remain_n, i)
            if m > 0:
                break
            else:
                remain_n = d
                power += 1
        if power > 0:
            power_div2, m = divmod(power, 2)
            if m == 1:
                return -1
            roo *= i**power_div2
        i += 1
    return roo if (n > 0) else 0


def fraction_root(f: Fraction) -> Fraction:
    lt_1 = f.numerator < f.denominator
    m1 = f.numerator if lt_1 else f.denominator
    m2 = f.denominator if lt_1 else f.numerator
    rm1 = int_root(m1)
    if rm1 == -1:
        return Fraction(-1, 1)
    rm2 = int_root(m2)
    if rm2 == -1:
        return Fraction(-1, 1)
    return Fraction(rm1, rm2) if lt_1 else Fraction(rm2, rm1)


def is_prime(p: int, minus_log2_err_prob=40) -> bool:
    u, t = (p - 1), 0
    while True:
        d, m = divmod(u, 2)
        if m == 0:
            u = d
            t += 1
        else:
            break
    # Run a series of minus_log2_err_prob tests. If one fails, p is not a prime.
    # If there is no failure, the probability that p is a prime is >= 1 - 2**-minus_log2_err_prob
    for i in range(minus_log2_err_prob):
        a = randint(1, p - 1)
        # Single test
        x = [efficient_int_pow_modulo(a, u, p)]
        for j in range(1, t+1):
            x.append((x[-1]**2) % p)
            if (x[-1] == 1) and (x[-2] != 1) and (x[-2] != p-1):
                # test fail
                return False
        if x[-1] != 1:
            # test fail
            return False
    return True


def find_prime(p_min: int,
               minus_log2_err_prob: int=40,
               filter: Callable[[int], bool]=lambda x: True) -> int:
    n = (p_min+1) if (p_min % 2 == 0) else p_min
    while not(is_prime(n, minus_log2_err_prob=minus_log2_err_prob) and filter(n)):
        n += 2
    return n


def find_prime_field_gen(p: int) -> int:
    return find_prime_field_element_with_order(e=p-1, p=p)


def find_prime_field_element_with_order(e: int, p: int) -> int:
    if (p-1) % e > 0:
        raise Exception("The order ({e}) should not divide p - 1 ({p-1}) !")
    g = 2
    while g < p:
        # Test elements. The density of generators in the sample is φ(p-1)/(p-2).
        g_pow, not_found = g, False
        for i in range(2, e):
            g_pow = g_pow*g % p
            if g_pow == 1:
                not_found = True
                break
        g_pow = g_pow*g % p
        if not(not_found) and g_pow == 1:
            return g
        g += 1


def log(y: int, base: int, p: int) -> Optional[int]:
    m = ceil((p-1)**0.5)
    store: Dict[int, int] = dict()
    for j in range(0, m):
        store[base**j % p] = j
    base_inv, _, gcd = ext_euclid_algo_for_ints(base, p)
    base_inv_pow_m = base_inv**m % p
    t = y
    for i in range(0, m):
        if t in store:
            return i*m + store[t]
        t = t*base_inv_pow_m % p


def int_root_modulo(y: int, p: int) -> Optional[int]:
    if y**((p-1)/2) % p == p-1:
        return None
    Q, S = p - 1, 0
    while Q % 2 == 0:
        Q = Q // 2
        S += 1
    z = 2
    while z < p:
        if z**((p-1)//2) % p == p - 1:
            break
        z += 1
    M = S
    c = z**Q % p
    t = y**Q % p
    R = y**((Q+1)//2) % p
    while True:
        if t == 0:
            return 0
        if t == 1:
            return R
        i_min, t_pow = 1, t*t % p
        while t_pow != 1 and i_min < M:
            i_min += 1
            t_pow = t_pow*t_pow % p
        if i_min == M:
            return None
        b = c**(2**(M-i_min-1) % (p-1)) % p
        M = i_min
        c = b*b % p
        t = t*c % p
        R = R*b % p
