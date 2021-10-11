from typing import Tuple, List, Union, Protocol
from fractions import Fraction


class Multiplicative(Protocol):

    @property
    def one(self):
        ...

    def __mul__(self, other):
        ...


class Additive(Protocol):

    @property
    def zero(self):
        ...

    def __add__(self, other):
        ...


def efficient_int_pow_modulo(a: int, n: int, p: int) -> int:
    o, remain, pro = 1, n, a
    while remain > 0:
        remain, digit = divmod(remain, 2)
        if digit > 0:
            o = o*pro % p
        pro = pro*pro % p
    return o


def efficient_pow(a: Multiplicative, n: int) -> Multiplicative:
    o, remain, pro = a.one, n, a
    while remain > 0:
        remain, digit = divmod(remain, 2)
        if digit > 0:
            o *= pro
        pro *= pro
    return o


def efficient_sum(a: Additive, n: int) -> Additive:
    o, remain, su = a.zero, n, a
    while remain > 0:
        remain, digit = divmod(remain, 2)
        if digit > 0:
            o += su
        su += su
    return o


class Group:

    @property
    def unit(self):
        raise NotImplementedError

    def __mul__(self, other):
        raise NotImplementedError

    def invert(self) -> "Group":
        raise NotImplementedError

    #############################################################

    @property
    def one(self) -> "Group":
        return self.unit

    def __pow__(self, power):
        o = self if (power > 0) else (self.invert() if (power < 0) else self.one)
        return efficient_pow(o, abs(power))

    def __truediv__(self, other):
        raise self.__mul__(self.__pow__(-1))


class Ring:

    @property
    def zero(self) -> "Ring":
        raise NotImplementedError

    @property
    def one(self) -> "Ring":
        raise NotImplementedError

    def __str__(self):
        raise NotImplementedError

    def __hash__(self):
        raise NotImplementedError

    def __add__(self, other):
        raise NotImplementedError

    def __sub__(self, other):
        raise NotImplementedError

    ###########################################################

    def __neg__(self):
        return self.zero.__sub__(self)

    def __mul__(self, other):
        # Integer multiplication with field element => sub-classes must use this !
        if isinstance(other, int):
            pro = efficient_sum(self, abs(other))
            return pro if (other > 0) else -pro
        else:
            # All else (eg Tensor) use the other side's __mul__
            return other*self

    def __eq__(self, other):
        return hash(self) == hash(other)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __pow__(self, power):
        return efficient_pow(self, power) if (power > 0) else self.one


class EuclideanDomainMixin:

    def rank(self) -> int:
        raise NotImplementedError

    def __divmod__(self, other):
        raise NotImplementedError

    ##################################################

    def __floordiv__(self, other):
        return self.__divmod__(other)[0]

    def __mod__(self, other):
        return self.__divmod__(other)[1]


class EuclideanDomain(EuclideanDomainMixin, Ring):
    pass


def ext_euclid_algo(a: EuclideanDomain, b: EuclideanDomain) \
        -> Tuple[EuclideanDomain, EuclideanDomain, EuclideanDomain]:
    """
    Extended Euclidean algorithm for rings. For elements a, b
    computes the parameters of the equation:

        x*a + y*b = gcd(a,b)

    :param a: ring element
    :param b: ring element
    :return: tuple (x, y, gcd(a, b)) when a > b else (y, x, gcd(a, b))
    """
    one, zero, a_gt_b = a.one, a.zero, a.rank() > b.rank()
    results: List[Tuple[EuclideanDomain, EuclideanDomain, EuclideanDomain]] = []
    if a_gt_b:
        results.append((a, one, zero))
        results.append((b, zero, one))
    else:
        results.append((b, one, zero))
        results.append((a, zero, one))
    while not(results[-1][0] == zero):
        q, new_r = divmod(results[-2][0],  results[-1][0])
        new_s = results[-2][1] - results[-1][1]*q
        new_t = results[-2][2] - results[-1][2]*q
        results.append((new_r, new_s, new_t))
    if a_gt_b:
        return results[-2][1], results[-2][2], results[-2][0]
    return results[-2][2], results[-2][1], results[-2][0]


def ext_euclid_algo_for_ints(a: int, b: int) -> Tuple[int, int, int]:
    """
    Extended Euclidean algorithm for integers. For a, b
    computes the parameters of the equation:

        x*a + y*b = gcd(a,b)

    :param a: integer
    :param b: integer
    :return: tuple (x, y, gcd(a, b)) when a > b else (y, x, gcd(a, b))
    """
    one, zero, a_gt_b = 1, 0, a > b
    results: List[Tuple[int, int, int]] = []
    if a_gt_b:
        results.append((a, one, zero))
        results.append((b, zero, one))
    else:
        results.append((b, one, zero))
        results.append((a, zero, one))
    while not(results[-1][0] == 0):
        q, new_r = divmod(results[-2][0],  results[-1][0])
        new_s = results[-2][1] - results[-1][1]*q
        new_t = results[-2][2] - results[-1][2]*q
        results.append((new_r, new_s, new_t))
    if a_gt_b:
        return results[-2][1], results[-2][2], results[-2][0]
    else:
        return results[-2][2], results[-2][1], results[-2][0]


class FieldMixin:

    # Not supporting __truediv__ : instead multiply with negative one power.

    @property
    def one(self) -> "FieldMixin":
        raise NotImplementedError

    def invert(self) -> "FieldMixin":
        raise NotImplementedError

    ###########################################################

    def __pow__(self, power):
        o = self if (power > 0) else (self.invert() if (power < 0) else self.one)
        return efficient_pow(o, abs(power))


class Field(FieldMixin, Ring):
    pass


class Q(Field):

    def __init__(self, *args):
        self._fraction: Fraction = args[0] if (len(args) == 1) else Fraction(args[0], args[1])

    def __eq__(self, other):
        return self._fraction == other._fraction

    def invert(self) -> "Field":
        return type(self)(Fraction.__truediv__(Fraction(1, 1), self._fraction))

    def __add__(self, other):
        return type(self)(Fraction.__add__(self._fraction, other._fraction))

    def __sub__(self, other):
        return type(self)(Fraction.__sub__(self._fraction, other._fraction))

    def __mul__(self, other):
        if not isinstance(other, type(self)):
            return super().__mul__(other)
        return type(self)(Fraction.__mul__(self._fraction, other._fraction))

    def __str__(self):
        return str(self._fraction)


setattr(Q, "zero", Q(0, 1))
setattr(Q, "one", Q(1, 1))

FieldType = Union[Field, Fraction]


class IntegerQuotientRing(EuclideanDomain):

    @classmethod
    def create(cls, n: int):

        class NewIntegerQuotientRing(IntegerQuotientRing):
            m = n

        setattr(NewIntegerQuotientRing, "zero", NewIntegerQuotientRing(0))
        setattr(NewIntegerQuotientRing, "one", NewIntegerQuotientRing(1))
        return NewIntegerQuotientRing

    def __init__(self, n: int):
        self._value = n % self.m

    def rank(self) -> int:
        return self._value

    @property
    def value(self) -> int:
        return self.value()

    def __hash__(self):
        return hash((self._value, self.m))

    def __str__(self):
        return str(self._value)

    def __divmod__(self, other):
        raise divmod(self._value, self.m)

    def __add__(self, other):
        return self.__class__((self._value + other._value) % self.m)

    def __sub__(self, other):
        return self.__class__((self._value - other._value) % self.m)

    def __mul__(self, other):
        if not isinstance(other, type(self)):
            return super().__mul__(other)
        return self.__class__((self._value * other._value) % self.m)
