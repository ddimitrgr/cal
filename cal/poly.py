from typing import List
from fractions import Fraction
from cal.base import FieldType, Ring, EuclideanDomainMixin
from cal.decorators import writer


class PolyRing(Ring):

    domain_of_coeff: type
    zero_coeff: FieldType
    unit_coeff: FieldType

    @classmethod
    def create(cls, domain: type=Fraction) -> "PolyRing":

        z: FieldType = Fraction(0, 1) if domain == Fraction else domain.zero
        o: FieldType = Fraction(1, 1) if domain == Fraction else domain.one

        class NewPolyType(cls):
            domain_of_coeff = domain
            zero_coeff = z
            unit_coeff = o

        NewPolyType.one = NewPolyType([NewPolyType.unit_coeff])
        NewPolyType.zero = NewPolyType([])
        return NewPolyType

    @writer
    def __init__(self, coeff: List[FieldType]):
        self._coeff: List[FieldType] = []
        for i, c in enumerate(reversed(coeff)):
            if c != self.zero_coeff:
                self._coeff = coeff[:-i] if (i > 0) else coeff
                break

    def derivative(self):
        if self.is_zero() or self.rank() == 1:
            return self
        return type(self)([(i+1)*c for i, c in enumerate(self._coeff[1:])])

    @property
    def coefficient_list(self) -> List[FieldType]:
        return self._coeff

    def is_zero(self) -> bool:
        return len(self._coeff) == 0

    @writer
    def normalize(self, c: FieldType) -> "Poly":
        for i in range(len(self)):
            self._coeff[i] = self._coeff[i] * c**-1
        return self

    @writer
    def make_monic(self) -> "Poly":
        return self if self.is_zero() else self.normalize(self._coeff[-1])

    def __len__(self):
        return len(self._coeff)

    def __getitem__(self, key):
        return self._coeff[key]

    def __setitem__(self, key, value):
        self._coeff[key] = value

    def __str__(self):
        if self.is_zero():
            return f"({str(self.zero_coeff)})"
        pts: List[str] = [""]*len(self)
        for i, c in enumerate(reversed(self._coeff[1:])):
            pts[i] = f"({c}) x**{len(self) - i - 1}"
        pts[-1] = f"({self._coeff[0]})"
        return " + ".join(pts)

    def rehash(self) -> None:
        self._hash = hash(tuple(self._coeff))

    def __hash__(self):
        if self._hash is None:
            self.rehash()
        return self._hash

    def __eq__(self, other):
        if len(self) != len(other):
            return False
        else:
            for i in range(len(self)):
                if self._coeff[i] != other._coeff[i]:
                    return False
        return True

    #########################################################################

    def rank(self) -> int:
        return len(self._coeff)

    def __add__(self, other):
        c = [self.zero_coeff]*max(len(self), len(other))
        for n in range(len(self)):
            c[n] += self[n]
        for n in range(len(other)):
            c[n] += other[n]
        return type(self)(c)

    def __sub__(self, other: "Poly") -> "Poly":
        c = [self.zero_coeff] * max(len(self), len(other))
        for n in range(len(self)):
            c[n] += self[n]
        for n in range(len(other)):
            c[n] -= other[n]
        return type(self)(c)

    def __mul__(self, other):
        if type(other) == self.domain_of_coeff:
            return type(self)([c*other for c in self._coeff])
        if not isinstance(other, type(self)):
            return super().__mul__(other)
        if min(len(self), len(other)) == 0:
            return self.zero
        c: List[FieldType] = [self.zero_coeff]*(len(self) + len(other) - 1)
        for n in range(len(c)):
            for i_a in range(0, 1 + min(n, len(self) - 1)):
                i_b = n - i_a
                if (i_b < len(other)) and (i_b >= 0):
                    c[n] += self[i_a]*other[i_b]
        return type(self)(c)


class Poly(PolyRing, EuclideanDomainMixin):

    def __divmod__(self, other):
        if self.is_zero():
            return type(self)([]), type(self)([])
        if other.is_zero():
            raise Exception("Divide by zero !")
        a = type(self)(self.coefficient_list)
        x: List[FieldType] = [self.zero_coeff]*(len(a) - len(other) + 1)
        while True:
            if len(a) < len(other):
                res = type(self)(x)
                return res, a
            else:
                e = len(a) - len(other)
                x[e] = a[-1] * other[-1]**-1
                y = type(self)([self.zero_coeff]*e + [x[e]])
                a = a - other*y
