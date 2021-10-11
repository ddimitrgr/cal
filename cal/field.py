from typing import cast, List
from fractions import Fraction
from cal.base import FieldType, Field, ext_euclid_algo, ext_euclid_algo_for_ints
from cal.poly import Poly


class PrimeField(Field):

    zero: "Field"
    one: "Field"
    prime: int
    _hash: int = None

    @classmethod
    def create(cls, p: int) -> "PrimeField":

        class NewPrimeField(PrimeField):
            prime = p
        
        NewPrimeField.zero = NewPrimeField(0)
        NewPrimeField.one = NewPrimeField(1)
        return NewPrimeField

    def __init__(self, n: int):
        self._value = n % self.prime

    @property
    def value(self):
        return self._value

    #############################################################################

    def __str__(self):
        return str(self._value)

    def __hash__(self):
        if self._hash is None:
            self._hash = hash((self._value, self.prime))
        return self._hash

    def __add__(self, other):
        return self.__class__((self._value + other._value) % self.prime)

    def __sub__(self, other):
        return self.__class__((self._value - other._value) % self.prime)

    def __mul__(self, other):
        if not isinstance(other, type(self)):
            return super().__mul__(other)
        return self.__class__((self._value * other._value) % self.prime)

    def invert(self) -> "PrimeField":
        if self._value == 0:
            raise Exception("Division by 0 !")
        _inv_value, _, gcd = ext_euclid_algo_for_ints(self._value, self.prime)
        if gcd != 1:
            raise Exception("Extended euclidean division fail !")
        return self.__class__(_inv_value)


class PolyField(Field):

    coeff_type: type
    one: "PolyField"
    zero: "PolyField"
    irreducible_poly: Poly
    _hash: int = None

    @classmethod
    def create(cls, irreducible: Poly) -> "PolyField":

        if irreducible.is_zero():
            raise Exception("Irreducible polynomial is zero !")

        class NewPolyField(PolyField):
            irreducible_poly = irreducible
            coeff_type = irreducible.domain_of_coeff

        NewPolyField.one = NewPolyField([irreducible.unit_coeff])
        NewPolyField.zero = NewPolyField([])
        return NewPolyField

    def __init__(self, coeff: List[FieldType]):
        self._poly = type(self.irreducible_poly)(coeff) % self.irreducible_poly

    @property
    def poly(self) -> Poly:
        return self._poly

    def __str__(self):
        return str(self._poly)

    def __hash__(self):
        if self._hash is None:
            self._hash = hash((self._poly, self.irreducible_poly))
        return self._hash

    def __getitem__(self, item):
        return self._poly[item]

    def __setitem__(self, key, value):
        self._poly[key] = value

    ####################################################################################

    def __add__(self, other):
        poly = (self._poly + other._poly) % self.irreducible_poly
        return type(self)(poly.coefficient_list)

    def __sub__(self, other):
        poly = (self._poly - other._poly) % self.irreducible_poly
        return type(self)(poly.coefficient_list)

    def __mul__(self, other):
        if not isinstance(other, type(self)):
            if isinstance(other, self.coeff_type):
                return type(self)([other*c for c in self.poly.coefficient_list])
            return super().__mul__(other)
        poly = (self._poly * other._poly) % self.irreducible_poly
        return type(self)(poly.coefficient_list)

    def invert(self) -> "PolyField":
        if self._poly.is_zero():
            raise Exception("Division by 0 !")
        inv_self_poly, _, gcd = ext_euclid_algo(self._poly, self.irreducible_poly)
        cast(Poly, inv_self_poly).normalize(cast(Poly, gcd).coefficient_list[0])
        if gcd.rank() != 1:
            raise Exception("Extended euclidean division fail !")
        return type(self)(cast(Poly, inv_self_poly).coefficient_list)


def is_poly_root(poly: Poly, root: PolyField) -> bool:
    res = root.zero
    for power, c in enumerate(poly.coefficient_list):
        res += type(root)([c])*(root**power)
    return res == root.zero
