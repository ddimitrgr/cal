import unittest
from fractions import Fraction
from cal.poly import Poly
from cal.field import PrimeField
from cal.tensor import Tensor, prod, \
    slow_inverse, companion_matrix, \
    characteristic_poly, eigen_vector_calc, \
    gaussian_elimination


class TestTensorOperations(unittest.TestCase):

    def test_notation(self):
        # Tensor with rational entries
        shape = (4, 3)
        n = prod(shape)[-1]
        b = [Fraction(i, 1) for i in range(n)]
        t = Tensor(b, shape=shape)

        t @ t.T
        t*4
        4*t
        t*Fraction(3, 5)
        Fraction(3, 5)*t
        -t
        t + Tensor.Identity(shape=t.shape, entry_type=t.entry_type)

        # Tensor with entries from finite field

        F = PrimeField.create(p=5)
        shape = (4, 3)
        n = prod(shape)[-1]
        b = [F(i) for i in range(n)]
        t = Tensor(b, shape=shape, entry_type=F)

        t @ t.T
        t*F.one
        F.one*t
        t*4
        4*t
        -t
        t + Tensor.Identity(shape=t.shape, entry_type=t.entry_type)

    def test_tensor_index(self):
        shape = (4, 3, 2, 3)
        n = prod(shape)[-1]
        b = [Fraction(i, 1) for i in range(n)]
        t = Tensor(b, shape=shape)
        for i, idx in enumerate(t.idx()):
            self.assertEqual(t.g(idx), i)

    def test_contract(self):
        shape = (4, 3, 2, 2)
        n = prod(shape)[-1]
        b = [Fraction(i % 4, 1) for i in range(n)]
        t = Tensor(b, shape=shape)
        tc = t.contract(3, 4)
        for _, e in tc:
            self.assertEqual(e, 6)
        t.contract(1, 2)
        t.contract(1, 3)
        t.contract(2, 3)
        t.contract(2, 4)

    ##############################################################

    def test_characteristic_poly(self):
        # Test 1
        P = Poly.create(domain=Fraction)
        c = [Fraction(-5, 2), Fraction(2, 1), Fraction(-3, 1), Fraction(3, 1), Fraction(1, 1)]
        p = P(c)
        t = companion_matrix(p)
        p2 = characteristic_poly(t)
        self.assertEqual(p, P(p2))

        # Test 2
        b = [
            [Fraction(0, 1), Fraction(2, 1), Fraction(3, 1)],
            [Fraction(-2, 1), Fraction(-3, 1), Fraction(1, 1)],
            [Fraction(5, 1), Fraction(1, 1), Fraction(-1, 1)]
        ]
        X = Tensor(b)
        p = characteristic_poly(X)
        I = Tensor.Identity(shape=X.shape, entry_type=X.entry_type)
        Y = I*p[0]
        Xp = X
        for c in p[1:]:
            Y += Xp*c
            Xp @= X
        for idx in Y.idx():
            self.assertEqual(Y.g(idx), Y.zero_entry)

    def test_inversion_by_leverrier_iter(self):
        b = [
            [Fraction(0, 1), Fraction(2, 1), Fraction(3, 1)],
            [Fraction(-2, 1), Fraction(-3, 1), Fraction(1, 1)],
            [Fraction(5, 1), Fraction(1, 1), Fraction(-1, 1)]
        ]
        X = Tensor(b)
        Xi = slow_inverse(X)
        I = X@Xi
        for idx in I.idx():
            if all([e == idx[0] for e in idx[1:]]):
                self.assertEqual(I.g(idx), I.unit_entry)
            else:
                self.assertEqual(I.g(idx), I.zero_entry)

    def test_eigen_decomposition(self):
        _D = [
            [Fraction(1, 2), Fraction(0, 1), Fraction(0, 1)],
            [Fraction(0, 1), Fraction(1, 3), Fraction(0, 1)],
            [Fraction(0, 1), Fraction(0, 1), Fraction(1, 3)],
        ]
        # Orthogonal row vectors with norm square 2, 24, 3
        _UT = [
            [Fraction(1, 1), Fraction(1, 1), Fraction(0, 1)],
            [Fraction(-2, 1), Fraction(2, 1), Fraction(4, 1)],
            [Fraction(-1, 1), Fraction(1, 1), Fraction(-1, 1)],
        ]
        # Eigenvalues 8 and 1 x2
        eig_set = {Fraction(1, 1): 2, Fraction(8, 1): 1}

        D = Tensor(_D, entry_type=Fraction)
        UT = Tensor(_UT, entry_type=Fraction)
        U = UT.T
        A = U @ D @ UT

        for eig, mult in eig_set.items():
            V = eigen_vector_calc(A, [eig], ortho=True)
            if V.dim == 1:
                self.assertEqual(mult, 1)
                self.assertEqual(A @ V, V*eig)
            else:
                self.assertEqual(mult, V.shape[1])
                for i in range(V.shape[1]):
                    self.assertEqual(A @ V[:, i], V[:, i] * eig)

    def test_gauss_elim(self):
        # Compare inverse from gaussian_elimination() with inverse from slow_inverse()
        li = [
            [Fraction(0, 1), Fraction(2, 1), Fraction(3, 1)],
            [Fraction(-2, 1), Fraction(-3, 1), Fraction(1, 1)],
            [Fraction(5, 1), Fraction(1, 1), Fraction(-1, 1)]
        ]

        X, Y = Tensor(li), Tensor(li)
        Xis = slow_inverse(X)
        PROW, PCOL, XREDUCED, _Xi, Ni = gaussian_elimination(X)
        # To recover inverse permute _Xi rows. For row i do: i -> PROW(i) -> PCOL(PROW(i))
        Xi = Tensor(shape=_Xi.shape, entry_type=_Xi.entry_type)
        for r in range(Xi.shape[0]):
            Xi[r, :] = _Xi[PCOL[PROW[r]], :]
        self.assertEqual(Xi, Xis)
