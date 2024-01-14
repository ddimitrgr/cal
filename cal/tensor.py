from fractions import Fraction
from copy import copy as python_copy
from copy import deepcopy
from typing import Sequence, List, Tuple, Union, Optional, Iterator
from cal.base import FieldType


def prod(li: Sequence[int]) -> List[int]:
    p: List[int] = [li[0]]
    for i in range(1, len(li)):
        p.append(p[-1]*li[i])
    return p


class Tensor:

    @classmethod
    def Identity(cls, shape: Sequence[int], entry_type: type=Fraction) -> "Tensor":
        I = cls(shape=shape, entry_type=entry_type)
        for i, j in enumerate(I.idx()):
            if all([(j[0] == x) for x in j[1:]]):
                I._buffer[i] = I.unit_entry
        return I

    @property
    def entry_type(self) -> type:
        return self._entry_type

    @property
    def zero_entry(self) -> FieldType:
        return self._zero_entry

    @property
    def unit_entry(self) -> FieldType:
        return self._unit_entry

    @property
    def shape(self) -> Tuple[int, ...]:
        return tuple(self._shape)

    @property
    def size(self) -> int:
        return prod(self._shape)[-1]

    @property
    def dim(self) -> int:
        return len(self._shape)

    def __init__(self,
                 data: Optional[Sequence[FieldType]]=None,
                 shape: Optional[Sequence[int]]=None,
                 entry_type: type=Fraction):

        self._entry_type = entry_type
        self._zero_entry: FieldType = \
            Fraction(0, 1) if (entry_type == Fraction) else entry_type.zero
        self._unit_entry: FieldType = \
            Fraction(1, 1) if entry_type == Fraction else entry_type.one
        self._offset = 0

        if shape is None:
            # Data is list of lists => get shape from data dimension
            self._shape: List[int] = []

            sub = data
            while isinstance(sub, list):
                self._shape.append(len(sub))
                sub = sub[0]

            self._base = [0] * len(self._shape)
            self._skips = prod(list(reversed(self._shape)))
            self._skips = list(reversed(self._skips[:-1])) + [1]

            # Store in linear buffer
            self._buffer = []
            for idx in self.idx():
                self._buffer.append(Tensor.get_element_from_list_of_lists(data, idx))
        else:
            # If data is provided is will be linear buffer and is assigned.
            # If just shape is provided => fill with zeros
            self._shape = list(shape)

            self._base = [0]*len(self._shape)
            self._skips = prod(list(reversed(self._shape)))
            self._skips = list(reversed(self._skips[:-1])) + [1]

            if data is not None:
                self._buffer = data
            else:
                # Fill linear buffer
                self._buffer = [self.zero_entry]*prod(shape)[-1]

    def __hash__(self):
        return hash((tuple(self._shape), tuple(self._buffer)))

    def __eq__(self, other):
        return hash(self) == hash(other)

    def copy(self) -> "Tensor":
        cp = Tensor(deepcopy(self._buffer), entry_type=self.entry_type, shape=self.shape)
        cp._offset = self._offset
        cp._skips = python_copy(self._skips)
        cp._base = python_copy(self._base)
        return cp

    def calc_linear_address(self, idx: Tuple[int, ...]) -> int:
        a = self._offset
        for i, v in enumerate(idx):
            a += v*self._skips[i]
        return a

    def g(self, idx: Tuple[int, ...]) -> FieldType:
        return self._buffer[self.calc_linear_address(idx)]

    def s(self, idx: Tuple[int, ...], e: FieldType) -> "Tensor":
        self._buffer[self.calc_linear_address(idx)] = e
        return self

    @staticmethod
    def get_element_from_list_of_lists(li: List[list], idx: Sequence[int]) -> FieldType:
        o = li[idx[0]]
        for k in range(1, len(idx)):
            o = o[idx[k]]
        return o

    def remove_singletons(self, *args) -> "Tensor":
        for dim in args:
            if self._shape[dim] > 1:
                raise Exception(f"Dimension {dim} is not a singleton")
        sorted_dims = sorted(args, reverse=True)
        for i, dim in enumerate(sorted_dims):
            self._offset += self._skips[dim-i]*self._base[dim-i]
            del self._shape[dim-i]
            del self._base[dim-i]
            del self._skips[dim-i]
        return self

    def slice(self, item: Union[slice, Tuple[slice]], compact: bool=True):
        # Removes singleton dimensions by default !
        offset = 0
        base = []
        shape = []
        skips = []
        if len(item) < self.dim:
            raise Exception("Not enough dimensions to specify a slice !")

        for d, slot in enumerate(item):
            if isinstance(slot, slice):
                sta = self._base[d] if (slot.start is None) else slot.start
                sto = self.shape[d] if (slot.stop is None) else slot.stop
                if sto > sta + 1:
                    base.append(sta)
                    shape.append(sto - sta)
                    skips.append(self._skips[d])
                elif sto == sta + 1:
                    offset += sta*self._skips[d]
                else:
                    raise Exception("Empty tensor slice !")
            else:
                offset += slot*self._skips[d]

        ref = Tensor(self._buffer, shape=shape, entry_type=self.entry_type)
        ref._offset = offset
        ref._base = base
        ref._skips = skips
        return ref

    def __getitem__(self, item):
        # TODO: Write efficiently
        if isinstance(item, tuple):
            find_slice = []
            for d, entry in enumerate(item):
                if isinstance(entry, slice):
                    sta = self._base[d] if (entry.start is None) else entry.start
                    sto = (self._base[d] + self._shape[d]) if (entry.stop is None) else entry.stop
                    find_slice.append(sto > sta + 1)
                else:
                    find_slice.append(False)
            has_slice = any(find_slice)
            if has_slice:
                return self.slice(item)
            else:
                # return single element
                return self.g(item)
        else:
            if self.dim > 1:
                raise Exception("Wrong dimension")
            if isinstance(item, slice):
                return self.slice((item, ))
            else:
                return self.g((item, ))

    def __setitem__(self, key, value):
        # TODO: Write efficiently
        if isinstance(key, tuple):
            find_slice = []
            for d, entry in enumerate(key):
                if isinstance(entry, slice):
                    sta = self._base[d] if (entry.start is None) else entry.start
                    sto = (self._base[d] + self._shape[d]) if (entry.stop is None) else entry.stop
                    find_slice.append(sto > sta + 1)
                else:
                    find_slice.append(False)
            has_slice = any(find_slice)
            if has_slice:
                sli = self.slice(key)
                for i in sli.idx():
                    sli.s(i, value.g(i))
            else:
                # set single element
                self.s(key, value)
        else:
            if self.dim > 1:
                raise Exception("Wrong dimension")
            if isinstance(key, slice):
                sli = self.slice((key, ))
                for i in sli.idx():
                    sli.s(i, value.g(i))
            else:
                self.s((key, ), value)

    def set_slice(self, loc, value):
        pass

    def __iter__(self) -> Iterator[Tuple[Tuple[int, ...], FieldType]]:
        for i in self.idx():
            yield i, self.g(i)

    def idx(self) -> Iterator[Tuple[int, ...]]:
        shape = self._shape
        base = self._base
        _fi = [base[k] + sh for (k, sh) in enumerate(shape)]
        max_pos = len(shape) - 1
        i = python_copy(base)
        lin_i = 0
        pos = 0
        skip = prod(list(reversed(shape)))
        num_left = skip[-1]
        skip[-1] = 0
        skip = list(reversed(skip[:-1])) + [1]
        while num_left > 0:
            if i[pos] == _fi[pos]:
                i[pos] = base[pos]
                lin_i += (base[pos] - _fi[pos])*skip[pos]
                pos -= 1
                i[pos] += 1
                lin_i += skip[pos]
            else:
                if pos < max_pos:
                    pos += 1
                else:
                    num_left -= 1
                    yield tuple(i)
                    i[pos] += 1
                    lin_i += skip[pos]

    def __matmul__(self, other: "Tensor"):
        shape = self._shape[:-1] + other._shape[1:]
        if self._shape[-1] != other._shape[0]:
            raise Exception(f"Multiplication fail: matrices have wrong dimension ({self._shape} x {other._shape}) !")
        if len(shape) > 0:
            o = Tensor(shape=shape, entry_type=self.entry_type)
            n1 = self.dim - 1
            for i in o.idx():
                o[i] = self[i[:n1]+(0,)]*other[(0,)+i[n1:]]
                for k in range(1, self._shape[-1]):
                    o[i] += self[i[:n1]+(k,)]*other[(k,)+i[n1:]]
            return o
        else:
            # Vector multiplication: return scalar
            o = self[0] * other[0]
            for k in range(1, self._shape[-1]):
                o += self[k] * other[k]
            return o

    @property
    def T(self) -> "Tensor":
        # Return transpose !
        ref = Tensor(self._buffer, entry_type=self.entry_type, shape=list(reversed(self._shape)))
        ref._offset = self._offset
        ref._base = list(reversed(self._base))
        ref._skips = list(reversed(self._skips))
        return ref

    def contract(self, i: int, j: int) -> Union[FieldType, "Tensor"]:
        """
        Contract indices i, j of tensor.

        :param i: 1-based index
        :param j: 1-based index
        :return: tensor or scalar accordingly
        """
        valid = (i != j) and (i > 0) and (j > 0) and (i <= self.dim) and (j <= self.dim)
        if not valid:
            raise Exception("Error in index specification")
        if self.dim == 2:
            # Return number
            o = self[0, 0]
            for i in range(1, self._shape[0]):
                o += self[i, i]
            return o
        else:
            # Return tensor
            i_le, i_ri = (i - 1, j - 1) if (i < j) else (j - 1, i - 1)
            shape = self.shape[:i_le] + self.shape[i_le+1:i_ri] + self.shape[i_ri+1:]
            o = Tensor(shape=shape, entry_type=self.entry_type)
            for idx in self.idx():
                cut_idx = idx[:i_le] + idx[i_le + 1:i_ri] + idx[i_ri + 1:]
                o.s(cut_idx, o.g(cut_idx) + self[idx])
            return o

    def __add__(self, other):
        o = Tensor(shape=self._shape, entry_type=self.entry_type)
        for i in self.idx():
            o.s(i, self.g(i) + other.g(i))
        return o

    def __sub__(self, other):
        o = Tensor(shape=self._shape, entry_type=self.entry_type)
        for i in self.idx():
            o.s(i, self.g(i) - other.g(i))
        return o

    def __mul__(self, other):
        o = Tensor(shape=self._shape, entry_type=self.entry_type)
        if isinstance(other, Tensor):
            for i in self.idx():
                o.s(i, self.g(i) * other.g(i))
        else:
            for i in self.idx():
                o.s(i, self.g(i) * other)
        return o

    def __rmul__(self, other):
        return self.__mul__(other)

    def __neg__(self):
        return self*(-self.unit_entry)


def companion_matrix(p: List[FieldType]) -> Tensor:
    n = len(p) - 1
    o = Tensor(shape=[n, n], entry_type=type(p[-1]))
    if p[-1] != o.unit_entry:
        raise Exception("Polynomial must be monic !")
    for i in range(n-1):
        o[i+1, i] = o.unit_entry
    for i in range(n):
        o[i, n-1] = -p[i]
    return o


def leverrier_iter(t: Tensor) -> Tuple[List[FieldType], Optional[Tensor]]:
    """
    Computes the characteristic polynomial of N x N matrix in O(N^4)

    :param t: N x N tensor
    :return: list of polynomial coefficients in ascending exponent
    """
    if t.dim != 2 or (t._shape[0] != t._shape[1]):
        raise Exception("Dimension must be 2 !")
    o: List[FieldType] = []
    I = Tensor.Identity(shape=t.shape, entry_type=t.entry_type)
    B = Z = t
    ti = None
    for k in range(0, t._shape[0]):
        co = Z.contract(1, 2)
        if co == Z.zero_entry:
            o.append(co)
        else:
            # TODO: Iteration fails in a prime field when (1+k) % p = 0
            o.append(-((1+k)*(co**-1))**-1)
        if k + 1 == t.shape[0]:
            ti = None
            if o[-1] != t.zero_entry:
                cc = -(o[-1]**-1)
                ti = B*cc
            break
        B = Z + I*o[-1]
        Z = t @ B
    o = list(reversed(o))
    o.append(t.unit_entry)
    return o, ti


def characteristic_poly(t: Tensor) -> List[FieldType]:
    return leverrier_iter(t)[0]


def slow_inverse(t: Tensor) -> Optional[Tensor]:
    return leverrier_iter(t)[1]


def gaussian_elimination(t: Tensor, row_pivot=True, col_pivot=True) -> Tuple[List[int], List[int], Tensor, Tensor, int]:
    """
    Gaussian eliminination. To recover inverse from the permuted inverse,
    rotate the rows: the actual i-th row row is located pcol[prow[i]]

    :param t: N x N tensor
    :return: tuple (prow - permuted rows,
                    pcol - permuted cols,
                    permuted reduced matrix,
                    permuted inverse matrix,
                    rank)
    """
    if t.dim != 2:
        raise Exception("Dimension must be 2 !")

    t2, z, o = t.copy(), t.zero_entry, t.unit_entry
    i2 = Tensor.Identity(shape=t.shape, entry_type=t.entry_type)
    prow, pcol, n_left = [i for i in range(t.shape[0])], [j for j in range(t.shape[1])], 0

    while n_left < min(t.shape[0], t.shape[1]):
        # Find pivot
        pivot_not_found = True
        for r_pivot in range(n_left, (t.shape[0] if row_pivot else (1+n_left))):
            for c_pivot in range(n_left, (t.shape[1] if col_pivot else (1+n_left))):
                if t2[prow[r_pivot], pcol[c_pivot]] != t.zero_entry:
                    pivot_not_found = False
                    if r_pivot > n_left:
                        prow[n_left], prow[r_pivot] = prow[r_pivot], prow[n_left]
                    if c_pivot > n_left:
                        pcol[n_left], pcol[c_pivot] = pcol[c_pivot], pcol[n_left]
                    break
            if not pivot_not_found:
                break

        if pivot_not_found:
            return prow, pcol, t2, i2, n_left

        m = o * t2[prow[n_left], pcol[n_left]]**-1

        for c in range(i2.shape[1]):
            i2[prow[n_left], pcol[c]] = i2[prow[n_left], pcol[c]]*m
            for r in range(0, n_left):
                i2[prow[r], pcol[c]] -= t2[prow[r], pcol[n_left]] * i2[prow[n_left], pcol[c]]
            for r in range(n_left+1, i2._shape[0]):
                i2[prow[r], pcol[c]] -= t2[prow[r], pcol[n_left]] * i2[prow[n_left], pcol[c]]

        for c in reversed(range(n_left, t2.shape[1])):
            t2[prow[n_left], pcol[c]] = t2[prow[n_left], pcol[c]]*m
            for r in range(0, n_left):
                t2[prow[r], pcol[c]] -= t2[prow[r], pcol[n_left]] * t2[prow[n_left], pcol[c]]
            for r in range(n_left+1, t2.shape[0]):
                t2[prow[r], pcol[c]] -= t2[prow[r], pcol[n_left]] * t2[prow[n_left], pcol[c]]

        n_left += 1
    return prow, pcol, t2, i2, n_left


def inverse(t: Tensor) -> Optional[Tensor]:
    if t.shape[0] != t.shape[1]:
        return None
    prow, pcol, red, ti, ra = gaussian_elimination(t, row_pivot=True, col_pivot=False)
    if ra < t.shape[0]:
        return None
    i = []
    for r in range(t.shape[0]):
        for c in range(t.shape[1]):
            i.append(ti[prow[r], pcol[c]])
    o = Tensor(i, shape=(len(prow), len(pcol)), entry_type=t.entry_type)
    return o


def gram_schmidt_ortho(t: Tensor) -> Optional[Tensor]:
    if t.dim == 1:
        return t
    if t.dim != 2:
        raise Exception("Dimension must be 2 !")
    V, N = t.copy(), t.shape[0]
    if V.shape[1] > 1:
        for i in range(V.shape[1]):
            Vc = V[:, i].copy()
            if V[:, i]@V[:, i] == V.zero_entry:
                if i > 1:
                    return V[:, :i]
                elif i == 1:
                    return V[:, 0]
                else:
                    return None
            for j in range(i):
                s = V[:, j].T @ V[:, j]
                sj = V[:, j].T @ Vc
                for k in range(V.shape[0]):
                    V.s((k, i), V.g((k, i)) - (sj*(s**-1))*V.g((k, j)))
    return V


def eigen_vector_calc(t: Tensor,
                      eigen_values: List[FieldType],
                      ortho: bool=True) -> Optional[Tensor]:
    if t.dim != 2:
        raise Exception("Input must be a square matrix !")
    V = Tensor(shape=t.shape, entry_type=t.entry_type)
    tot_num_eig_vec = 0
    for i, v in enumerate(eigen_values):
        m = t.copy()
        for j in range(t.shape[1]):
            m.s((j, j), m.g((j, j)) - v)
        r_perm, c_perm, i_cert, t_inv, n_dim = gaussian_elimination(m)
        num_eig_vec = t.shape[1] - n_dim
        for k in range(num_eig_vec):
            for dep_var in range(0, n_dim):
                V.s(
                    (c_perm[dep_var], tot_num_eig_vec + k),
                    t.zero_entry - i_cert[r_perm[dep_var], c_perm[n_dim + k]] * t.unit_entry
                )
            V.s((c_perm[n_dim + k], tot_num_eig_vec + k), t.unit_entry)
        tot_num_eig_vec += num_eig_vec
    if ortho:
        V = gram_schmidt_ortho(V)
    if tot_num_eig_vec == 0:
        return None
    elif V.dim == 1:
        return V
    else:
        V = V[:, :tot_num_eig_vec]
    return V

