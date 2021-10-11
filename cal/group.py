from math import ceil
from fractions import Fraction
from typing import List, Hashable, Dict, Set, Optional, Tuple
from cal.base import Group, FieldType
from cal.poly import Poly
from cal.util import int_list_gcd_and_lcd, find_prime, \
    find_prime_field_element_with_order, int_root_modulo
from cal.cyclo import calc_cyclotomic_field
from cal.field import PrimeField, PolyField
from cal.factor import factor
from cal.tensor import Tensor, characteristic_poly, eigen_vector_calc

import logging as _logging

_logger = _logging.getLogger(__name__)


class GroupAsTable(Group):

    name: str
    table: List[List[Hashable]]
    symbol_to_loc: Dict[Hashable, int]
    loc_to_symbol: list
    inverses: Dict[Hashable, Hashable]
    classes: dict
    symbol_to_class_rep: Dict[Hashable, set]
    struct_constants: Tensor
    character_table: Tensor

    @classmethod
    def create(cls, t: List[List[Hashable]], name: str="Group"):

        symbol_to_loc = dict()
        loc_to_symbol = [0]*len(t)
        symbol_to_class_rep: Dict[Hashable, set] = dict()
        classes = dict()
        table: List[List[int]] = [[0]*len(t) for _ in range(len(t))]
        inverses: Dict[Hashable, Hashable] = dict()
        orders: List[int] = []

        for i, sym in enumerate(t[0]):
            symbol_to_loc[sym] = i
            loc_to_symbol[i] = sym

        for r in range(len(table)):
            for c in range(len(table[0])):
                table[r][c] = symbol_to_loc[t[r][c]]
                if table[r][c] == 0:
                    inverses[loc_to_symbol[r]] = loc_to_symbol[c]

        for r in range(len(table)):
            order, pro = 1, r
            while pro != 0:
                pro = table[r][pro]
                order += 1
            orders.append(order)

        for sym, loc in symbol_to_loc.items():
            if sym not in symbol_to_class_rep:
                # classify
                for c in range(len(table)):
                    if c != loc:
                        prod_idx = table[c][loc]
                        final_idx = table[prod_idx][symbol_to_loc[inverses[loc_to_symbol[c]]]]
                        final_sym = loc_to_symbol[final_idx]
                        if (final_sym not in symbol_to_class_rep) and (sym not in symbol_to_class_rep):
                            new_class = set([sym, final_sym])
                            classes[sym] = new_class
                            symbol_to_class_rep[sym] = sym
                            symbol_to_class_rep[final_sym] = sym
                        elif final_sym in symbol_to_class_rep:
                            rep = symbol_to_class_rep[final_sym]
                            classes[rep].add(sym)
                            symbol_to_class_rep[sym] = rep
                        elif sym in symbol_to_class_rep:
                            rep = symbol_to_class_rep[sym]
                            classes[rep].add(final_sym)
                            symbol_to_class_rep[final_sym] = rep

        class_list = []
        class_linear_index = dict()
        for rep, cl in classes.items():
            class_list.append(rep)
            class_linear_index[rep] = len(class_list[1:])
        nc = len(classes)
        struct_constants_in_fractions: Tensor = Tensor(shape=(nc, nc, nc), entry_type=Fraction)
        for i in range(len(classes)):
            for j in range(len(classes)):
                for i_sym in classes[class_list[i]]:
                    for j_sym in classes[class_list[j]]:
                        prod = loc_to_symbol[table[symbol_to_loc[i_sym]][symbol_to_loc[j_sym]]]
                        cl = symbol_to_class_rep[prod]
                        k = class_linear_index[cl]
                        struct_constants_in_fractions.s(
                            (i, j, k), struct_constants_in_fractions.g((i, j, k)) + Fraction(1, len(classes[cl]))
                        )

        _, e = int_list_gcd_and_lcd(orders)
        p_min = 2*ceil(len(orders)**0.5)
        p = find_prime(p_min, filter=lambda x: (x-1) % e == 0)
        _logger.info(f"LCD of orders = {e}")
        _logger.info(f"Need prime field of characteristic p >= {p_min} with p = 1 mod {e}")
        _logger.info(f"Using prime field of characteristic {p}")
        F = PrimeField.create(p=p)
        P = Poly.create(domain=F)
        struct_constants: Tensor = Tensor(shape=(nc, nc, nc), entry_type=F)
        for idx in struct_constants.idx():
            n = int(struct_constants_in_fractions.g(idx))
            struct_constants.s(idx, F(n))

        class NewGroupAsTable(GroupAsTable):
            pass

        NewGroupAsTable.name = name
        NewGroupAsTable.table = table
        NewGroupAsTable.loc_to_symbol = loc_to_symbol
        NewGroupAsTable.symbol_to_loc = symbol_to_loc
        NewGroupAsTable.inverses = inverses
        NewGroupAsTable.orders = orders
        NewGroupAsTable.classes = classes
        NewGroupAsTable.symbol_to_class_rep = symbol_to_class_rep
        NewGroupAsTable.struct_constants = struct_constants
        NewGroupAsTable.unit = NewGroupAsTable(NewGroupAsTable.loc_to_symbol[0])

        class_root_to_pow_maps: List[Dict[F, int]] = []
        for i in range(len(classes)):
            li = []
            for j in range(struct_constants._shape[0]):
                li.append([])
                for k in range(struct_constants._shape[0]):
                    li[j].append(struct_constants[i, j, k])
            t = Tensor(li, entry_type=F)
            cp = P(characteristic_poly(t))
            factor_to_pow_map = factor(cp)
            s = ""
            roots: Dict[PrimeField, int] = dict()
            for fp, mult in factor_to_pow_map.items():
                s += f"({fp})**{mult} "
                if fp.rank() > 2:
                    raise Exception("Error: characteristic polynomial of structure matrix eigenproblem is not reducible: {fp}")
                roots[-fp.coefficient_list[0]] = mult
            _logger.info(f"Characteristic polynomial #{i} -> {cp} === {s}")
            _logger.info(f"Eigenvalues -> {' | '.join([str(r) for r in roots])}")
            class_root_to_pow_maps.append(roots)

        NewGroupAsTable.character_table = NewGroupAsTable.calc_group_characters_from_struct_matrix(class_root_to_pow_maps, e)

        return NewGroupAsTable

    @property
    def sym(self):
        return self._sym

    def __init__(self, sym: Hashable):
        self._sym = sym

    def __str__(self):
        return str(self._sym)

    def __mul__(self, other: "GroupAsTable") -> "GroupAsTable":
        l1 = self.symbol_to_loc[self._sym]
        l2 = other.symbol_to_loc[other._sym]
        return type(self)(self.loc_to_symbol[self.table[l1][l2]])

    def invert(self) -> "GroupAsTable":
        return type(self)(self.inverses[self._sym])

    @classmethod
    def print_classes(cls):
        m = max([len(rep) for rep in cls.classes])
        tag = "Class"
        m = max(m, len(tag))
        des = f"{tag.ljust(m)} | Elements"
        sarr = []
        for rep, cla in cls.classes.items():
            ljust_rep, all_cla = rep.ljust(m), ', '.join(cla)
            sarr.append(f"{ljust_rep} | {all_cla}")
            print()
        M = max([len(s) for s in sarr])
        M = max(M, len(des))
        print(des)
        print("-"*M)
        for s in sarr:
            print(s)
        print()

    @classmethod
    def calc_group_characters_from_struct_matrix(group, root_map: Dict[FieldType, int], lcd_of_orders: int) -> Tensor:
        e = lcd_of_orders
        s = group.struct_constants
        F = s.entry_type

        # Generate the cyclotomic field for the final representation of the characters
        QP, _ = calc_cyclotomic_field(n=e, p=0)
        W = find_prime_field_element_with_order(e=e, p=F.prime)
        _logger.info(f"Representing in cyclotomic extension of Q with roots of x**{e} - 1 of dimension φ({e}) = {QP.irreducible_poly.rank()-1}")
        _logger.info(f"Cyclotomic polynomial = {QP.irreducible_poly}")
        _logger.info(f"Generator in number field = {W} ({W}**{e-1} = 1 mod {F.prime}")

        cl_sizes = [len(cl) for _, cl in group.classes.items()]
        cln = len(cl_sizes)
        cl_sizes_in_F = [F(cls) for cls in cl_sizes]
        classes_as_list: List[Tuple[str, Set[str]]] =\
            [(rep, cls_set) for rep, cls_set in group.classes.items()]
        class_rep_to_idx_map: Dict[str, int] = \
            {rep: i for i, (rep, cls_set) in enumerate(group.classes.items())}
        g = sum(cl_sizes)
        e_inv_in_F = F(e) ** -1

        # Calculate the normalized eigen-vectors of the structure matrices in the number field
        V: Set[Tuple[FieldType, ...]] = set()
        calc_single_group_char(root=None, mult=None, depth=0,
                               s=s, list_of_root_enum=root_map, char_set=V)

        L = list(V)
        for i, vec in enumerate(V):
            _logger.info(f"Unit vector #{i+1} -> "+" | ".join([str(v) for v in vec]))

        # Character table: X[r, c] is the character of "c" class is representation "r"
        X = Tensor(shape=(cln, cln), entry_type=QP)

        # The Fourier coefficients of the character expansion:
        # X[r, c] = Σ_k m[r, c, k] x**k, x**e = 1
        m: Tensor = Tensor(shape=(cln, cln, e), entry_type=F)

        for i in range(cln):
            # Calculate the characters of the i-th representation:

            # Make sure eigen-vector is normalized
            no = L[i][0]**-1
            L[i] = [L[i][k]*no for k in range(cln)]

            # Find dimension of representation
            su = F.zero
            for k in range(len(L[i])):
                cls_rep = classes_as_list[k][0]
                cls_rep_inv = group.inverses[cls_rep]
                cls_of_inv = class_rep_to_idx_map[group.symbol_to_class_rep[cls_rep_inv]]
                su += L[i][k] * L[i][cls_of_inv] * cl_sizes_in_F[k]**-1

            x2 = F(g) * su**-1
            xint = int_root_modulo(y=x2.value, p=F.prime)
            x = F(min(xint, F.prime - xint))


            # Find the map of the characters in Fp
            Char = [L[i][k] * x * cl_sizes_in_F[k]**-1 for k in range(len(L[i]))]

            # Lift from Fp to Z[x] where x**e = 1 % p
            char_str_list: str = []
            for j in range(cln):
                        # Calculate Fourier coefficients m[i, j, k]
                        cls_rep = classes_as_list[j][0]
                        cls_rep_loc = group.symbol_to_loc[cls_rep]
                        for k in range(e):
                            identity_class_loc = 0
                            identity_class_rep = classes_as_list[identity_class_loc][0]
                            ne_elem_loc = group.symbol_to_loc[identity_class_rep]
                            for l in range(e):
                                ne_elem = group.loc_to_symbol[ne_elem_loc]
                                ne_rep = group.symbol_to_class_rep[ne_elem]
                                ne_cls_idx = class_rep_to_idx_map[ne_rep]   # j[l] = class of inverse of l-th class rep
                                power = (-k*l) % e
                                X0 = Char[ne_cls_idx]*e_inv_in_F*(W**power)   # Θ(X[i, j[l]]) / Θ(e)
                                m.s((i, j, k), m.g((i, j, k)) + X0)
                                ne_elem_loc = group.table[cls_rep_loc][ne_elem_loc]

                        # Obtain character X[i, j] from Fourier coefficients m[i, j, k]
                        Char_i_j = QP([Fraction(0, 1)])
                        for k in range(e):
                            PXC = [Fraction(0, 1)] * e
                            PXC[k] = Fraction(1, 1)
                            M = QP(PXC)
                            v = m.g((i, j, k)).value
                            Char_i_j += M * QP([Fraction(v, 1)])
                        X.s((i, j), Char_i_j)
                        char_str_list.append(str(Char_i_j))
            _logging.info(f"Rep #{i} (dim = {x}): " + "  |  ".join(char_str_list))

            # Check orthogonality with previous representations
            for j in range(i):
                        pro = QP.zero
                        for k in range(cln):
                            cls_rep = classes_as_list[k][0]
                            cls_rep_inv = group.inverses[cls_rep]
                            cls_of_inv = class_rep_to_idx_map[group.symbol_to_class_rep[cls_rep_inv]]
                            pro += X.g((i, k)) * X.g((j, cls_of_inv)) * cl_sizes[k]
                        status = chr(0x2714) if pro == pro.zero else chr(0x2718)
                        if pro != pro.zero:
                            _logging.error(f"Orthogonality product {i} x {j} = {pro} {status}")
                            raise Exception("Fail ! Characters are not orthogonal")
                        else:
                            _logging.info(f"Orthogonality product {i} x {j} = {pro} {status}")
        return X

    @classmethod
    def print_characters(group) -> None:
        classes_as_list: List[Tuple[str, Set[str]]] = \
            [(rep, cls_set) for rep, cls_set in group.classes.items()]
        X = group.character_table
        cyclotomic_poly = X.entry_type.irreducible_poly
        # Symbol table
        symbols: Dict[PolyField, str] = dict()
        idx_sy, max_sy_si = 0, 0
        for idx in X.idx():
            c = X.g(idx)
            if c.poly.rank() > 1:
                if c not in symbols:
                    sy = gen_unique_alphanum_symbol(idx_sy)
                    max_sy_si = max(max_sy_si, len(sy))
                    symbols[c] = sy
                    idx_sy += 1
        if symbols:
            print("Cyclotomic polynomial:")
            print()
            print(f"{cyclotomic_poly} = 0")
            print()
            print("Symbol table:")
            print()
            for c, sy in symbols.items():
                print(f"{sy} = {c}")
            print()

        rows = [[] for _ in range(X.shape[1])]
        max_sis = [len(classes_as_list[cl][0]) for cl in range(X.shape[0])]
        for cl in range(X.shape[1]):
            max_si = max_sis[cl]
            for rep in range(X.shape[0]):
                po = X.g((rep, cl))
                s = symbols[po] if (po in symbols) \
                    else str(po.poly.coefficient_list[0] if (len(po.poly.coefficient_list) > 0) else 0)
                max_si = max(max_si, len(s))
                rows[rep].append(s)
            max_sis[cl] = max_si
            for rep in range(X.shape[0]):
                rows[rep][-1] = rows[rep][-1].rjust(max_sis[cl])
        top = [classes_as_list[cl][0].rjust(max_sis[cl]) for cl in range(X.shape[0])]
        print("Character table:")
        print()
        print(" | ".join(top))
        for rep in range(X.shape[0]):
            print(" | ".join(rows[rep]))
        print()
        return X


def gen_unique_alphanum_symbol(idx: int) -> str:
    remain_idx, sy = idx, ""
    while True:
        remain_idx, code = divmod(remain_idx, 24)
        sy += chr(97 + code)
        if remain_idx == 0:
            break
    return sy


def calc_single_group_char(root: Optional[FieldType],
                           mult: Optional[int],
                           depth: int,
                           s: Tensor,
                           list_of_root_enum: List[Dict[FieldType, int]],
                           basis: Optional[Tensor]=None,
                           char_set: Set[Poly]=set()):
    se = {root: mult} if (depth > 0) else list_of_root_enum[0]
    for ro, mu in se.items():
        A = s[depth, :, :]
        if basis is not None:
            # depth > 0
            A = A @ basis - basis * ro
            V = eigen_vector_calc(A, [A.zero_entry], ortho=False)
            if V is not None:
                if V.dim > 1:
                    V = V[:basis.shape[1], :]
                else:
                    # scalar
                    V = V[:basis.shape[1]]
        else:
            # depth = 0
            V = eigen_vector_calc(A, [ro], ortho=False)
        if V is None:
            continue
        if len(V.shape) == 1 or V.shape[1] == 0:
            if basis is not None:
                V = basis @ V
            # Single vector found => done
            V = V*(V[0]**-1)
            char_set.add(tuple([V[i] for i in V.idx()]))
            return
        elif V.shape[1] == 1:
            if basis is not None:
                V = basis @ V
            # Found single vector => done !
            V = V*(V[0, 0]**-1)
            char_set.add(tuple([V[i] for i in V.idx()]))
            return
        else:
            if basis is not None:
                V = basis @ V
            # More than one vector => need to narrow it down
            if depth+1 < len(list_of_root_enum):
                for r, m in list_of_root_enum[depth+1].items():
                    calc_single_group_char(r, m, depth+1, s, list_of_root_enum, basis=V, char_set=char_set)


# Sample group tables
Cyclic3MultTable = [
    ['E', 'A', 'A2'],
    ['A', 'A2', 'E'],
    ['A2', 'E', 'A']
]

Cyclic4MultTable = [
    ['E', 'A', 'A2', 'A3'],
    ['A', 'A2', 'A3', 'E'],
    ['A2', 'A3', 'E', 'A'],
    ['A3', 'E', 'A', 'A2']
]

Cyclic5MultTable = [
    ['E', 'A', 'A2', 'A3', 'A4'],
    ['A', 'A2', 'A3', 'A4', 'E'],
    ['A2', 'A3', 'A4', 'E', 'A'],
    ['A3', 'A4', 'E', 'A', 'A2'],
    ['A4', 'E', 'A', 'A2', 'A3']
]

Cyclic6MultTable = [
    ['E', 'A', 'A2', 'A3', 'A4', 'A5'],
    ['A', 'A2', 'A3', 'A4', 'A5', 'E'],
    ['A2', 'A3', 'A4', 'A5', 'E', 'A'],
    ['A3', 'A4', 'A5', 'E', 'A', 'A2'],
    ['A4', 'A5', 'E', 'A', 'A2', 'A3'],
    ['A5', 'E', 'A', 'A2', 'A3', 'A4'],
]

Cyclic7MultTable = [
    ['E', 'A', 'A2', 'A3', 'A4', 'A5', 'A6'],
    ['A', 'A2', 'A3', 'A4', 'A5', 'A6', 'E'],
    ['A2', 'A3', 'A4', 'A5', 'A6', 'E', 'A'],
    ['A3', 'A4', 'A5', 'A6', 'E', 'A', 'A2'],
    ['A4', 'A5', 'A6', 'E', 'A', 'A2', 'A3'],
    ['A5', 'A6', 'E', 'A', 'A2', 'A3', 'A4'],
    ['A6', 'E', 'A', 'A2', 'A3', 'A4', 'A5']
]

QuartenionMultTable = [
    ['E', 'EM', 'I', 'IM', 'J', 'JM', 'K', 'KM'],
    ['EM', 'E', 'IM', 'I', 'JM', 'J', 'KM', 'K'],
    ['I', 'IM', 'EM', 'E', 'K', 'KM', 'JM', 'J'],
    ['IM', 'I', 'E', 'EM', 'KM', 'K', 'J', 'JM'],
    ['J', 'JM', 'KM', 'K', 'EM', 'E', 'I', 'IM'],
    ['JM', 'J', 'K', 'KM', 'E', 'EM', 'IM', 'I'],
    ['K', 'KM', 'J', 'JM', 'IM', 'I', 'EM', 'E'],
    ['KM', 'K', 'JM', 'J', 'I', 'IM', 'E', 'EM']
]

# Dihedral (D4) group: non-isomorphic to the Quartention but has the same character table
DihedralMultTable = [
    ['E', 'R', 'R1', 'R2', 'H', 'V', 'D', 'D1'],
    ['R', 'R1', 'R2', 'E', 'D', 'D1', 'V', 'H'],
    ['R1', 'R2', 'E', 'R', 'V', 'H', 'D1', 'D'],
    ['R2', 'E', 'R', 'R1', 'D1', 'D', 'H', 'V'],
    ['H', 'D1', 'V', 'D', 'E', 'R1', 'R2', 'R'],
    ['V', 'D', 'H', 'D1', 'R1', 'E', 'R', 'R2'],
    ['D', 'H', 'D1', 'V', 'R', 'R2', 'E', 'R1'],
    ['D1', 'V', 'D', 'H', 'R2', 'R', 'R1', 'E'],
]

C3hMultTable = [
    ['E',      'C3',     'C3x2',   'S',     'C3xS',   'C3x2xS'],
    ['C3',     'C3x2',   'E',      'C3xS',  'C3x2xS', 'S'],
    ['C3x2',   'E',      'C3',     'C3x2xS', 'S',     'C3xS'],
    ['S',      'C3xS',   'C3x2xS', 'E',      'C3',    'C3x2'],
    ['C3xS',   'C3x2xS', 'S',      'C3',     'C3x2',  'E'],
    ['C3x2xS', 'S',      'C3xS',   'C3x2',   'E',     'C3'],
]

C3vMultTable = [
    ['I', 'a', 'b', 'c', 'd', 'e'],
    ['a', 'b', 'I', 'd', 'e', 'c'],
    ['b', 'I', 'a', 'e', 'c', 'd'],
    ['c', 'd', 'd', 'I', 'b', 'a'],
    ['d', 'c', 'e', 'a', 'I', 'b'],
    ['e', 'd', 'c', 'b', 'a', 'I'],
]


def run_samples(log_level=_logging.ERROR):
    _logging.basicConfig(level=log_level)
    groups = (
        ("Cyclic 3", Cyclic3MultTable),
        ("Cyclic 4", Cyclic4MultTable),
        ("Cyclic 5", Cyclic5MultTable),
        ("Cyclic 6", Cyclic6MultTable),
        ("Cyclic 7", Cyclic7MultTable),
        ("Quartenion", QuartenionMultTable),
        ("Dihedral (D4)", DihedralMultTable),
        ("C3h", C3hMultTable),
        ("C3v", C3vMultTable)
    )
    for name, table in groups:
        print("**************************************************************************************")
        print(name)
        print()
        GroupAsTable.create(table, name=name).print_characters()
