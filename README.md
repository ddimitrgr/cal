# CAL
Algebra library for calculating characters (traces) of matrix representations of groups.
## Installation

```bash
pip install git+https://github.com/ddimitrgr/cal.git@master
```

## Usage

For example to calculate the character table for the quartenion group:

```python
import cal.GroupAsTable

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
GroupAsTable.create(QuartenionMultTable).print_characters()
```

and we get:
```
Character table:

E | EM |  I |  J |  K
1 |  1 |  1 | -1 | -1
1 |  1 | -1 |  1 | -1
1 |  1 | -1 | -1 |  1
1 |  1 |  1 |  1 |  1
2 | -2 |  0 |  0 |  0
```

Similarly to calculate for the cyclic group of order 5:
```python
Cyclic5MultTable = [
    ['E', 'A', 'A2', 'A3', 'A4'],
    ['A', 'A2', 'A3', 'A4', 'E'],
    ['A2', 'A3', 'A4', 'E', 'A'],
    ['A3', 'A4', 'E', 'A', 'A2'],
    ['A4', 'E', 'A', 'A2', 'A3']
]

GroupAsTable.create(Cyclic5Table).print_characters()
```

and we obtain the characters in terms of algebraic numbers:
```
Cyclotomic polynomial:

(1) x**4 + (1) x**3 + (1) x**2 + (1) x**1 + (1) = 0

Symbol table:

a = (1) x**2 + (0) x**1 + (0)
b = (-1) x**3 + (-1) x**2 + (-1) x**1 + (-1)
c = (1) x**1 + (0)
d = (1) x**3 + (0) x**2 + (0) x**1 + (0)

Character table:

E | A | A2 | A3 | A4
1 | a |  b |  c |  d
1 | c |  a |  d |  b
1 | b |  d |  a |  c
1 | d |  c |  b |  a
1 | 1 |  1 |  1 |  1
```
