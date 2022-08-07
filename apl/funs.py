"""
funs.py

Implementations of the APL built-ins:

⍴ - rho()
⍉ - transpose()

This file contains doctests.

To run the doctests, do:

    python funs.py [-v]
"""
from collections import defaultdict
from math import prod
from typing import Optional

from apl.arr import Array, NYIError, RankError, LengthError, Scalar, Vector, coords, decode, enclose, select, SimpleScalar

class RankError(Exception):
    pass

def rotate_first(alpha: list[int], omega: Array) -> Array:
    pass

def rho(*, alpha: Optional[Array] = None, omega: Array) -> Array:
    """
    Monadic: shape
    Dyadic: reshape

    Apply the shape alpha to the ravel of omega. The rank of alpha must be 0 or 1

    APL> ,5 5⍴3 3⍴3 2 0 8 6 5 4 7 1
    3 2 0 8 6 5 4 7 1 3 2 0 8 6 5 4 7 1 3 2 0 8 6 5 4

    >>> rho(alpha=Vector([5, 5]), omega=Array([3, 3], [3, 2, 0, 8, 6, 5, 4, 7, 1])).data
    [3, 2, 0, 8, 6, 5, 4, 7, 1, 3, 2, 0, 8, 6, 5, 4, 7, 1, 3, 2, 0, 8, 6, 5, 4]

    >>> rho(alpha=Vector([2, 2]), omega=Scalar(5)).data
    [5, 5, 5, 5]

    >>> rho(omega=Array([2, 2], [1, 2, 3, 4])).data
    [2, 2]
    """

    if alpha is None: # Monadic
        return Array([omega.bound], omega.shape)
    if alpha.rank > 1:
        raise RankError

    # Dyadic. Omega can be a scalar, so we need to ensure that we extend that
    data = [omega.data] if omega.rank == 0 else omega.data

    return Array(alpha.data, [data[idx%len(data)] for idx in range(prod(alpha.data))])


def _reorder_axes(spec:list[int], coords:list[int]) -> list[int]:
    """
    Given an axis reorder spec as given by the left arg to dyadic transpose,
    transform a coordinate vector to the new basis. This is at the root of
    what makes dyadic transpose hard to grasp. The spec is NOT the new shape,
    it's the *grade vector*.
    """
    uniques = len(set(spec))
    pairs = zip(spec, coords)
    if len(spec) == uniques: # No repeated axes
        return [i[-1] for i in sorted(pairs)]
    
    # We have repeated axes, meaning that the rank will be the number of
    # unique axes in the spec list.
    transformed = [-1]*uniques

    # Replace any pairs where the first index is the same with min
    for i, c in pairs:
        if transformed[i] == -1:
            transformed[i] = c
        else:
            transformed[i] = min(transformed[i], c)

    return transformed

def _repeated(spec: list[int]) -> dict[int,list[int]]:
    """
    Return a dictionary where the keys are the unique axis indices 
    of axes that are repeated, and the values are lists of the 
    corresponding locations.

    >>> _repeated([0, 1, 0])
    {0: [0, 2]}

    >>> _repeated([0, 1, 2])
    {}

    >>> _repeated([0, 1, 0, 1])
    {0: [0, 2], 1: [1, 3]}

    >>> _repeated([])
    {}
    """
    unique = defaultdict(list)
    for idx, elem in enumerate(spec):
        unique[elem].append(idx)

    # Only keep entries for the repeated axes
    return {k: v for k, v in unique.items() if len(v) > 1}

def _skip(rax: dict[int, list[int]], coord: list[int]) -> bool:
    """
    Given a dictionary showing the locations of repeated axes, 
    show if the given coordinate vector should be skipped, i.e
    it must have equality where axes are repeated.

    >>> _skip({0:[0, 2]}, [1, 0, 1]) # No skipping; axes 0 and 2 are equal
    False

    >>> _skip({0:[0, 2]}, [1, 9, 1]) # No skipping; axes 0 and 2 are equal
    False

    >>> _skip({0:[0, 2]}, [1, 2, 3]) # Skipping; axes 0 and 2 are not equal
    True
    """
    for i in range(len(coord)):
        for m in rax.get(i, []):
            if coord[m] != coord[i]:
                return True
    return False

def transpose(*, alpha: Optional[Array]=None, omega: Array) -> Array:
    """
    Monadic and dyadic transpose. Dyadic transpose is a generalisation of
    the monadic case, which reverses the axes. In the dyadic case, axes can
    be in any order. 
    
    The general dyadic transpose: reorder axes

    APL> ,2 0 1⍉2 3 4⍴⍳×/2 3 4
    ┌→────────────────────────────────────────────────────────────┐
    │0 12 1 13 2 14 3 15 4 16 5 17 6 18 7 19 8 20 9 21 10 22 11 23│
    └~────────────────────────────────────────────────────────────┘

    >>> a = Array([2, 3, 4], list(range(2*3*4))); transpose(alpha=Vector([2, 0, 1]), omega=a).data
    [0, 12, 1, 13, 2, 14, 3, 15, 4, 16, 5, 17, 6, 18, 7, 19, 8, 20, 9, 21, 10, 22, 11, 23]

    Complication: repeated axes in the dyadic case:

    APL> 0 0⍉3 3⍴⍳9 ⍝ Dyadic transpose with repeated axes gets the diagonal
    ┌→────┐
    │0 4 8│
    └~────┘

    >>> a = Array([3, 3], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]); transpose(alpha=Vector([0, 0]), omega=a).data
    [0, 4, 8]
    """
    shape = omega.shape
    if omega.rank < 2: # For scalars and vectors: identity
        return Array(shape, omega.data)

    # For the monadic transpose: transformation is the reverse axis ordering
    repeated_axes = {}
    if alpha is None:
        alpha = Vector(list(range(omega.rank-1, -1, -1)))
    else:
        if alpha.rank > 1:
            raise RankError

        # Check for repeated axes in the axis reordering spec. 
        repeated_axes = _repeated(alpha.data)

    new_shape = _reorder_axes(alpha.data, shape)
    newdata = [0]*prod(new_shape)
    idx = 0 # index into ravel of omega
    for cvec in coords(shape):
        if repeated_axes and _skip(repeated_axes, cvec):
            idx += 1
            continue
        newdata[decode(new_shape, _reorder_axes(alpha.data, cvec))] = omega.data[idx]
        idx += 1 

    return Array(new_shape, newdata)

def iota(*, alpha: Optional[Array]=None, omega: Array, IO: int = 0) -> Array:
    """
    Monadic: the APL range function, which can apply to any shape. Note that
    space and time grows very quick with the rank.

    Dyadic: index-of (nyi)

    APL> ⍳5
    0 1 2 3 4

    >>> iota(omega=Scalar(5)).data
    [0, 1, 2, 3, 4]

    APL> ⍳2 3
    ┌───┬───┬───┬───┬───┬───┐
    │0 0│0 1│0 2│1 0│1 1│1 2│
    └───┴───┴───┴───┴───┴───┘

    >>> [a.data for a in iota(omega=Vector([2, 3])).data]
    [[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2]]
    """
    if alpha is not None:
        raise NYIError

    shape = omega.data
    if omega.rank == 0:
        return Vector(list(range(IO, shape+IO)))
    return Array(shape, [Vector(cvec) for cvec in coords(shape, IO)])

def drop(*, alpha: int, omega: Array) -> Array:
    """
    Drop alpha elements from omega. Pos drops from beginning, neg from the end.

    Note: rank 1 only for now.

    >>> drop(alpha=2, omega=Vector([1, 2, 3, 4]))
    Vector([3, 4])
    """
    if omega.rank != 1:
        raise NYIError("↓ is only implemented for vectors")
    
    if alpha>0:
        return Vector(omega.data[alpha:])
    return Vector(omega.data[:-alpha])
        
def squad(*, alpha: Optional[Array], omega: Array) -> Array:
    """
    Mondadic: materialise (NYI)
    Dyadic: index

    Case 1: 1 2 3⌷A ⍝ A is rank 3
    Pick major cells recursively L-R

    >>> squad(alpha=Vector([1]), omega=Array([2, 2], [1, 2, 3, 4]))
    Vector([3, 4])
    """

    if alpha is None:
        raise NYIError('materialise is not yet implemented')

    if alpha.bound == 0: # How distinguish between scalar and len 1 vector?
        return omega

    cell = select(omega, [alpha.data[0]]) # Returns list of cells
    if alpha.rank == 0:
        return cell[alpha.data[0]]
    
    next_axis = drop(alpha=1, omega=alpha)
    return squad(alpha=next_axis, omega=cell)


if __name__ == "__main__":
    # To run the doctests (verbosely), do 
    #
    # python apl/funs.py -v
    #
    # See: https://docs.python.org/3/library/doctest.html
    import doctest
    doctest.testmod()