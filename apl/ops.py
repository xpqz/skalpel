"""
ops.py

Implementations of the APL built-in primitive operators

/, ⌿ - reduce()

This file contains doctests.

To run the doctests, do:

    python ops.py [-v]
"""
from collections import defaultdict
from math import prod
from typing import Callable

from apl.arr import Array, RankError, Scalar, ValueError
from apl.funs import SCALAR_DYADS

def reduce(omega: Array, axis: int, f: Callable|str) -> Array:
    """
    Reduction along axis. 
    
    Originally by @ngn: 
       * https://chat.stackexchange.com/transcript/message/47158587#47158587
       * https://pastebin.com/mcPAxtqx

    A version also used in dzaima/apl:
       * https://github.com/dzaima/APL/blob/master/src/APL/types/functions/builtins/mops/ReduceBuiltin.java#L156-L178

    >>> reduce(Array([2, 2], [1, 2, 3, 4]), 0, lambda x, y:x+y).data
    [4, 6]

    >>> reduce(Array([2, 2], [1, 2, 3, 4]), 1, lambda x, y:x+y).data
    [3, 7]
    """
    if omega.rank == 0:
        return omega
    if axis < 0:
        axis += omega.rank

    if axis >= omega.rank:
        raise RankError

    if isinstance(f, str):
        if f not in SCALAR_DYADS:
            raise ValueError(f"VALUE ERROR: Undefined name: '{str}'")
        fn = SCALAR_DYADS[f]
    else:
        fn = f

    # Some vector + built-in special cases
    if omega.rank == 1 and isinstance(f, str):
        match f:
            case '+':
                return Scalar(sum(omega.data))
            case '×':
                return Scalar(prod(omega.data))
            case '⌊':
                return Scalar(min(omega.data))
            case '⌈':
                return Scalar(max(omega.data))

    n0 = prod(omega.shape[:axis])             # prouct of dims before axis
    n1 = omega.shape[axis]                    # reduction axis size
    n2 = prod(omega.shape[axis+1:omega.rank]) # product of dims after axis

    shape = omega.shape[:]; del shape[axis]   # new shape is old shape with the reducing axis removed
    ravel = [0 for _ in range(n0*n2)]

    for i in range(n0):
        for k in range(n2):
            acc = omega.data[i*n1*n2 + (n1-1)*n2 + k]
            for j in range(n1-2, -1, -1):     # R-L
                acc = f(omega.data[i*n1*n2 + j*n2 + k], acc)
            ravel[i*n2 + k] = acc

    return Array(shape, ravel)
    
if __name__ == "__main__":
    # To run the doctests (verbosely), do 
    #
    # python apl/funs.py -v
    #
    # See: https://docs.python.org/3/library/doctest.html
    import doctest
    doctest.testmod()