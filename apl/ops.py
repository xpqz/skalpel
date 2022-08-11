"""
ops.py

Implementations of the APL built-in primitive operators

/, ⌿ - reduce()

This file contains doctests.

To run the doctests, do:

    python ops.py [-v]
"""
from dataclasses import dataclass
from math import prod
from typing import Callable, Optional

from apl.arr import Array, RankError, Scalar
from apl.errors import ArityError, ValueError
from apl.funs import FUNS, Arity


@dataclass(frozen=True)
class Operator:
    f: Callable
    derives: Arity                  # Arity of derived function
    arity: Arity                    # Am I a monadic or a dyadic operator?
    left: Arity                     # Arity of left operand
    right: Optional[Arity] = None   # Arity of right operand, if I am dyadic


def derive(operator: Callable, left: Callable|str, right: Optional[Callable|str], arity: Arity) -> Callable:
    if arity == Arity.MONAD:
        def derived_monad(omega: Array) -> Array:
            return operator(left, right, None, omega)
        return derived_monad
    def derived_dyad(alpha: Array, omega: Array) -> Array:
        return operator(left, right, alpha, omega)
    return derived_dyad

def commute(left: Callable, right: Optional[Callable], alpha: Array, omega: Array) -> Array:
    """
    Outward-facing 'A f⍨ B' (commute)
    """
    if right is not None:
        raise ArityError("'⍨' takes no right operand")

    return left(omega, alpha) # swap argument order

def over(left: Callable, right: Optional[Callable], alpha: Array, omega: Array) -> Array:
    """
    Outward-facing '⍥'
    """
    if right is None:
        raise ArityError("'⍥' takes a right operand")

    return left(right(alpha), right(omega))

def reduce(left: Callable|str, right: Optional[Callable|str], alpha: Optional[Array], omega: Array) -> Array:
    """
    Outward-facing '/' (trailling axis reduce)
    """
    if right is not None:
        raise ArityError("'/' takes no right operand")

    if alpha is not None:
        raise ArityError("function derived by '/' takes no left argument")

    return _reduce(operand=left, axis=omega.rank-1, omega=omega)


def reduce_first(left: Callable|str, right: Optional[Callable|str], alpha: Optional[Array], omega: Array) -> Array:
    """
    Outward-facing '⌿' (leading axis reduce)
    """
    if right is not None:
        raise ArityError("'⌿' takes no right operand")

    if alpha is not None:
        raise ArityError("function derived by '⌿' takes no left argument")

    return _reduce(operand=left, axis=0, omega=omega)


def _reduce(*, operand: Callable|str, axis: int, omega: Array) -> Array:
    """
    [private]

    Reduction along axis. 
    
    Originally by @ngn: 
       * https://chat.stackexchange.com/transcript/message/47158587#47158587
       * https://pastebin.com/mcPAxtqx

    A version also used in dzaima/apl:
       * https://github.com/dzaima/APL/blob/master/src/APL/types/functions/builtins/mops/ReduceBuiltin.java#L156-L178

    >>> _reduce(operand=lambda x, y:x+y, axis=0, omega=Array([2, 2], [1, 2, 3, 4])).data
    [4, 6]

    >>> _reduce(operand:lambda x, y:x+y, axis=1, omega=Array([2, 2], [1, 2, 3, 4])).data
    [3, 7]
    """
    if omega.rank == 0:
        return omega
    if axis < 0:
        axis += omega.rank

    if axis >= omega.rank:
        raise RankError

    if isinstance(operand, str):
        if operand not in FUNS:
            raise ValueError(f"VALUE ERROR: Undefined operand: '{operand}'")
        fn = FUNS[operand][1]
        if fn is None:
            raise ArityError(f"function {operand} must be dyadic")
    else:
        fn = operand

    # Some vector + built-in special cases
    if omega.rank == 1 and isinstance(operand, str):
        match operand:
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
                acc = fn(omega.data[i*n1*n2 + j*n2 + k], acc)
            ravel[i*n2 + k] = acc

    return Array(shape, ravel)

OPERATORS = {
    #-------------implem--------derived-isa--self-isa-----L-isa-------R-isa
    '/': Operator(reduce,       Arity.MONAD, Arity.MONAD, Arity.DYAD, None), 
    '⌿': Operator(reduce_first, Arity.MONAD, Arity.MONAD, Arity.DYAD, None),
    '⍨': Operator(commute,      Arity.DYAD,  Arity.MONAD, Arity.DYAD, None),
    '⍥': Operator(over,         Arity.DYAD,  Arity.DYAD,  Arity.DYAD, Arity.MONAD)
}

if __name__ == "__main__":
    # To run the doctests (verbosely), do 
    #
    # python apl/funs.py -v
    #
    # See: https://docs.python.org/3/library/doctest.html
    import doctest
    doctest.testmod()
