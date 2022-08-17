from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
from math import prod
import operator
from typing import Callable, Optional, Sequence, TypeAlias

from apl.arr import Array, A, S, V, coords, decode, issimple, match
from apl.errors import ArityError, DomainError, NYIError, RankError

Signature: TypeAlias = tuple[Optional[Callable], Optional[Callable]]

class Arity(Enum):
    MONAD=0
    DYAD=1

@dataclass(frozen=True)
class Operator:
    f: Callable
    derives: Arity                  # Arity of derived function
    arity: Arity                    # Am I a monadic or a dyadic operator?
    left: Arity                     # Arity of left operand
    right: Optional[Arity] = None   # Arity of right operand, if I am dyadic

def pervade(f: Callable) -> Callable:
    """
    Pervade a simple function f into either arguments of equal shapes, 
    or between a scalar and an argument of any shape.

    f must have a signature of fn(alpha: int|float, omega: int|float) -> int|float
    """
    def pervaded(alpha: Array, omega: Array) -> Array:
        """
        Case 0: 1 + 2           # both scalar
        Case 1: 1 + 1 2 3       # left is scalar
        Case 2: 1 2 3 + 1       # right is scalar
        Case 3: 1 2 3 + 4 5 6   # equal shapes
        Case 4: ....            # rank error
        """
        if issimple(alpha) and issimple(omega):       # Case 0: both scalar
            return S(f(alpha.data[0], omega.data[0]))

        if issimple(alpha):                           # Case 1: left is scalar
            return A(omega.shape, list(map(lambda x: pervaded(alpha, x), omega.data)))

        if issimple(omega):                           # Case 2: right is scalar
            return A(alpha.shape, list(map(lambda x: pervaded(x, omega), alpha.data)))

        if alpha.shape == omega.shape:                # Case 3: equal shapes
            data = list(map(lambda x: pervaded(alpha.data[x], omega.data[x]), range(alpha.bound)))
            return A(alpha.shape, data)
        raise RankError                               # Case 4: unequal shapes; rank error

    return pervaded

def commute(left: Callable|str, right: Optional[Callable|str], alpha: Array, omega: Array) -> Array:
    """
    Outward-facing 'A f⍨ B' (commute)
    """
    if right is not None:
        raise ArityError("'⍨' takes no right operand")
    
    fn = Voc.get_fn(left, Arity.DYAD)

    return fn(omega, alpha) # swap argument order

def over(left: Callable|str, right: Optional[Callable|str], alpha: Array, omega: Array) -> Array:
    """
    Outward-facing '⍥'
    """
    if right is None:
        raise ArityError("'⍥' takes a right operand")

    fn = Voc.get_fn(left, Arity.DYAD)
    pp = Voc.get_fn(right, Arity.MONAD)

    return fn(pp(alpha), pp(omega))

def _reorder_axes(spec:Sequence[int], coords:Sequence[int]) -> list[int]:
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

def _repeated(spec: Sequence[int]) -> dict[int,list[int]]:
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

def _skip(rax: dict[int, list[int]], coord: Sequence[int]) -> bool:
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

def transpose(alpha: Sequence[int], omega: Array) -> Array:
    """
    Monadic and dyadic transpose. Dyadic transpose is a generalisation of
    the monadic case, which reverses the axes. In the dyadic case, axes can
    be in any order. 
    
    The general dyadic transpose: reorder axes

    APL> ,2 0 1⍉2 3 4⍴⍳×/2 3 4
    ┌→────────────────────────────────────────────────────────────┐
    │0 12 1 13 2 14 3 15 4 16 5 17 6 18 7 19 8 20 9 21 10 22 11 23│
    └~────────────────────────────────────────────────────────────┘

    >>> a = A([2, 3, 4], list(range(2*3*4)))
    >>> at = transpose([2, 0, 1], a)
    >>> truth = A([3, 4, 2], [0, 12, 1, 13, 2, 14, 3, 15, 4, 16, 5, 17, 6, 18, 7, 19, 8, 20, 9, 21, 10, 22, 11, 23])
    >>> match(at, truth)
    True

    Complication: repeated axes in the dyadic case:

    APL> 0 0⍉3 3⍴⍳9 ⍝ Dyadic transpose with repeated axes gets the diagonal
    ┌→────┐
    │0 4 8│
    └~────┘

    >>> a = A([3, 3], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    >>> at = transpose([0, 0], a)
    >>> match(at, V([0, 4, 8]))
    True
    """
    if omega.rank < 2: # For scalars and vectors: identity
        return deepcopy(omega)   # NOTE: need to clone here, wtf Guido

    # For the monadic transpose: transformation is the reverse axis ordering
    repeated_axes = {}
    if not alpha:
        alpha = list(range(omega.rank-1, -1, -1))
    else: # Check for repeated axes in the axis reordering spec
        repeated_axes = _repeated(alpha)

    new_shape = _reorder_axes(alpha, omega.shape)
    newdata = [0]*prod(new_shape)
    for idx, cvec in enumerate(coords(omega.shape)):
        if repeated_axes and _skip(repeated_axes, cvec):
            continue
        newdata[decode(new_shape, _reorder_axes(alpha, cvec))] = omega.data[idx]

    return A(new_shape, deepcopy(newdata))

def reduce(left: str, right: Optional[str], alpha: Optional[Array], omega: Array) -> Array:
    """
    Outward-facing '/' (trailling axis reduce)

    >>> r = reduce('+', None, None, A([2, 2], [1, 2, 3, 4]))
    >>> match(r, V([3, 7]))
    True
    """
    if right is not None:
        raise ArityError("'/' takes no right operand")

    if alpha is not None:
        raise NYIError("left argument for function derived by '/' is not implemented yet")

    return _reduce(operand=Voc.get_fn(left, Arity.DYAD), axis=omega.rank-1, omega=omega)


def reduce_first(left: str, right: Optional[str], alpha: Optional[Array], omega: Array) -> Array:
    """
    Outward-facing '⌿' (leading axis reduce)

    >>> r = reduce_first('+', None, None, A([2, 2], [1, 2, 3, 4]))
    >>> match(r, V([4, 6]))
    True
    """
    if right is not None:
        raise ArityError("'⌿' takes no right operand")

    if alpha is not None:
        raise NYIError("left argument for function derived by '⌿' is not implemented yet")

    return _reduce(operand=Voc.get_fn(left, Arity.DYAD), axis=0, omega=omega)


def _reduce(*, operand: Callable, axis: int, omega: Array) -> Array:
    """
    [private]

    Reduction along axis. 
    
    Originally by @ngn: 
       * https://chat.stackexchange.com/transcript/message/47158587#47158587
       * https://pastebin.com/mcPAxtqx

    Also used in dzaima/apl:
       * https://github.com/dzaima/APL/blob/master/src/APL/types/functions/builtins/mops/ReduceBuiltin.java#L156-L178

    >>> r = _reduce(operand=lambda x, y:x.data[0]+y.data[0], axis=0, omega=A([2, 2], [1, 2, 3, 4]))
    >>> match(r, V([4, 6]))
    True

    >>> r = _reduce(operand=lambda x, y:x.data[0]+y.data[0], axis=1, omega=A([2, 2], [1, 2, 3, 4]))
    >>> match(r, V([3, 7]))
    True
    """
    if omega.rank == 0:
        return omega
    if axis < 0:
        axis += omega.rank

    if axis >= omega.rank:
        raise RankError

    n0 = prod(omega.shape[:axis])             # prouct of dims before axis
    n1 = omega.shape[axis]                    # reduction axis size
    n2 = prod(omega.shape[axis+1:omega.rank]) # product of dims after axis

    shape = omega.shape[:]; del shape[axis]   # new shape is old shape with the reducing axis removed
    ravel = [0 for _ in range(n0*n2)]

    for i in range(n0):
        for k in range(n2):
            acc = omega.data[i*n1*n2 + (n1-1)*n2 + k]
            for j in range(n1-2, -1, -1):     # R-L
                acc = operand(omega.data[i*n1*n2 + j*n2 + k], acc)
            ravel[i*n2 + k] = acc # type: ignore

    return A(shape, ravel)

def iota(alpha: Optional[Array], omega: Array, IO: int=0) -> Array:
    if alpha is not None:
        raise NYIError('dyadic iota')

    if issimple(omega):
        if not isinstance(omega.data[0], int):
            raise DomainError('right arg must be integer')
        return V(list(range(IO, omega.data[0])))

    if omega.rank != 1:
        raise RankError('rank of right arg must be 0 or 1')

    shape = omega.to_list()
    return A(shape, list(coords(shape, IO)))

class Voc:
    """
    Voc is the global vocabulary of built-in functions and operators. This class should not
    be instantiated.
    """
    funs: dict[str, Signature] = { # Monadic, dyadic
        '⍳': (lambda y: iota(None, y), None),
        '≢': (lambda y: S(y.shape[0]), match),
        '⌊': (None,                    pervade(min)),
        '+': (None,                    pervade(operator.add)),
        # '=': (None,                    pervade(operator.eq)),  # Need to solve boolean array first
        # '>': (None,                    pervade(operator.gt)),
        # '<': (None,                    pervade(operator.lt)),
    }

    ops: dict[str, Operator] = {
        #-------------Implem--------Derived-isa--Self-isa-----L-oper-isa---R-oper-isa
        '/': Operator(reduce,       Arity.MONAD, Arity.MONAD, Arity.DYAD,  None),
        '⌿': Operator(reduce_first, Arity.MONAD, Arity.MONAD, Arity.DYAD,  None),
        '⍨': Operator(commute,      Arity.DYAD,  Arity.MONAD, Arity.DYAD,  None),
        '⍥': Operator(over,         Arity.DYAD,  Arity.DYAD,  Arity.DYAD,  Arity.MONAD)
    }

    @classmethod
    def get_fn(cls, f: Callable|str, arity: Arity) -> Callable:
        """
        Lookup a function from the global symbol table
        """
        if isinstance(f, str):
            try:
                sig = cls.funs[f]
            except KeyError:
                raise ValueError(f"Undefined function: '{f}'")
            fn = sig[arity.value]
            if fn is None:
                raise ArityError(f"function '{f}' has no {['monadic', 'dyadic'][arity.value]} form")
            return fn
        return f

    @classmethod
    def get_op(cls, n: str) -> Operator:
        """
        Lookup an operator from the global symbol table
        """
        try:
            return cls.ops[n]
        except KeyError:
            raise ValueError(f"Undefined operator: '{n}'")

def rho(alpha: Sequence[int],  omega: Array) -> Array:
    """
    Monadic: shape
    Dyadic: reshape

    Apply the shape alpha to the ravel of omega. The rank of alpha must be 0 or 1

    APL> ,5 5⍴3 3⍴3 2 0 8 6 5 4 7 1
    3 2 0 8 6 5 4 7 1 3 2 0 8 6 5 4 7 1 3 2 0 8 6 5 4

    >>> reshaped = rho([5, 5], A([3, 3], [3, 2, 0, 8, 6, 5, 4, 7, 1]))
    >>> expected = A([5, 5], [3, 2, 0, 8, 6, 5, 4, 7, 1, 3, 2, 0, 8, 6, 5, 4, 7, 1, 3, 2, 0, 8, 6, 5, 4])
    >>> match(reshaped, expected)
    True

    >>> reshaped = rho([2, 2], S(5))
    >>> expected = A([2, 2], [5, 5, 5, 5])
    >>> match(reshaped, expected)
    True

    >>> shape = rho([], A([2, 2], [1, 2, 3, 4]))
    >>> match(shape, V([2, 2]))
    True
    """
    if not alpha: # Monadic
        return V(omega.shape)

    return A(alpha, omega.data) # type: ignore

def derive(operator: Callable, left: Callable|str, right: Optional[Callable|str], arity: Arity) -> Callable:
    if arity == Arity.MONAD:
        def derived_monad(omega: Array) -> Array:
            return operator(left, right, None, omega)
        return derived_monad
    def derived_dyad(alpha: Array, omega: Array) -> Array:
        return operator(left, right, alpha, omega)
    return derived_dyad

if __name__ == "__main__":
    # To run the doctests (verbosely), do 
    #
    # python voc.py -v
    #
    # See: https://docs.python.org/3/library/doctest.html
    import doctest
    doctest.testmod()