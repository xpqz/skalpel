from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
import itertools
import cmath, math
import operator
from typing import Any, Callable, Optional, Sequence, TypeAlias

from bitarray import bitarray

import apl.arr as arr
from apl.errors import ArityError, DomainError, LengthError, NYIError, RankError
from apl.stack import Stack

Signature: TypeAlias = tuple[Optional[Callable], Optional[Callable]]

class INSTR(Enum):
    psh=0
    pop=1
    set=2
    seti=3
    get=4
    geti=5
    mon=6
    dya=7
    vec=8
    dfn=9
    fget=10

class TYPE(Enum):
    arr=0
    fun=1
    dfn=2

class Value:
    def __init__(self, payload: arr.Array|str|list[tuple], kind:TYPE) -> None:
        self.payload = payload
        self.kind = kind

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

def run(code:list[tuple], env:dict[str, Value], ip:int, stack:Stack) -> None:
    while ip < len(code):
        (instr, arg) = code[ip]
        ip += 1
        if instr == INSTR.psh:
            if isinstance(arg, str):
                stack.push([Value(arg, TYPE.fun)])
            else:
                stack.push([Value(arr.S(arg), TYPE.arr)])

        elif instr == INSTR.set:
            env[arg] = stack.pop()[0]

        elif instr == INSTR.seti:
            if arg not in env:
                raise ValueError(f'VALUE ERROR: Undefined name: "{arg}"')
            if env[arg].kind != TYPE.arr:
                raise SyntaxError("SYNTAX ERROR: Invalid modified assignment, or an attempt was made to change name class on assignment")
            (val, idx) = stack.pop(2)
            env[arg].payload.mutate(idx.payload, val.payload) # type: ignore

        elif instr == INSTR.get:
            if arg not in env:
                raise ValueError(f'VALUE ERROR: Undefined name: "{arg}"')
            stack.push([env[arg]])

        elif instr == INSTR.geti:
            if not arg: # index into literal vector
                (val, idx) = stack.pop(2)
                if val.kind != TYPE.arr:
                    raise RankError
                stack.push([Value(val.payload.at(idx.payload), TYPE.arr)]) # type: ignore
            else:
                if arg not in env:
                    raise ValueError(f'VALUE ERROR: Undefined name: "{arg}"')
                if env[arg].kind != TYPE.arr:
                    raise RankError
                idx = stack.pop()[0]
                stack.push([Value(env[arg].payload.at(idx.payload), TYPE.arr)]) # type: ignore

        elif instr == INSTR.dfn:
            if type(arg) == str: # by reference
                if arg not in env:
                    raise ValueError(f'VALUE ERROR: Undefined name: "{arg}"')
                f = env[arg]
                assert f.kind == TYPE.dfn
                stack.push([f])
            else: # direct definition
                stack.push([Value(code[ip:ip+arg], TYPE.dfn)])
                ip += arg

        elif instr == INSTR.dya:
            if arg is None: # In-line dfn
                dfn = stack.pop()[0]
                (alpha, omega) = stack.pop(2)
                assert dfn is not None and dfn.kind == TYPE.dfn
                run(dfn.payload, {'⍺': alpha, '⍵': omega}, 0, stack) # type: ignore
                continue
 
            if Voc.has_builtin(arg): # Built-in function
                fn = Voc.get_fn(arg, Arity.DYAD)
                (alpha, omega) = stack.pop(2)
                stack.push([Value(fn(alpha.payload, omega.payload), TYPE.arr)])
                continue
            
            if arg in env: # Dfn-by-name
                assert env[arg].kind == TYPE.dfn
                (alpha, omega) = stack.pop(2)
                run(env[arg].payload, {'⍺': alpha, '⍵': omega}, 0, stack) # type: ignore
                continue
            
            if Voc.has_operator(arg): # Built-in operator
                op = Voc.get_op(arg)
                omomega = None
                if op.arity == Arity.DYAD:
                    (alfalfa, omomega) = (stack.pop()[0].payload, stack.pop()[0].payload)
                else:
                    alfalfa = stack.pop()[0].payload
                (alpha, omega) = stack.pop(2)
                fn = derive(op.f, alfalfa, omomega, Arity.DYAD)
                stack.push([Value(fn(alpha.payload, omega.payload, env, stack), TYPE.arr)])
            else:
                raise ValueError(f'VALUE ERROR: unknown name {arg}')

        elif instr == INSTR.mon:
            if arg is None: # In-line dfn
                dfn = stack.pop()[0]
                assert dfn is not None and dfn.kind == TYPE.dfn
                run(dfn.payload, {'⍵': stack.pop()[0]}, 0, stack) # type: ignore
                continue
 
            if Voc.has_builtin(arg): # Built-in function
                fn = Voc.get_fn(arg, Arity.MONAD)
                stack.push([Value(fn(stack.pop()[0].payload), TYPE.arr)])
                continue
            
            if arg in env: # Dfn-by-name
                assert env[arg].kind == TYPE.dfn
                run(env[arg].payload, {'⍵': omega}, 0, stack) # type: ignore
                continue
            
            if Voc.has_operator(arg): # Built-in operator
                op = Voc.get_op(arg)
                omomega = None
                if op.arity == Arity.DYAD:
                    # same as (omomega, alfalfa) = stack.pop(2) but with the payloads
                    (alfalfa, omomega) = (stack.pop()[0].payload, stack.pop()[0].payload) # Note: order
                else:
                    alfalfa = stack.pop()[0].payload
                omega = stack.pop()[0]
                fn = derive(op.f, alfalfa, omomega, Arity.MONAD)
                stack.push([Value(fn(omega.payload, env, stack), TYPE.arr)])
            else:
                raise ValueError(f'VALUE ERROR: unknown name {arg}')

        elif instr == INSTR.vec:
            stack.push([Value(arr.V([e.payload for e in stack.pop(arg)]), TYPE.arr)])

def mpervade(f: Callable, direct: bool=False) -> Callable:
    """
    Pervade a simple function f into omega

    f must have a signature of fn(omega: int|float) -> int|float
    """
    def pervaded(omega: arr.Array, *args) -> arr.Array:
        """
        `pervaded()` takes a variable number of arguments such that the VM does 
        not have to distinguish between how it calls native monadic functions (like `-`),
        and dfns, which also need the environment and stack, e.g. (skalpel.py):

        stack.push([Value(arg(omega.payload, env, stack), TYPE.arr)])

        Case 0: +2           
        Case 1: +1 ¯2 0
        """
        if omega.array_type == arr.ArrayType.FLAT:
            if direct and omega.type == arr.DataType.UINT1: # Optimise for boolean arrays if we can
                return arr.Array(omega.shape, arr.DataType.UINT1, arr.ArrayType.FLAT, f(omega.data))
            return arr.Aflat(omega.shape, list(map(f, omega.data))) 
        return arr.A(omega.shape, [pervaded(arr.disclose(x)) for x in omega.data]) # type: ignore

    return pervaded

def pervade(f: Callable, direct: bool=False) -> Callable:
    """
    Pervade a simple function f into either arguments of equal shapes, 
    or between a scalar and an argument of any shape.

    f must have a signature of fn(alpha: int|float, omega: int|float) -> int|float
    """
    def pervaded(alpha: Any, omega: Any, *args) -> Any:
        """
        `pervaded()` takes a variable number of arguments such that the VM does 
        not have to distinguish between how it calls native dyadic functions (like `+`),
        and dfns, which also need the environment and stack, e.g. (skalpel.py):

        stack.push([Value(arg(alpha.payload, omega.payload, env, stack), TYPE.arr)])

        Case 0: 1 + 2           # both scalar
        Case 1: 1 + 1 2 3       # left is scalar
        Case 2: 1 2 3 + 1       # right is scalar
        Case 3: 1 2 3 + 4 5 6   # equal shapes
        Case 4: ....            # rank error
        """

        # flat scalars
        if not isinstance(alpha, arr.Array): 
            if not isinstance(omega, arr.Array):
                return f(alpha, omega) # both are flat
            return pervaded(arr.S(alpha), omega)
        elif not isinstance(omega, arr.Array):
            return pervaded(alpha, arr.S(omega))

        if alpha.array_type == arr.ArrayType.FLAT and omega.array_type == arr.ArrayType.FLAT:
            if alpha.shape == omega.shape:
                if direct and alpha.type == arr.DataType.UINT1 and omega.type == arr.DataType.UINT1: # be fast if we can
                    return arr.Array(alpha.shape, arr.DataType.UINT1, arr.ArrayType.FLAT, f(alpha.data, omega.data))
                return arr.Aflat(alpha.shape, [f(alpha.data[i], omega.data[i]) for i in range(alpha.bound)])
            if alpha.shape == []:
                return arr.Aflat(omega.shape, [f(alpha.data[0], omega.data[i]) for i in range(omega.bound)])
            if omega.shape == []:
                return arr.Aflat(alpha.shape, [f(alpha.data[i], omega.data[0]) for i in range(alpha.bound)])
            if alpha.rank == 1 and omega.rank == 1:       # Case 4a: unequal lengths
               raise LengthError("LENGTH ERROR: Mismatched left and right argument shapes")    
            raise RankError("RANK ERROR")                 # Case 4b: unequal shapes; rank error

        # At least one of the arrays is nested.
        if alpha.shape == omega.shape:                # Case 3: equal shapes
            data = [
                pervaded(arr.disclose(alpha.data[x]), arr.disclose(omega.data[x])) 
                for x in range(alpha.bound)
            ] # type: ignore
            return arr.A(alpha.shape, data)

        if arr.issimple(alpha):                           # Case 1: left is scalar
            return arr.A(omega.shape, [pervaded(alpha, arr.disclose(e)) for e in omega.data]) # type: ignore

        if arr.issimple(omega):                           # Case 2: right is scalar
            return arr.A(omega.shape, [pervaded(arr.disclose(e), omega) for e in alpha.data]) # type: ignore

        if alpha.rank == 1 and omega.rank == 1:       # Case 4a: unequal lengths
            raise LengthError("LENGTH ERROR: Mismatched left and right argument shapes")    
        raise RankError("RANK ERROR")                 # Case 4b: unequal shapes; rank error

    return pervaded

def _make_operand(oper: str|list[tuple], arity: Arity, env:dict[str, Value], stack: Stack) -> Callable:
    if type(oper) == str:
        if Voc.has_builtin(oper): # built-in primitive, e.g ⌊
            return Voc.get_fn(oper, arity)
        if oper not in env or env[oper].kind != TYPE.dfn:
            raise ValueError(f"VALUE ERROR: Undefined name: {oper}")
        bytecode = env[oper].payload # reference to dfn
    else:
        bytecode = oper # in-line dfn reference

    if arity == Arity.DYAD:
        def dyad(a: arr.Array, o: arr.Array) -> arr.Array:
            run(bytecode, env|{'⍺': Value(a, TYPE.arr), '⍵': Value(o, TYPE.arr)}, 0, stack) # type: ignore
            return stack.pop()[0].payload
        return dyad

    def monad(o: arr.Array) -> arr.Array:
        run(bytecode, env|{'⍵': Value(o, TYPE.arr)}, 0, stack) # type: ignore
        return stack.pop()[0].payload
    return monad
        
def commute(left: str, right: Optional[Any], alpha: arr.Array, omega: arr.Array, env:dict[str, Value], stack:Stack) -> arr.Array:
    """
    Outward-facing 'A f⍨ B' (commute) -- swap argument order
    """
    if right is not None:
        raise ArityError("ARITY ERROR: '⍨' takes no right operand")
    fun = _make_operand(left, Arity.DYAD, env, stack)
    
    return fun(omega, alpha)

def over(left: str|list[tuple], right: Optional[str|list[tuple]], alpha: arr.Array, omega: arr.Array, env:dict[str, Value], stack:Stack) -> arr.Array:
    """
    Outward-facing '⍥'
    """
    if right is None:
        raise ArityError("'⍥' takes a right operand")

    fn = _make_operand(left, Arity.DYAD, env, stack)
    pp = _make_operand(right, Arity.MONAD, env, stack)
    
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

def transpose(alpha: Sequence[int], omega: arr.Array) -> arr.Array:
    """
    Monadic and dyadic transpose. Dyadic transpose is a generalisation of
    the monadic case, which reverses the axes. In the dyadic case, axes can
    be in any order. 
    
    The general dyadic transpose: reorder axes

    APL> ,2 0 1⍉2 3 4⍴⍳×/2 3 4
    ┌→────────────────────────────────────────────────────────────┐
    │0 12 1 13 2 14 3 15 4 16 5 17 6 18 7 19 8 20 9 21 10 22 11 23│
    └~────────────────────────────────────────────────────────────┘

    >>> a = arr.Aflat([2, 3, 4], list(range(2*3*4)))
    >>> transpose([2, 0, 1], a)
    A([3, 4, 2], UINT8, FLAT, [0, 12, 1, 13, 2, 14, 3, 15, 4, 16, 5, 17, 6, 18, 7, 19, 8, 20, 9, 21, 10, 22, 11, 23])

    Complication: repeated axes in the dyadic case:

    APL> 0 0⍉3 3⍴⍳9 ⍝ Dyadic transpose with repeated axes gets the diagonal
    ┌→────┐
    │0 4 8│
    └~────┘

    >>> a = arr.Aflat([3, 3], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    >>> transpose([0, 0], a)
    V(UINT8, FLAT, [0, 4, 8])
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
    newdata = [0]*math.prod(new_shape)
    for idx, cvec in enumerate(arr.coords(omega.shape)):
        if repeated_axes and _skip(repeated_axes, cvec):
            continue
        newdata[arr.decode(new_shape, _reorder_axes(alpha, cvec))] = omega.data[idx]

    # Check if we can make this flat.
    for e in newdata:
        if isinstance(e, arr.Array):
            return arr.A(new_shape, deepcopy(newdata))
    
    return arr.Aflat(new_shape, deepcopy(newdata)) # Possibly won't need deepcopy() here.

def each(left: str|list[tuple], right: Optional[str], alpha: Optional[Any],  omega: arr.Array, env:dict[str, Value], stack:Stack) -> arr.Array:
    """
    Each ¨ - monadic operator deriving monad

    TODO: squeeze the result array
    """
    if right is not None:
        raise ArityError("'¨' takes no right operand")

    if alpha is not None:
        raise ArityError("function derived by '¨' takes no left argument")

    fun = _make_operand(left, Arity.MONAD, env, stack)

    return arr.A(omega.shape, [fun(arr.disclose(o)) for o in omega.data])

def reduce(left: str|list[tuple], right: Optional[Any], alpha: Optional[arr.Array], omega: arr.Array, env:dict[str, Value], stack:Stack) -> arr.Array:
    """
    Outward-facing '/' (trailling axis reduce)

    >>> reduce('+', None, None, arr.Aflat([2, 2], [1, 2, 3, 4]))
    V(UINT8, FLAT, [3, 7])
    """
    if right is not None:
        raise ArityError("'/' takes no right operand")

    if alpha is not None:
        raise NYIError("left argument for function derived by '/' is not implemented yet")

    # Local optimisations go here.
    if omega.array_type == arr.ArrayType.FLAT:
        # Dirty hacks for flat vectors
        if omega.rank == 1:
            if left == '+':
                if omega.type == arr.DataType.UINT1:
                    return arr.S(omega.data.count()) # type: ignore
                return arr.S(sum(omega.data))
            if left == '×':
                return arr.S(math.prod(omega.data))

        # Dirty hacks for flat matrices
        if omega.rank == 2:
            if left == '+':
                if omega.type == arr.DataType.UINT1:
                    return arr.V([row.count() for row in arr.rows_flat_matrix(omega)])
                return arr.V([sum(row) for row in arr.rows_flat_matrix(omega)])
            if left == '×':
                return arr.V([math.prod(row) for row in arr.rows_flat_matrix(omega)])

    fun = _make_operand(left, Arity.DYAD, env, stack)

    return _reduce(operand=fun, axis=omega.rank-1, omega=omega)

def reduce_first(left: str|list[tuple], right: Optional[Any], alpha: Optional[arr.Array], omega: arr.Array, env:dict[str, Value], stack:Stack) -> arr.Array:
    """
    Outward-facing '⌿' (leading axis reduce)

    >>> reduce_first('+', None, None, arr.Aflat([2, 2], [1, 2, 3, 4]))
    V(UINT8, FLAT, [4, 6])
    """
    if right is not None:
        raise ArityError("'/' takes no right operand")

    if alpha is not None:
        raise NYIError("left argument for function derived by '/' is not implemented yet")

    # Local optimisations go here.
    if omega.array_type == arr.ArrayType.FLAT:
        if omega.rank == 1:
            if left == '+':
                if omega.type == arr.DataType.UINT1:
                    return arr.S(omega.data.count()) # type: ignore
                return arr.S(sum(omega.data))
            if left == '×':
                return arr.S(math.prod(omega.data))

    fun = _make_operand(left, Arity.DYAD, env, stack)

    return _reduce(operand=fun, axis=0, omega=omega)

def _reduce(*, operand: Callable, axis: int, omega: arr.Array) -> arr.Array:
    """
    [private]

    Reduction along axis. 
    
    Originally by @ngn: 
       * https://chat.stackexchange.com/transcript/message/47158587#47158587
       * https://pastebin.com/mcPAxtqx

    Also used in dzaima/apl:
       * https://github.com/dzaima/APL/blob/master/src/APL/types/functions/builtins/mops/ReduceBuiltin.java#L156-L178

    >>> _reduce(operand=lambda x, y:x+y, axis=0, omega=arr.Aflat([2, 2], [1, 2, 3, 4]))
    V(UINT8, FLAT, [4, 6])

    >>> _reduce(operand=lambda x, y:x+y, axis=1, omega=arr.Aflat([2, 2], [1, 2, 3, 4]))
    V(UINT8, FLAT, [3, 7])
    """
    if omega.rank == 0:
        return omega
    if axis < 0:
        axis += omega.rank

    if axis >= omega.rank:
        raise RankError

    n0 = math.prod(omega.shape[:axis])             # prouct of dims before axis
    n1 = omega.shape[axis]                    # reduction axis size
    n2 = math.prod(omega.shape[axis+1:omega.rank]) # math.product of dims after axis

    shape = omega.shape[:]; del shape[axis]   # new shape is old shape with the reducing axis removed
    ravel = [0 for _ in range(n0*n2)]

    for i in range(n0):
        for k in range(n2):
            acc = omega.data[i*n1*n2 + (n1-1)*n2 + k]
            for j in range(n1-2, -1, -1):     # R-L
                acc = operand(omega.data[i*n1*n2 + j*n2 + k], acc)
            ravel[i*n2 + k] = acc # type: ignore

   # Check if we can make this flat. TODO: this is too generic
    for e in ravel:
        if isinstance(e, arr.Array):
            return arr.A(shape, ravel)
    
    return arr.Aflat(shape, ravel)

def index_gen(omega: arr.Array, IO: int=0) -> arr.Array:
    """
    ⍳ - index generator (monadic)

    >>> index_gen(arr.S(5))      # Scalar omega
    V(UINT8, FLAT, [0, 1, 2, 3, 4])

    >>> index_gen(arr.V([2, 2])) # Vector omega
    A([2, 2], MIXED, NESTED, [<V(UINT1, FLAT, [0, 0])>, <V(UINT1, FLAT, [0, 1])>, <V(UINT1, FLAT, [1, 0])>, <V(UINT1, FLAT, [1, 1])>])
     """
    if arr.issimple(omega):
        if not isinstance(omega.data[0], int):
            raise DomainError('DOMAIN ERROR: right arg must be integer')
        return arr.Aflat([omega.data[0]], range(IO, omega.data[0]))

    if omega.rank > 1:
        raise RankError('RANK ERROR: rank of right arg must be 0 or 1')

    shape = omega.to_list()
    return arr.A(shape, [arr.Aflat([len(c)], c) for c in arr.coords(shape, IO)])

def where(omega: arr.Array) -> arr.Array:
    if omega.type != arr.DataType.UINT1: # for now
        raise DomainError('DOMAIN ERROR: expected Boolean array')

    if omega.rank == 1:
        return arr.Aflat([omega.data.count()], omega.data.search(bitarray([True]))) # type: ignore

    return arr.V([
        arr.Array([omega.rank], arr.DataType.INT, arr.ArrayType.FLAT, arr.encode(omega.shape, idx)) # type: ignore
        for idx in omega.data.search(bitarray([True])) # type: ignore
    ])

def rho(alpha: Optional[arr.Array], omega: arr.Array) -> arr.Array:
    """
    Monadic: shape
    Dyadic: reshape

    Apply the shape alpha to the ravel of omega. The rank of alpha must be 0 or 1

    APL> ,5 5⍴3 3⍴3 2 0 8 6 5 4 7 1
    3 2 0 8 6 5 4 7 1 3 2 0 8 6 5 4 7 1 3 2 0 8 6 5 4

    >>> rho(arr.V([5, 5]), arr.Aflat([3, 3], [3, 2, 0, 8, 6, 5, 4, 7, 1]))
    A([5, 5], UINT8, FLAT, [3, 2, 0, 8, 6, 5, 4, 7, 1, 3, 2, 0, 8, 6, 5, 4, 7, 1, 3, 2, 0, 8, 6, 5, 4])

    >>> rho(arr.V([2, 2]), arr.S(5))
    A([2, 2], UINT8, FLAT, [5, 5, 5, 5])

    >>> rho(None, arr.Aflat([2, 2], [1, 2, 3, 4]))
    V(UINT8, FLAT, [2, 2])
    """
    if alpha is None: # Monadic
        return arr.Aflat([len(omega.shape)], omega.shape)

    if not arr.isnested(omega):
        return arr.Aflat(alpha.to_list(), omega.data) # type: ignore

    return arr.A(alpha.to_list(), omega.data) # type: ignore

def enlist(omega: arr.Array) -> arr.Array:
    """
    Monadic ∊ - create a vector of all simple scalars contained in omega, recursively
    drilling into any nesting and shapes.

    >>> enlist(arr.V([arr.S(9), arr.V([1, 2])]))
    V(UINT8, FLAT, [9, 1, 2])
    """
    data = []
    def inner(o: arr.Array) -> None:
        for e in o.data:
            elem = arr.disclose(e)
            if arr.issimple(elem): # type: ignore
                data.append(elem.data[0]) # type: ignore
            else:
                inner(elem) # type: ignore
    inner(omega)
    return arr.Aflat([len(data)], data) 

def member(alpha: arr.Array, omega: arr.Array) -> arr.Array:
    """
    Dyadic ∊ - which cells in alpha are found as a cell in omega?
    
    >>> a = arr.A([3, 2],[arr.V([1, 2]), arr.V([1, 3]), arr.S(4), arr.V([1, 2, 3, 4])])
    >>> b = arr.V([arr.V([1, 2]), arr.V([1, 2, 3, 4]), arr.V([1, 4])])
    >>> member(b, a)
    V(UINT1, FLAT, [1, 1, 0])
    """
    result = []
    for ac in arr.coords(alpha.shape):
        cell_a = arr.disclose(alpha.get(ac))
        found = 0
        for oc in arr.coords(omega.shape):
            cell_o = arr.disclose(omega.get(oc))
            if arr.match(cell_a, cell_o):
                found = 1
                break
        result.append(found)

    return arr.Array(alpha.shape, arr.DataType.UINT1, arr.ArrayType.FLAT, bitarray(result))
                
def tally(omega: arr.Array) -> arr.Array:
    return arr.S(len(omega.data))

def enclose(omega: arr.Array) -> arr.Array:
    if arr.issimple(omega):
        return omega
    return arr.S(omega)

def or_gcd(alpha: arr.Array, omega: arr.Array) -> arr.Array:
    if alpha.type == omega.type == arr.DataType.UINT1:
        f = pervade(lambda x, y:x|y, direct=True)
    else:
        f = pervade(math.gcd)
    return f(alpha, omega)

def and_lcm(alpha: arr.Array, omega: arr.Array) -> arr.Array:
    if alpha.type == omega.type == arr.DataType.UINT1:
        f = pervade(lambda x, y:x&y, direct=True)
    else:
        f = pervade(math.lcm)
    return f(alpha, omega)

def bool_not(omega: arr.Array) -> arr.Array:
    if omega.type != arr.DataType.UINT1:
        raise DomainError("DOMAIN ERROR: expected boolean array")
    f = mpervade(lambda y:~y)
    return f(omega)

def replicate(alpha: arr.Array, omega: arr.Array) -> arr.Array:
    if alpha.rank != 1 or alpha.array_type != arr.ArrayType.FLAT:
        raise DomainError('DOMAIN ERROR')
    if alpha.type == arr.DataType.UINT1: # compress
        return arr.V([arr.disclose(omega.data[c]) for c in range(omega.bound) if alpha.data[c] == 1])

    # Replicate. NOTE: may need to disclose ⍺, too
    return arr.V(list(itertools.chain.from_iterable(itertools.repeat(arr.disclose(omega.data[i]), alpha.data[i]) for i in range(alpha.bound))))
    
def without(alpha: arr.Array, omega: arr.Array) -> arr.Array:
    if alpha.rank > 1 or omega.rank > 1:
        raise RankError('RANK ERROR: dyadic ~ requires argument arrays to be max rank 1')

    if alpha.array_type == omega.array_type == arr.ArrayType.FLAT:
        return arr.V(list(set(alpha.data)-set(omega.data))) # recent pythons retain set ordering

    # For nested vectors, we need to be slow, sadly -- Arrays aren't hashable
    mask = member(alpha, omega)
    mask.data = ~mask.data # type: ignore
    return replicate(mask, alpha)

def circle(alpha: int, omega: int|float|complex) -> float|complex:
    """
     ⍺   ⍺ ○ ⍵         ⍺   ⍺ ○ ⍵     
                       0   (1-⍵*2)*0.5 
    ¯1   Arcsin ⍵      1   Sine ⍵ 
    ¯2   Arccos ⍵      2   Cosine ⍵ 
    ¯3   Arctan ⍵      3   Tangent ⍵ 
    ¯4   (¯1+⍵*2)*0.5  4   (1+⍵*2)*0.5 
    ¯5   Arcsinh ⍵     5   Sinh ⍵ 
    ¯6   Arccosh ⍵     6   Cosh ⍵ 
    ¯7   Arctanh ⍵     7   Tanh ⍵ 
    ¯8   -8○⍵          8   (-1+⍵*2)*0.5 
    ¯9   ⍵             9   real part of ⍵ 
    ¯10  +⍵           10   |⍵ 
    ¯11  ⍵×0J1        11   imaginary part of ⍵ 
    ¯12  *⍵×0J1       12   phase of ⍵ 
    """

    # My least favourite part of APL :/
    if alpha == 1:
        if isinstance(omega, complex):
            return cmath.sin(omega)
        return math.sin(omega)
    if alpha == -1:
        if isinstance(omega, complex):
            return cmath.asin(omega)
        return math.asin(omega)
    if alpha == 2:
        if isinstance(omega, complex):
            return cmath.cos(omega)
        return math.cos(omega)
    if alpha == -2:
        if isinstance(omega, complex):
            return cmath.acos(omega)
        return math.acos(omega)
    if alpha == 3:
        if isinstance(omega, complex):
            return cmath.tan(omega)
        return math.tan(omega)
    if alpha == -3:
        if isinstance(omega, complex):
            return cmath.atan(omega)
        return math.atan(omega)
    if alpha == 4:
        return (1+omega**2)**0.5
    if alpha == -4:
        return (-1+omega**2)**0.5
    if alpha == 5:
        if isinstance(omega, complex):
            return cmath.sinh(omega)
        return math.sinh(omega)
    if alpha == -5:
        if isinstance(omega, complex):
            return cmath.asinh(omega)
        return math.asinh(omega)
    if alpha == 6:
        if isinstance(omega, complex):
            return cmath.cosh(omega)
        return math.cosh(omega)
    if alpha == -6:
        if isinstance(omega, complex):
            return cmath.acosh(omega)
        return math.acosh(omega)
    if alpha == 7:
        if isinstance(omega, complex):
            return cmath.tanh(omega)
        return math.tanh(omega)
    if alpha == -7:
        if isinstance(omega, complex):
            return cmath.atanh(omega)
        return math.atanh(omega)
    if alpha == 8:
        return (-1+omega**2)**0.5
    if alpha == -8:
        return -(-1+omega**2)**0.5
    if alpha == 9:
        return omega.real
    if alpha == -9:
        return omega
    if alpha == 10:
        return abs(omega)
    if alpha == -10:
        if isinstance(omega, complex):
            return omega.conjugate()
        return omega
    if alpha == 11:
        if isinstance(omega, complex):
            return omega.imag
        return 0.0
    if alpha == -11:
        return omega*complex(0, 1)
    if alpha == 12:
        if isinstance(omega, complex):
            return cmath.phase(omega)
        return 0.0
    if alpha == -12:
        a = omega*complex(0, 1)
        return math.hypot(a.real, a.imag)

    raise DomainError(f"DOMAIN ERROR: unknown magic number for ○: {alpha}")

class Voc:
    """
    Voc is the global vocabulary of built-in functions and operators. This class should not
    be instantiated.
    """
    funs: dict[str, Signature] = { 
        #--- Monadic-----------------------Dyadic----------------
        '~': (bool_not,                     without),
        '∨': (None,                         or_gcd),
        '∧': (None,                         and_lcm),
        '∊': (enlist,                       member),
        '⊂': (enclose,                      None), 
        '⍉': (lambda y: transpose([], y),   lambda x, y: transpose(x.to_list(), y)),
        '⍴': (lambda y: rho(None, y),       rho),
        '⍳': (index_gen,                    None),
        '⍸': (where,                        None),
        '≢': (tally,                        lambda x, y: arr.S(int(not arr.match(x, y)))),
        '≡': (None,                         lambda x, y: arr.S(int(arr.match(x, y)))),
        '⌈': (None,                         pervade(max)),
        '⌊': (None,                         pervade(min)),
        '!': (mpervade(math.factorial),     None),
        '○': (mpervade(lambda o:o*math.pi), pervade(circle)),
        '+': (None,                         pervade(operator.add)),
        '-': (mpervade(operator.neg),       pervade(operator.sub)),
        '×': (mpervade(lambda y:y/abs(y)),  pervade(operator.mul)),
        '|': (mpervade(lambda y:abs(y)),    pervade(lambda x,y:y%x)),       # DYADIC NOTE ARG ORDER
        '=': (None,                         pervade(lambda x,y:int(x==y))),
        '>': (None,                         pervade(lambda x,y:int(x>y))),
        '<': (None,                         pervade(lambda x,y:int(x<y))),
        '≠': (None,                         pervade(lambda x,y:int(x!=y))),
        '≥': (None,                         pervade(lambda x,y:int(x>=y))),
        '≤': (None,                         pervade(lambda x,y:int(x<=y))),
    }

    ops: dict[str, Operator] = {
        #-------------Implem--------Derived-isa--Self-isa-----L-oper-isa---R-oper-isa
        '/': Operator(reduce,       Arity.MONAD, Arity.MONAD, Arity.DYAD,  None),
        '⌿': Operator(reduce_first, Arity.MONAD, Arity.MONAD, Arity.DYAD,  None),
        '¨': Operator(each,         Arity.MONAD, Arity.MONAD, Arity.MONAD, None),
        '⍨': Operator(commute,      Arity.DYAD,  Arity.MONAD, Arity.DYAD,  None),
        '⍥': Operator(over,         Arity.DYAD,  Arity.DYAD,  Arity.DYAD,  Arity.MONAD)
    }

    @classmethod
    def has_builtin(cls, f: str) -> bool:
        return f in cls.funs

    @classmethod
    def has_operator(cls, f: str) -> bool:
        return f in cls.ops

    @classmethod
    def get_fn(cls, f: Callable|str, arity: Arity) -> Callable:
        """
        Lookup a function from the global symbol table
        """
        if isinstance(f, str):
            try:
                sig = cls.funs[f]
            except KeyError:
                raise ValueError(f"VALUE ERROR: Undefined function: '{f}'")
            fn = sig[arity.value]
            if fn is None:
                raise ArityError(f"ARITY ERROR: function '{f}' has no {['monadic', 'dyadic'][arity.value]} form")
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
            raise ValueError(f"VALUE ERROR: Undefined operator: '{n}'")

def derive(operator: Callable, left: str, right: Optional[str], arity: Arity) -> Callable:
    """
    Only built-in operators for now
    """
    if arity == Arity.MONAD:
        def derived_monad(omega: arr.Array, env:dict[str, Value], stack:Stack) -> arr.Array:
            return operator(left, right, None, omega, env, stack)
        return derived_monad

    def derived_dyad(alpha: arr.Array, omega: arr.Array, env:dict[str, Value], stack:Stack) -> arr.Array:
        return operator(left, right, alpha, omega, env, stack)
    return derived_dyad

if __name__ == "__main__":
    # To run the doctests (verbosely), do 
    #
    # python skalpel.py -v
    #
    # See: https://docs.python.org/3/library/doctest.html
    import doctest
    doctest.testmod()    

