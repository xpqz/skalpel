import cmath
from copy import deepcopy
from dataclasses import dataclass
from enum import auto, Enum, Flag
import itertools
import math
import operator
import random
import secrets
from string import ascii_letters
from typing import Any, Callable, Optional, TypeAlias

from bitarray import bitarray

import apl.arr as arr
from apl.errors import (ArityError, DomainError, LengthError, NYIError,
                        RankError)
from apl.stack import Stack

Signature: TypeAlias = tuple[Optional[Callable], Optional[Callable]]

class INSTR(Enum):
    push=0     # push scalar
    fun=1      # push function name
    pop=2      # pop stack
    set=3      # create variable
    seti=4     # modify variable by index
    get=5      # retrieve value by name
    geti=6     # retrieve value @ index
    mon=7      # monadic call
    dya=8      # dyadic call
    vec=9      # vector
    cvec=10    # character vector (needed to ensure that empty '' gets the right type)
    dfn=11     # define dfn

class TYPE(Enum):
    arr=0
    fun=1
    dfn=2

class Value:
    def __init__(self, payload: arr.Array|str|list[tuple]|int|float|complex, kind:TYPE) -> None:
        self.payload = payload
        self.kind = kind

class Arity(Enum):
    MONAD=0
    DYAD=1
    AMBIV=2

class OperandType(Flag):
    ARRAY = auto()
    MONAD = auto()
    DYAD = auto()

@dataclass(frozen=True)
class Operator:
    f: Callable
    derives: Arity                        # Arity of derived function
    arity: Arity                          # Am I a monadic or a dyadic operator?
    left: OperandType                     # Spec of left operand
    right: Optional[OperandType] = None   # Spec of right operand, if I am dyadic

def run(code:list[tuple], env:dict[str, Value], ip:int, stack:Stack) -> None:
    while ip < len(code):
        (instr, arg) = code[ip]
        ip += 1
        if instr == INSTR.push:
            stack.push([Value(arr.S(arg), TYPE.arr)])

        elif instr == INSTR.fun:
            stack.push([Value(arg, TYPE.fun)])

        elif instr == INSTR.set:
            val = stack.pop()[0]
            env[arg] = val
            if val.kind == TYPE.arr:
                stack.push([env[arg]])    # array assignments evaluate to their value, e.g 1+a←2

        elif instr == INSTR.seti:
            if arg not in env:
                raise ValueError(f'VALUE ERROR: Undefined name: "{arg}"')
            if env[arg].kind != TYPE.arr:
                raise SyntaxError("SYNTAX ERROR: Invalid modified assignment, or an attempt was made to change name class on assignment")
            (val, idx) = stack.pop(2)
            env[arg].payload.mutate(idx.payload, val.payload) # type: ignore

        elif instr == INSTR.get:
            if arg.upper() in Voc.arrs:
                stack.push([Value(Voc.arrs[arg.upper()], TYPE.arr)])
            else:
                if arg not in env:
                    raise ValueError(f'VALUE ERROR: Undefined name: "{arg}"')
                stack.push([env[arg]])

        elif instr == INSTR.geti:
            if not arg: # index into literal vector
                (idx, val) = stack.pop(2)
                if val.kind != TYPE.arr:
                    raise RankError
                stack.push([Value(val.payload.at(idx.payload), TYPE.arr)]) # type: ignore
            else:
                idx = stack.pop()[0]
                if arg.upper() in Voc.arrs: # System array constant
                    sysarr = Voc.arrs[arg.upper()]
                    stack.push([Value(sysarr.at(idx.payload), TYPE.arr)])
                    continue

                if arg not in env:
                    raise ValueError(f'VALUE ERROR: Undefined name: "{arg}"')
                if env[arg].kind != TYPE.arr:
                    raise RankError
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
                (omega, alpha) = stack.pop(2) # note: no .payload!
                assert dfn is not None and dfn.kind == TYPE.dfn
                run(dfn.payload, {'⍺': alpha, '⍵': omega}, 0, stack) # type: ignore
                continue
 
            if Voc.has_builtin(arg): # Built-in function
                fn = Voc.get_fn(arg, Arity.DYAD)
                (omega, alpha) = stack.pop(2)
                stack.push([Value(fn(alpha.payload, omega.payload), TYPE.arr)])
                continue
            
            if arg in env: # Dfn-by-name
                assert env[arg].kind == TYPE.dfn
                (omega, alpha) = stack.pop(2) # note: no .payload!
                run(env[arg].payload, {'⍺': alpha, '⍵': omega}, 0, stack) # type: ignore
                continue
            
            if Voc.has_operator(arg): # Built-in operator
                op = Voc.get_op(arg)
                omomega = None
                if op.arity == Arity.DYAD:
                    (alfalfa, omomega) = (stack.pop()[0].payload, stack.pop()[0].payload) # FIXME: order. Fix emit order in node.py, too
                else:
                    alfalfa = stack.pop()[0].payload
                (omega, alpha) = stack.pop(2)
                fn = derive(op.f, alfalfa, omomega, Arity.DYAD)
                stack.push([Value(fn(alpha.payload, omega.payload, env, stack), TYPE.arr)])
            else:
                raise ValueError(f'VALUE ERROR: unknown name {arg}')

        elif instr == INSTR.mon:
            if arg is None: # In-line dfn
                dfn = stack.pop()[0]
                assert dfn is not None and dfn.kind == TYPE.dfn
                omega = stack.pop()[0] # note: no .payload!
                run(dfn.payload, {'⍵': omega}, 0, stack) # type: ignore
                continue
 
            if Voc.has_builtin(arg): # Built-in function
                fn = Voc.get_fn(arg, Arity.MONAD)
                omega = stack.pop()[0].payload
                stack.push([Value(fn(omega), TYPE.arr)])
                continue
            
            if arg in env: # Dfn-by-name
                assert env[arg].kind == TYPE.dfn
                omega = stack.pop()[0] # note: no .payload!
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
                omega = stack.pop()[0].payload
                fn = derive(op.f, alfalfa, omomega, Arity.MONAD)
                stack.push([Value(fn(omega, env, stack), TYPE.arr)])
            else:
                raise ValueError(f'VALUE ERROR: unknown name {arg}')

        elif instr == INSTR.vec:
            stack.push([Value(arr.V([e.payload for e in stack.pop(arg)]), TYPE.arr)])

        elif instr == INSTR.cvec:
            data = [e.payload for e in stack.pop(arg)]
            if data:
                stack.push([Value(arr.V(data), TYPE.arr)])
            else:
                stack.push([Value(arr.Array([0], [' ']), TYPE.arr)])

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
        if not omega.nested:
            return arr.Array(omega.shape, list(map(f, omega.data))) 
        return arr.Array(omega.shape, [pervaded(arr.enclose_if_simple(x)) for x in omega.data])

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

        # True scalars
        if not isinstance(alpha, arr.Array): 
            if not isinstance(omega, arr.Array):
                return f(alpha, omega) # both are simple (python non-arrays)
            return pervaded(arr.S(alpha), omega)
        elif not isinstance(omega, arr.Array):
            return pervaded(alpha, arr.S(omega))

       # Singleton extension: https://aplwiki.com/wiki/Scalar_extension#Singleton_extension
        if alpha.issingleton() and omega.issingleton():  # If both are singletons, extend lower rank.
            if alpha.rank < omega.rank:
                alpha = arr.S(alpha.data[0])
            elif omega.rank < alpha.rank:
                omega = arr.S(omega.data[0])
        elif alpha.issingleton():
            alpha = arr.S(alpha.data[0])
        elif omega.issingleton():
            omega = arr.S(omega.data[0])

        # Non-simple scalar extension, e.g. 1 2 3+⊂100 200

        if not alpha.nested and not omega.nested:
            if alpha.shape == omega.shape:
                return arr.Array(alpha.shape, [f(alpha.data[i], omega.data[i]) for i in range(alpha.bound)])
            if not alpha.shape:
                return arr.Array(omega.shape, [f(alpha.data[0], omega.data[i]) for i in range(omega.bound)])
            if omega.shape == []:
                return arr.Array(alpha.shape, [f(alpha.data[i], omega.data[0]) for i in range(alpha.bound)])
            if alpha.rank == omega.rank == 1:   # Case 4a: unequal lengths
               raise LengthError("LENGTH ERROR: Mismatched left and right argument shapes")    
            raise RankError("RANK ERROR")                       # Case 4b: unequal shapes; rank error

        # At least one of the arrays is nested.
        if alpha.shape == omega.shape:                          # Case 3: equal shapes
            data = [
                pervaded(alpha.data[x], omega.data[x])
                for x in range(alpha.bound)
            ]
            return arr.Array(alpha.shape, data)

        if alpha.isscalar():                                    # Case 1: left is scalar
            return arr.Array(omega.shape, [pervaded(alpha, e).unbox() for e in omega.data])

        if omega.isscalar():                                    # Case 2: right is scalar
            return arr.Array(alpha.shape, [pervaded(e, omega).unbox() for e in alpha.data])

        if alpha.rank == omega.rank == 1:                       # Case 4a: unequal lengths
            raise LengthError("LENGTH ERROR: Mismatched left and right argument shapes")    
        raise RankError("RANK ERROR")                           # Case 4b: unequal shapes; rank error

    return pervaded

def _make_operand(oper: str|list[tuple]|Callable, arity: Arity, env:dict[str, Value], stack: Stack) -> Callable:
    if callable(oper):
        return oper
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

def each(left: str|list[tuple], right: Optional[str], alpha: Optional[Any],  omega: arr.Array, env:dict[str, Value], stack:Stack) -> arr.Array:
    """
    Each ¨ - monadic operator deriving monad
    """
    if right is not None:
        raise ArityError("operator '¨' takes no right operand")

    if left is None:
        raise ArityError("operator '¨' needs a left operand")

    if alpha is not None:
        raise ArityError("function derived by '¨' takes no left argument")

    assert isinstance(omega, arr.Array)

    fun = _make_operand(left, Arity.MONAD, env, stack)
    d = [fun(arr.enclose_if_simple(o)) for o in omega.data] # apply left operand to each element
    return arr.Array(omega.shape, deepcopy(d))

def reduce(left: str|list[tuple], right: Optional[Any], alpha: Optional[arr.Array], omega: arr.Array, env:dict[str, Value], stack:Stack) -> arr.Array:
    """
    Outward-facing '/' (trailling axis reduce)
    """
    if right is not None:
        raise ArityError("'/' takes no right operand")

    if alpha is not None:
        raise NYIError("left argument for function derived by '/' is not implemented yet")

    fun = _make_operand(left, Arity.DYAD, env, stack)

    return omega.foldr(operand=fun, axis=omega.rank-1)

def reduce_first(left: str|list[tuple], right: Optional[Any], alpha: Optional[arr.Array], omega: arr.Array, env:dict[str, Value], stack:Stack) -> arr.Array:
    """
    Outward-facing '⌿' (leading axis reduce)
    """
    if right is not None:
        raise ArityError("'/' takes no right operand")

    if alpha is not None:
        raise NYIError("left argument for function derived by '/' is not implemented yet")

    fun = _make_operand(left, Arity.DYAD, env, stack)

    return omega.foldr(operand=fun, axis=0)

def amend(left: str|list[tuple]|arr.Array|Callable, right: str|list[tuple]|arr.Array|Callable, alpha: Optional[arr.Array], omega: arr.Array, env:dict[str, Value], stack:Stack) -> arr.Array:
    """
    Amend - dyadic operator @ deriving ambivalent, returning a copy, whilst modifying elements at 
    specific positions in an array. 
    
    If an alpha is given, it's implicitly bound to the left operand function, so that

        ⍺ (L@R) ⍵
        
    is equivalent to 
    
        (⍺∘L@R) ⍵

    Both operands can be either functions or arrays. 
    
    For the right operand, if it's a function, it needs to return a Boolean array of the same 
    shape as ⍵, with ones in the positions to be modified. If it's an array, its major cells 
    represent the coordinates of ⍵ to modify.

    For the left operand, if it's a function, it is applied monadically to a vector of values to
    modify, and thus should return an array of the same shape as its right argument. If it's an
    array, it must either have same shape as the array whose major cells are the selected cells 
    from ⍵, or a single scalar which will then be used as the value for all selected cells.

    https://aplwiki.com/wiki/At
    https://help.dyalog.com/latest/index.htm#Language/Primitive%20Operators/At.htm
    https://xpqz.github.io/learnapl/at.html
    https://xpqz.github.io/cultivations/Operators.html#at

    (1 2 3@4 5 6)7 5 4 10 3 6 9 2 1 8 ⍝ Straight replacement
    ┌→───────────────────┐
    │7 5 4 10 1 2 3 2 1 8│
    └~───────────────────┘
    
    (0@2 4) 1 2 3 4 5                 ⍝ Scalar extension
    ┌→────────┐
    │1 2 0 4 0│
    └~────────┘

    10 (×@2 4) 1 2 3 4 5              ⍝ Left is function
    ┌→──────────┐
    │1 2 30 4 50│
    └~──────────┘

    '*'@(2∘|) 1 2 3 4 5               ⍝ Right is function (returning Bool array)
    ┌→────────┐
    │* 2 * 4 *│
    └+────────┘

    10 (×@(≤∘3)) 3 1 4 1 5            ⍝ Left and right functions
    ┌→───────────┐
    │30 10 4 10 5│
    └~───────────┘
    
    ⌽@(2∘|) 1 2 3 4 5                 ⍝ Rotate sub-array
    ┌→────────┐
    │5 2 3 4 1│
    └~────────┘

    """
    if not isinstance(right, arr.Array):
        selector = _make_operand(right, Arity.MONAD, env, stack)
        indices = selector(omega).where()
    else: 
        indices = right

    selected = omega.at(indices)

    if not isinstance(left, arr.Array):
        val = _make_operand(left, Arity.DYAD, env, stack) if alpha else _make_operand(left, Arity.MONAD, env, stack)
        values = val(alpha, selected) if alpha else val(selected)
    else:
        if not left.shape: # Extend scalar
            values = arr.V([deepcopy(left.data[0]) for _ in range(selected.bound)])
        else:
            values = left

    if selected.shape != values.shape:
        raise LengthError('LENGTH ERROR')

    result = deepcopy(omega)
    result.mutate(indices, values)

    return result

def power(left: str|list[tuple], right: str|list[tuple]|arr.Array, alpha: Optional[arr.Array], omega: arr.Array, env:dict[str, Value], stack:Stack) -> arr.Array:
    """
    Power - dyadic operator ⍣ deriving monad. If an alpha is given, it's implicitly bound to 
    the left operand function, so that

        ⍺ L⍣R ⍵
        
    is equivalent to 
    
        ⍺∘L⍣R ⍵

    If R is an integer scalar, it's an iteration count.

    APL Wiki: https://aplwiki.com/wiki/Power_(operator)

        Power X(f⍣g)Y repeatedly applies f to Y based on the type of operand g:

        * Function: Must be dyadic and must return a boolean singleton. The 
          previous iteration value is provided as the right argument to f, 
          and the current iteration value is given as the left argument. f 
          is repeatedly applied until this function returns 1.

        * Integer: Applies f g times to Y. If g is negative, then the inverse 
          of f (if available) is applied.

        
    https://help.dyalog.com/latest/index.htm#Language/Primitive%20Operators/Power%20Operator.htm
    https://xpqz.github.io/learnapl/iteration.html#power
    https://xpqz.github.io/cultivations/Power1.html
    https://xpqz.github.io/cultivations/Power2.html
    https://wiki.nars2000.org/index.php?title=Power
    """

    left_fun = _make_operand(left, Arity.DYAD, env, stack) if alpha else _make_operand(left, Arity.MONAD, env, stack)

    if isinstance(right, arr.Array):
        if not right.issimple():
            raise RankError('RANK ERROR')

        count = right.data[0]

        if type(count) != int:
            raise DomainError('DOMAIN ERROR')
        if count == -1:
            raise NYIError('NYI ERROR: inverses not implemented')
        if count < 0:
            raise DomainError('DOMAIN ERROR')

        current = omega
        while count:
            current = left_fun(alpha, current) if alpha else left_fun(current)
            count -= 1

        return current

    # Right operand is a function
    right_fun = _make_operand(right, Arity.DYAD, env, stack)
    pre = omega
    post = left_fun(alpha, pre) if alpha else left_fun(pre)
    while True:
        done = right_fun(post, pre)
        if not done.issimple() or done.issimple() and done.data[0] not in {0, 1}:
            raise DomainError('DOMAIN ERROR: right operand must return Boolean')
        if done.data[0] == 1:
            break
        pre = post
        post = left_fun(alpha, pre) if alpha else left_fun(pre)

    return post

def rank(left: str|list[tuple], right: arr.Array, alpha: Optional[arr.Array], omega: arr.Array, env:dict[str, Value], stack:Stack) -> arr.Array:
    """
    Rank - dyadic operator ⍤ deriving ambivalent

    https://aplwiki.com/wiki/Rank_(operator)
    https://help.dyalog.com/latest/index.htm#Language/Primitive%20Operators/Rank.htm
    https://xpqz.github.io/learnapl/rank.html
    https://xpqz.github.io/cultivations/Rank.html
    https://wiki.nars2000.org/index.php?title=Rank
    https://dl.acm.org/doi/pdf/10.1145/55626.55632

    Note that the right operand is always an integer scalar or vector of two elements, 
    not a function. The tree-element version is nyi.
    """
    if right.rank > 1:
        raise RankError('RANK ERROR')

    if right.bound > 2:
        raise NYIError('NYIError: right operand can only be length 1 or 2')

    arity = Arity.DYAD if alpha else Arity.MONAD
    fun = _make_operand(left, arity, env, stack)

    # Deriving monad
    if arity == Arity.MONAD:
        # If right is zero or positive it selects k-cells of the corresponding 
        # argument. If it is negative, it selects (r+right)-cells where r is the  
        # rank of the corresponding argument. A value of ¯1 selects major cells.
        selector = right.data[0]
        if selector < 0:
            selector = omega.rank - selector
        if selector > omega.rank:
            selector = omega.rank

        shape = omega.shape[:-selector]

        return arr.Array(shape, [fun(cell).disclose() for cell in omega.kcells(selector)])

    # Deriving dyad
    assert alpha
    l_selector = right.data[0]
    if l_selector < 0:
        l_selector = alpha.rank - l_selector
    if l_selector > alpha.rank:
        l_selector = alpha.rank

    try:
        r_selector = right.data[1]
    except IndexError:
        r_selector = right.data[0]

    if r_selector < 0:
        r_selector = omega.rank - r_selector
    if r_selector > omega.rank:
        r_selector = omega.rank

    acells = alpha.kcells_array(l_selector)
    wcells = omega.kcells_array(r_selector)

    if acells.bound == 1 and not acells.issimple():   # Extend left
        acells = acells.reshape(wcells.shape)
    elif wcells.bound == 1 and not wcells.issimple(): # Extend right
        wcells = wcells.reshape(acells.shape)

    return fun(acells, wcells).mix()

def decode(alpha: arr.Array, omega: arr.Array) -> arr.Array:
    """
    Decode - dyadic ⊥

    See https://aplwiki.com/wiki/Decode
        https://xpqz.github.io/cultivations/Decode.html
        https://help.dyalog.com/latest/index.htm#Language/Primitive%20Functions/Decode.htm

    This is a facade of arr.Array.decode(), handling a few rank combinations:

    2 ⊥ 1 1 0 1
  
    13

    24 60 60 ⊥ 2 46 40

    10000

    Note that we're really doing an inner product:

    (4 3⍴1 1 1 2 2 2 3 3 3 4 4 4)⊥3 8⍴0 0 0 0 1 1 1 1 0 0 1 1 0 0 1 1 0 1 0 1 0 1 0 1
    ┌→──────────────────┐
    ↓0 1 1 2  1  2  2  3│
    │0 1 2 3  4  5  6  7│
    │0 1 3 4  9 10 12 13│
    │0 1 4 5 16 17 20 21│
    └~──────────────────┘

    Dyalog's docs say:

        R←X⊥Y

        Y must be a simple numeric array.  X must be a simple numeric array.  R is the 
        numeric array which results from the evaluation of Y in the number system with radix X.

        X and Y are conformable if the length of the last axis of X is the same as the length 
        of the first axis of Y.  A scalar or 1-element vector is extended to a vector of the 
        required length.  If the last axis of X or the first axis of Y has a length of 1, the 
        array is extended along that axis to conform with the other argument.

        The shape of R is the catenation of the shape of X less the last dimension with the 
        shape of Y less the first dimension.  That is:

        ⍴R ←→ (¯1↓⍴X),1↓⍴Y

        For vector arguments, each element of X defines the ratio between the units for corresponding 
        pairs of elements in Y.  The first element of X has no effect on the result.

    """
    if alpha.issimple(): # Left scalar is extended
        left = alpha.reshape([omega.shape[0]])
    else:
        left = alpha

    if omega.issimple(): # Right scalar is extended, too
        right = omega.reshape([left.shape[0]])
    else:
        right = omega

    # If the last axis of X or the first axis of Y has a length of 1, the 
    # array is extended along that axis to conform with the other argument.
    if left.shape[-1] == 1: # extend trailling
        shape = left.shape[:]      
        shape[-1] = right.shape[0]
        ravel = []
        for i in left.data:
            ravel.extend([i]*right.shape[0])
        left = arr.Array(shape, ravel)

    if right.shape[0] == 1: # extend leading
        shape = right.shape[:]
        shape[0] = left.shape[-1]
        right = right.reshape(shape)

    if left.shape[-1] != right.shape[0]:
        raise RankError('RANK ERROR')

    if left.rank == right.rank == 1:
        return arr.S(arr.decode(left.data, right.data))

    # At least one side is hirank; we're doing an inner product
    shape = left.shape[:-1]+right.shape[1:]
    if left.rank == 1: # Treat vectors as 1-row matrix
        left.shape = [1, left.shape[0]]

    right = right.transpose()
    ravel = []
    for lc in left.major_cells():
        for rc in right.major_cells():
            decoded = arr.decode(lc.data, rc.data)
            ravel.append(decoded)

    return arr.Array(shape, ravel)

def encode(alpha: arr.Array, omega: arr.Array) -> arr.Array:
    """
    Encode - dyadic ⊤

    See https://aplwiki.com/wiki/Encode
        https://xpqz.github.io/cultivations/Encode.html

    This is a facade of arr.Array.encode(), handling a few rank combinations:

    24 60 60 ⊤ 10000
    ┌→──────┐
    │2 46 40│
    └~──────┘

    10⊤5 15 125
    ┌→────┐
    │5 5 5│
    └~────┘

    2 2 2 2 ⊤ 5 7 12
    ┌→────┐
    ↓0 0 1│
    │1 1 1│
    │0 1 0│
    │1 1 0│
    └~────┘

    (8 3⍴2 10 16)⊤121
    ┌→────┐
    ↓0 0 0│
    │1 0 0│
    │1 0 0│
    │1 0 0│
    │1 0 0│
    │0 1 0│
    │0 2 7│
    │1 1 9│
    └~────┘
    """
    if omega.rank > 1:
        raise RankError('RANK ERROR') # Can ⍵ even be of rank > 1?

    if omega.issimple() and alpha.rank == 1:
        try:
            return arr.V(arr.encode(alpha.data, omega.data[0]))
        except TypeError:
            raise DomainError('DOMAIN ERROR')

    if alpha.issimple() and omega.rank == 1:
        try:
            return arr.V([arr.encode(alpha.data, o)[0] for o in omega.data])
        except TypeError:
            raise DomainError('DOMAIN ERROR')
        
    if alpha.rank == 1:
        radix = [c.data for c in alpha.reshape([len(omega.data)]+alpha.shape).major_cells()]
    else:
        radix = [c.data for c in alpha.transpose().major_cells()]

    try:
        data = [arr.encode(*args) for args in itertools.zip_longest(radix, omega.data, fillvalue=omega.data[0])]
    except TypeError:
        raise DomainError('DOMAIN ERROR')

    return arr.Array([len(data[0]), len(data)], list(itertools.chain.from_iterable(zip(*data))))

def rho(alpha: Optional[arr.Array], omega: arr.Array) -> arr.Array:
    """
    Monadic: shape
    Dyadic: reshape

    Apply the shape alpha to the ravel of omega. The rank of alpha must be 0 or 1

    APL> ,5 5⍴3 3⍴3 2 0 8 6 5 4 7 1
    3 2 0 8 6 5 4 7 1 3 2 0 8 6 5 4 7 1 3 2 0 8 6 5 4

    >>> rho(arr.V([5, 5]), arr.Array([3, 3], [3, 2, 0, 8, 6, 5, 4, 7, 1]))
    A([5, 5], UINT8, FLAT, [3, 2, 0, 8, 6, 5, 4, 7, 1, 3, 2, 0, 8, 6, 5, 4, 7, 1, 3, 2, 0, 8, 6, 5, 4])

    >>> rho(arr.V([2, 2]), arr.S(5))
    A([2, 2], UINT8, FLAT, [5, 5, 5, 5])

    >>> rho(None, arr.Array([2, 2], [1, 2, 3, 4]))
    V(UINT8, FLAT, [2, 2])
    """
    if alpha is None: # Monadic
        return arr.V(omega.shape)

    return omega.reshape(alpha.data)

def ravel(omega: arr.Array) -> arr.Array:
    """
    Monadic , - create a vector of all major cells contained in omega
    """
    return arr.V(omega.data)
                
def tally(omega: arr.Array) -> arr.Array:
    if omega.issimple():
        return arr.S(1)
    return arr.S(len(list(omega.major_cells())))

def enclose(omega: arr.Array) -> arr.Array:
    if omega.issimple():
        return omega
    return arr.S(omega)

def or_gcd(alpha: arr.Array, omega: arr.Array) -> arr.Array:
    if any(x not in [0, 1] for x in alpha.data) or any(x not in [0, 1] for x in omega.data):
        f = pervade(math.gcd)
    else:
        f = pervade(lambda x, y:x|y, direct=True)
    return f(alpha, omega)

def and_lcm(alpha: arr.Array, omega: arr.Array) -> arr.Array:
    if any(x not in [0, 1] for x in alpha.data) or any(x not in [0, 1] for x in omega.data):
        f = pervade(math.lcm)
    else:
        f = pervade(lambda x, y:x&y, direct=True)
    return f(alpha, omega)

def bool_not(omega: arr.Array) -> arr.Array:
    if not omega.nested:
        try:
            ba = bitarray(omega.data)
        except ValueError:
            raise DomainError("DOMAIN ERROR: expected boolean array")
        return arr.Array(omega.shape, list(~ba))

    f = mpervade(lambda y:y^1)
    return f(omega)

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

def ucs(o: Any) -> Any:
    return ord(o) if type(o) == str else chr(o)

def conj(o: Any) -> Any:
    if type(o) == complex:
        return o.conjugate()
    return o

def recip(o: Any) -> Any:
    if isinstance(o, (int, float, complex)):
        return 1/o
    raise DomainError('DOMAIN ERROR')

def flr(o: Any) -> Any:
    if type(o) == complex:
        return complex(math.floor(o.real), math.floor(o.imag))
    if isinstance(o, (int, float)):
        return int(math.floor(o))
    raise DomainError('DOMAIN ERROR')

def ceiling(o: Any) -> Any:
    if type(o) == complex:
        return complex(math.ceil(o.real), math.ceil(o.imag))
    if isinstance(o, (int, float)):
        return math.ceil(o)
    raise DomainError('DOMAIN ERROR')

def roll(o: int) -> arr.Array:
    if type(o) != int:
        raise DomainError('DOMAIN ERROR: expected simple integer')
    if o == 0:
        return arr.S(random.random())
    return arr.S(secrets.randbelow(o+1))

def deal(alpha: arr.Array, omega: arr.Array) -> arr.Array:
    if not omega.issimple() or type(omega.data[0]) != int:
        raise LengthError('LENGTH ERROR')
    if (alpha.issimple() or alpha.shape == [1, 1]) and type(alpha.data[0]) == int:
        return arr.V(random.sample(range(omega.data[0]), alpha.data[0]))
    else:
        raise LengthError('LENGTH ERROR') 

class Voc:
    """
    Voc is the global vocabulary of built-in arrays, functions and operators. This class should not
    be instantiated.
    """
    arrs: dict[str, arr.Array] = {
        '⍬': arr.Array([0], []),
        '⎕IO': arr.S(0),
        '⎕A': arr.V(ascii_letters[26:]),
        '⎕D': arr.V(list(range(10))),
    }

    funs: dict[str, Signature] = {
        #--- Monadic-------------------------Dyadic---------------------------
        '↑': (lambda o: o.mix(),                      lambda a, o: o.take(a)),
        '↓': (lambda o: o.split(),                    lambda a, o: o.drop(a)),
        '⍪':  (lambda o: o.table(),                   lambda a, o: a.laminate(o)),
        '~': (bool_not,                               lambda a, o: a.without(o)),
        '?': (mpervade(roll),                         deal),
        '⍋': (lambda o: arr.V(o.grade()),             None),
        '⍒': (lambda o: arr.V(o.grade(reverse=True)), None),
        '⊤': (None,                                   encode),
        '⊥': (None,                                   decode),
        '⊣': (lambda o: o,                            lambda a, o: a),
        '⊢': (lambda o: o,                            lambda a, o: o),
        '∨': (None,                                   or_gcd),
        '∧': (None,                                   and_lcm),
        '∊': (lambda o: o.enlist(),                   lambda a, o: o.contains(a)),
        '⊂': (enclose,                                None),
        '⊃': (lambda o: o.disclose(),                 lambda a, o: o.pick(a)),
        ',': (ravel,                                  lambda a, o: a.catenate(o)),
        '⍉': (lambda o: o.transpose(),                lambda a, o: o.transpose(a.as_list())),
        '⍴': (lambda o: rho(None, o),                 rho),
        '⍳': (lambda o: o.index_gen(),                lambda a, o: a.index_of(o)),
        '⍸': (lambda o: o.where(),                    None),
        '≢': (tally,                                  lambda x, y: arr.S(int(not arr.match(x, y)))),
        '≡': (lambda o: arr.S(arr.Array.depth(o)),    lambda x, y: arr.S(int(arr.match(x, y)))),
        '⌈': (mpervade(ceiling),                      pervade(max)),
        '⌊': (mpervade(flr),                          pervade(min)),
        '!': (mpervade(math.factorial),               None),
        '○': (mpervade(lambda o: o*math.pi),          pervade(circle)),
        '+': (mpervade(conj),                         pervade(operator.add)),
        '-': (mpervade(operator.neg),                 pervade(operator.sub)),
        '×': (mpervade(lambda o:o/abs(o)),            pervade(operator.mul)),
        '÷': (mpervade(recip),                        pervade(lambda a, o: a/o)),
        '*': (mpervade(math.exp),                     pervade(operator.pow)),
        '|': (mpervade(operator.abs),                 pervade(lambda a, o: o%a)),       # DYADIC NOTE ARG ORDER
        '=': (None,                                   pervade(lambda a, o: int(a==o))),
        '>': (None,                                   pervade(lambda a, o: int(a>o))),
        '<': (None,                                   pervade(lambda a, o: int(a<o))),
        '≠': (None,                                   pervade(lambda a, o: int(a!=o))),
        '≥': (None,                                   pervade(lambda a, o: int(a>=o))),
        '≤': (None,                                   pervade(lambda a, o: int(a<=o))),
        '⎕UCS': (mpervade(ucs),                       None),
    }

    ops: dict[str, Operator] = {
        #-------------Implem--------Derived-isa--Self-isa-----L-oper-isa--------------------------R-oper-isa
        '/': Operator(reduce,       Arity.MONAD, Arity.MONAD, OperandType.DYAD,                   None),
        '⌿': Operator(reduce_first, Arity.MONAD, Arity.MONAD, OperandType.DYAD,                   None),
        '¨': Operator(each,         Arity.MONAD, Arity.MONAD, OperandType.MONAD,                  None),
        '⍨': Operator(commute,      Arity.DYAD,  Arity.MONAD, OperandType.DYAD,                   None),
        '⍥': Operator(over,         Arity.DYAD,  Arity.DYAD,  OperandType.DYAD,                   OperandType.MONAD),
        '⍤': Operator(rank,         Arity.AMBIV, Arity.DYAD,  OperandType.MONAD|OperandType.DYAD, OperandType.ARRAY),
        '⍣': Operator(power,        Arity.MONAD, Arity.DYAD,  OperandType.MONAD|OperandType.DYAD, OperandType.ARRAY|OperandType.DYAD),
        '@': Operator(amend,        Arity.AMBIV, Arity.DYAD,  OperandType.ARRAY|OperandType.DYAD, OperandType.ARRAY|OperandType.MONAD),
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

