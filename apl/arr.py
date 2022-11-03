from collections import defaultdict
from copy import deepcopy
import itertools
import math
from typing import Any, Callable, Generator, Optional, Sequence

from bitarray import bitarray

from apl.errors import DomainError, LengthError, RankError

def strides(shape: list[int]) -> list[int]:
    r = [0 for _ in range(len(shape))]
    u = 1
    for i in range(len(r)-1, -1, -1):
        r[i] = u
        u *= shape[i]
    return r

class Array:
    """
    An Array has a sequence of items, and a integer-valued shape vector. Each item
    in the data sequence is either a scalar (int, float, complex or length-1 string, 
    or another Array.
    """
    def __init__(self, shape: list[int], data: Sequence) -> None:
        assert type(data) == list
        self.shape = shape
        self.rank = len(shape)
        self.bound = math.prod(shape)
        self.nested = False

        # Turn strings of len > 1 into character vectors, and check for
        # flat vs nested.
        for i in range(len(data)):
            if type(data[i]) == str and len(data[i]) > 1:
                data[i] = Array([len(data[i])], list(data[i]))
            if isinstance(data[i], Array):
                if data[i].issimple():
                    data[i] = data[i].data[0]
                else:
                    self.nested = True
        self.data = data

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Array):
            return NotImplemented
        return match(self, other)

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, Array):
            return NotImplemented
        for i, e in enumerate(self.data):
            if i>=len(other.data):
                return False
            if e<other.data[i]:
                return True
        return False

    def issimple(self):
        return self.shape == [] and len(self.data) == 1 and not isinstance(self.data[0], Array)

    def unbox(self) -> 'Array':
        if self.issimple():
            return self
        if not self.shape and self.bound == 1:
            return self.data[0]
        return self

    def as_list(self) -> list:
        if self.rank != 1:
            raise RankError
        data = []
        for e in self.data:
            if isinstance(e, Array):
                if not e.issimple():
                    raise DomainError
                data.append(e.data[0])
            else:
                data.append(e)
        return data

    @classmethod
    def fill(cls, shape: list[int], data: Sequence) -> 'Array':
        """
        Create new array from shape and data, repeating items circularly if we're not 
        given sufficient numbers, or drop elements if we have too many.
        """
        return cls(shape, list(itertools.islice(itertools.cycle(data), math.prod(shape))))

    def index_gen(self) -> 'Array':
        """
        ⍳ - index generator (monadic)
        """
        if self.rank > 1: raise RankError
        if type(self.data[0]) != int: raise DomainError
        if not self.shape: return V(list(range(self.data[0])))
        return Array(self.data, [V(c) for c in Array.coords(self.data)])

    def get(self, coords: Any) -> Any:
        """
        Get item at coords.
        """
        if type(coords) == int:
            coords = [coords]
        elif isinstance(coords, Array):
            if coords.rank != 1:
                raise RankError
            coords = coords.data
        return self.data[_ravel_index(self.shape, coords)]

    def at(self, idx: 'Array') -> 'Array':
        """ 
        Basic and choose indexing. See https://aplwiki.com/wiki/Bracket_indexing
        Reach indexing nyi
        """
        return Array(idx.shape, [self.get(c) for c in idx.data])

    def rect(self, spec: list[list[int]]) -> 'Array':
        """
        APL Wiki:

          For higher-rank array X with rank n, the notation X[Y1;Y2;...;Yn] selects the 
          indexes of X over each axis. If some Yk is omitted, it implies all indices of 
          k-th axis is selected, which is equivalent to specifying ⍳(⍴X)[k]. The resulting 
          shape is the concatenation of shapes of Y1, Y2, ..., Yn. 

        a←3 3⍴⍳9
        a[;2]
        ┌→────┐
        │2 5 8│
        └~────┘

        a = Array([3, 3], list(range(9))
        a.rect([[],[2]])
        """
        if len(spec) != self.rank:
            raise RankError

        # Empty spec implies a full selection as given by shape
        for axis in range(len(spec)):
            if not spec[axis]:
                spec[axis] = list(range(self.shape[axis]))

        data = []
        for coords in itertools.product(*spec):
            data.append(self.get(coords))

        # Scalar selections are excluded from shape; see 
        #   https://aplwiki.com/wiki/Bracket_indexing
        shape = [len(a) for a in spec if len(a) > 1] 

        return Array(shape, data)

    @staticmethod
    def coords(shape: list[int]) -> Generator:
        """
        Generator. Step through the space defined by the shape, generating 
        each coordinate vector in turn.

        >>> list(Array.coords([2, 3]))
        [[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2]]
        """
        rnk = len(shape)
        bound = math.prod(shape)
        offsets = [0 for _ in range(rnk)]
        coords = [0 for _ in range(rnk)]
        yield coords[:]
        for idx in range(1, bound):
            axis = rnk - 1
            coords[axis] += 1
            while axis>0 and coords[axis] == shape[axis] + offsets[axis]:
                coords[axis] = 0
                coords[axis-1] += 1
                axis -= 1
            yield coords[:]

    def kcells(self, k: int) -> Generator:
        """
        Generator. Yield cells, dropping the first rank-k axes.
        """
        if k > self.rank:
            raise RankError

        if k == self.rank:
            return Array([], self.data[:])

        # Shape and bound of result
        rsh = self.shape[:self.rank-k]
        rbnd = math.prod(rsh)

        # Shape and bound of each cell
        csh = self.shape[self.rank-k:]
        cbnd = math.prod(csh)
        for cell in range(rbnd):
            yield Array(csh, self.data[cell*cbnd:(cell+1)*cbnd])

    def major_cells(self) -> Generator:
        """
        Generator. An array's major cells are the cells along the leading axis.
        """
        if not self.shape:
            return self

        size = math.prod(self.shape[1:])
        for i in range(self.shape[0]):
            yield Array(self.shape[1:], self.data[i*size:(i+1)*size])

    def prot(self) -> 'Array':
        """
        Prototypal element.
        """
        if not len(self.data):
            if type(self.data) == str:
                return Array([], [" "])
            return Array([], [0])

        first = self.data[0]
        if not isinstance(first, Array):
            elements = Array([], [first])
        else:
            elements = deepcopy(first)

        def inner(o: Array) -> None:
            for i, e in enumerate(o.data):
                if not isinstance(e, Array):
                    if isinstance(e, (int, float, complex)):
                        o.data[i] = 0
                    elif type(e) == str:
                        o.data[i] = " "
                else:
                    inner(e)

        inner(elements)
        return elements

    def reshape(self, shape: list[int]) -> 'Array':
        """
        Dyadic ⍴ -- reshape. Will truncate, or cycle the data if the 
        new shape implies fewer or more elements that current.
        """
        if not self.bound: # ⍴ coerces the prototype if there is no data
            data = [Array.prot(self)]
        else:
            data = deepcopy(self.data) # type: ignore

        if len(data) == math.prod(shape):
            return Array(shape, data)

        return Array.fill(shape, data)

    def take(self, x: 'Array') -> 'Array':
        """
        Dyadic ↑ -- https://aplwiki.com/wiki/Take

        Note: this one adapted from ngn/apl:
        
            https://github.com/abrudz/ngn-apl/blob/master/apl.js#L970
        """
        if not len(self.shape): # disclose
            y = Array([1 for _ in range(x.shape[0])] if len(x.shape) else [1], [self.data[0]])
        else:
            y = self

        shape = y.shape[:]
        for i in range(len(x.data)):
            if type(x.data[i]) != int:
                raise DomainError
            if i >= len(shape):
                shape.append(abs(x.data[i]))
            else:
                shape[i] = abs(x.data[i])

        d = [0 for _ in range(len(shape))]
        d[len(d)-1] = 1
        
        for i in range(len(d)-1, 0, -1):
            d[i-1] = d[i] * shape[i]

        cs = shape[:]  # shape to copy
        p = 0
        q = 0
        xd = strides(y.shape)

        for i in range(len(x.data)):
            u = x.data[i]
            cs[i] = min(y.shape[i], abs(u))
            if u < 0:
                if u < -y.shape[i]:
                    q -= (u + y.shape[i]) * d[i]
                else:
                    p += (u + y.shape[i]) * xd[i]

        ravel = []
        for i in range(math.prod(shape)):
            prot = y.prot()
            if not prot.shape and len(prot.data) == 1: # keep simple scalars unboxed
                prot = prot.data[0]
            ravel.append(prot)

        if math.prod(cs):
            ci = [0 for _ in range(len(cs))] # indices for copying

            while q<len(ravel): 
                ravel[q] = y.data[p]
                h = len(cs)-1
                while h >= 0 and ci[h] + 1 == cs[h]:
                    p -= ci[h] * xd[h]
                    q -= ci[h] * d[h]
                    h -= 1
                    ci[h] = 0
                
                if h < 0: break
                p += xd[h]
                q += d[h]
                ci[h] += 1

        if not ravel and shape == [0]:
            return Array(shape, y.prot().data)
        return Array(shape, ravel)

    def drop(self, a):
        """
        Dyadic ↓ -- https://aplwiki.com/wiki/Drop

        Drop ← {
            s ← ⍴⍵
            s ,← ((0=≢s)×≢⍺)⍴1          ⍝ Scalar rank extension
            s ← (≢⍺)↑s                  ⍝ SHARP APL extension
            ((s×¯1*⍺>0) + (-s)⌈s⌊⍺) ↑ ⍵ ⍝ Note: exp
        }
        
        """
        if not a.bound:
            return self

        s = self.shape[:]
        tally_a = len(a.data)

        s.extend([1 for _ in range(int(0==len(s))*tally_a)]) # Scalar rank extension
        s = V(s).take(S(tally_a))                            # SHARP APL extension

        spec = [
            ss*(-1)**int(a.data[i]>0) + max(-ss, min(ss, a.data[i]))  # ((s×¯1*⍺>0) + (-s)⌈s⌊⍺)
            for i, ss in enumerate(s.data)
        ]

        return self.take(V(spec))

    def mix(self) -> 'Array':
        """
        Monadic ↑ - trade depth for rank

        See https://aplwiki.com/wiki/Mix
        """
        if not len(self.shape) or not math.prod(self.shape) or (self.rank == 1 and not self.nested): return self
        shape = self.shape[:]
        shapes = [enclose_if_simple(e).shape[:] for e in self.data]
        r = max(len(s) for s in shapes)

        sshapes = []
        for s in shapes: # (⍴↓(r⍴1)∘,)¨shapes ⍝ prepend 1 to each shape to equal max length
            rr1 = [1]*r
            if len(s):
                rr1[-len(s):] = s
            sshapes.append(rr1)

        smax = [max(v) for v in zip(*sshapes)] # max per axis
        smax_v = V(smax)  
        ravel = []
        for i, cell in enumerate(self.data):
            elem = enclose_if_simple(cell)
            reshaped = elem.reshape(sshapes[i])
            taken = reshaped.take(smax_v)
            ravel.extend(taken.data)

        return Array(shape+smax, ravel)

    def split(self):
        """
        R ← ↓self
        Self may be any array. Trailling axis
        The items of R are the sub-arrays of self along the trailling axis.  R is a scalar if self is a scalar. 
        """
        if not self.bound: # if self is empty, return enclosed prototype
            return Array([], [self.prot()])

        if not self.shape: # scalar
            return self

        size = self.shape[self.rank-1]
        count = math.prod(self.shape)//size
        ravel = [None for _ in range(count)]
        for i in range(count):
            c = [self.data[j+i*size] for j in range(size)]
            ravel[i] = Array([len(c)], c)

        return Array(self.shape[:-1], ravel)

    def foldr(self, operand: Callable, axis: Optional[int] = None) -> 'Array':
        """
        Dyadic / - reduction along axis. Trailling axis by default. APL's reduce is `foldr`. Sadly.

        Note that this has operand-dependent behaviour when applied to empty arrays:

            +/⍬
        0
 
            ⌊/⍬               
        1.797693135E308  ⍝ Inf

        These will need to be handled by the caller of this function, as we don't know what the 
        operand does by the time we land here.

        Originally by @ngn: 
        * https://chat.stackexchange.com/transcript/message/47158587#47158587
        * https://pastebin.com/mcPAxtqx

        Also used in dzaima/apl:
        * https://github.com/dzaima/APL/blob/master/src/APL/types/functions/builtins/mops/ReduceBuiltin.java#L156-L178

        """
        if not self.bound:
            raise DomainError("DOMAIN ERROR: Reduction over empty array unhandled")

        if axis is None: # Trailling axis by default.
            axis = self.rank - 1

        if not self.rank: return self
        if axis < 0: axis += self.rank
        if axis >= self.rank: raise RankError("RANK ERROR: Invalid axis")

        before = math.prod(self.shape[:axis])            # prouct of dims before axis
        red = self.shape[axis]                           # reduction axis size
        after = math.prod(self.shape[axis+1:self.rank])  # product of dims after axis

        shape = self.shape[:]; del shape[axis]           # new shape is old shape with the reducing axis removed
        ravel = [0 for _ in range(before*after)]

        for i in range(before):
            for k in range(after):
                acc = self.data[i*red*after + (red-1)*after + k]
                for j in range(red-2, -1, -1):           # R-L
                    acc = operand(self.data[i*red*after + j*after + k], acc)
                ravel[i*after + k] = acc
        
        return Array(shape, ravel)

    def transpose(self, alpha: Optional[list[int]] = None) -> 'Array':
        """
        Monadic and dyadic transpose a⍉b.
        
        Dyadic transpose is a generalisation of the monadic case, which reverses
        the axes. In the dyadic case, axes can be in any order. 
        
        The general dyadic transpose: reorder axes

        ,2 0 1⍉2 3 4⍴⍳×/2 3 4
        ┌→────────────────────────────────────────────────────────────┐
        │0 12 1 13 2 14 3 15 4 16 5 17 6 18 7 19 8 20 9 21 10 22 11 23│
        └~────────────────────────────────────────────────────────────┘

        Complication: repeated axes in the dyadic case:

        0 0⍉3 3⍴⍳9 ⍝ Dyadic transpose with repeated axes gets the diagonal
        ┌→────┐
        │0 4 8│
        └~────┘
        """
        if self.rank < 2: # For scalars and vectors: identity
            return deepcopy(self)   # NOTE: need to clone here, wtf Guido

        # For the monadic transpose: transformation is the reverse axis ordering
        repeated_axes = {}
        if not alpha:
            alpha = list(range(self.rank-1, -1, -1))
        else: # Check for repeated axes in the axis reordering spec
            repeated_axes = _repeated(alpha)

        new_shape = _reorder_axes(alpha, self.shape)
        newdata = [0]*math.prod(new_shape)
        for idx, cvec in enumerate(Array.coords(self.shape)):
            if repeated_axes and _skip(repeated_axes, cvec):
                continue
            newdata[decode(new_shape, _reorder_axes(alpha, cvec))] = self.data[idx]

        # Check if we can make this flat.
        for e in newdata:
            if isinstance(e, Array):
                return Array(new_shape, deepcopy(newdata))
        
        return Array(new_shape, deepcopy(newdata)) # Possibly don't need deepcopy() here.

    def enlist(self) -> 'Array':
        """
        Monadic ∊ - create a vector of all simple scalars contained in self, recursively
        drilling into any nesting and shapes.
        """
        ravel = []

        def inner(o: Any) -> None:
            for e in o.data:
                if not isinstance(e, Array):
                    ravel.append(e)
                else:
                    inner(e) # type: ignore
        inner(self)

        return V(ravel)

    def where(self) -> 'Array':
        """
        Where - mondadic ⍸ - create a vector of the indices of 1s in a Boolean array
        """
        try:
            bool_arr = bitarray(self.data)
        except:
            raise DomainError('DOMAIN ERROR: expected Boolean array')

        if self.rank == 1:
            return Array([bool_arr.count()], bool_arr.search(bitarray([True])))

        return V([
            Array([self.rank], encode(self.shape, idx))
            for idx in bool_arr.search(bitarray([True]))
        ])

    def laminate(self, omega: 'Array') -> 'Array':
        """
        Laminate, (catenate-first) ⍺⍪⍵ -- the major cells of omega are appended as major cells to self
        
        If omega is scalar, it's extended to fit
        """
        if not omega.shape and len(omega.data) == 1:
            omega = Array.fill([1]+self.shape[1:], omega.data)

        if self.rank > omega.rank:
            omega = omega.reshape([1]+omega.shape)

        if self.shape[1:] != omega.shape[1:]: raise LengthError

        data = deepcopy(self.data)
        for c in omega.major_cells():
            data.extend(c.data)
        shape = self.shape[:]
        shape[0] += omega.shape[0]

        return Array(shape, data)

    def table(self) -> 'Array':
        """
        Monadic ⍪ -- https://aplwiki.com/wiki/Table

        Table is equivalent to reshaping with the shape where all trailing axis lengths have been 
        replaced by their product.
        """
        if self.issimple():
            return Array([1, 1], self.data[:])
        if self.rank == 1:
            return Array(self.shape[:]+[1], deepcopy(self.data))
        shape = [self.shape[0]]+[math.prod(self.shape[1:])]
        return Array(shape, deepcopy(self.data))

    def disclose(self) -> 'Array':
        if self.issimple():
            return self

        if not self.bound: # empty coerces the prototype
            return self.prot()
        
        if isinstance(self.data[0], Array):
            return self.data[0]
        
        return Array([], [self.data[0]])

    def mutate(self, idx: 'Array', vals: 'Array') -> None:
        """
        Only for a subset of indexing modes for now.

        See https://aplwiki.com/wiki/Bracket_indexing
        """
        if idx.shape != vals.shape:
            raise RankError("RANK ERROR")

        if idx.rank == 0:
            if not idx.nested:
                loc = [_ravel_index(self.shape, idx.data)]
            else:
                loc = [_ravel_index(self.shape, idx.data[0].data)]
        else:
            loc = [_ravel_index(self.shape, _list(c)) for c in idx.data]
            
        valc = 0
        for c in loc:
            self.data[c] = vals.data[valc] # type: ignore
            valc += 1

    def contains(self, arr: 'Array') -> 'Array':
        """
        Dyadic ∊ - which cells in arr are found as cells in self?
        """
        result = []

        if self.issimple():
            for ac in Array.coords(arr.shape):
                cell = enclose_if_simple(arr.get(ac))
                result.append(int(match(cell, self)))
        else:
            for ac in Array.coords(arr.shape):
                cell_a = enclose_if_simple(arr.get(ac))
                found = 0
                for oc in Array.coords(self.shape):
                    cell_o = enclose_if_simple(self.get(oc))
                    if match(cell_a, cell_o):
                        found = 1
                        break
                result.append(found)

        return Array(arr.shape, result)

    def replicate(self, mask: 'Array') -> 'Array':
        """
        Dyadic X/Y - for arrays X and Y: https://aplwiki.com/wiki/Replicate
        """
        if mask.rank > 1:
            raise RankError

        if mask.rank == 0: # If argument is scalar, we need to extend to the shape of self
            a = mask.reshape(self.shape)
        else:
            a = mask

        return V(list(itertools.chain.from_iterable(itertools.repeat(self.data[i], a.data[i]) for i in range(a.bound))))

    def grade(self, reverse: bool = False) -> list:
        """
        Grade - monadic ⍋ ⍒

        https://aplwiki.com/wiki/Grade
        """
        cells = list(self.major_cells())
        perm = [(cells[i], i) for i in range(len(cells))]
        perm.sort(reverse=reverse)
        return [t[1] for t in perm]

    def without(self, arr: 'Array') -> 'Array':
        """
        Dyadic ~ 

        https://aplwiki.com/wiki/Without
        
        """
        mask = arr.contains(self)
        mask.data = list(~bitarray(mask.data)) # type: ignore
        return self.replicate(mask)
            
    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        if len(self.shape) == 1:
            return f"V({self.shape}, {self.data})"
        return f"A({self.shape}, {self.data})"

# A few convenience pseudo-constructors
def V(data: list|str) -> Array:
    if type(data) == str:
        return Array([len(data)], list(data))
    return Array([len(data)], data)

def S(item: int|float|complex|Array|str) -> Array:
    return Array([], [item])

def match(a: 'Array', w: 'Array') -> bool:

    if a.shape != w.shape:
        return False

    if not a.shape: # Simple scalar, or enclosed something
        if isinstance(a.data[0], Array):
            return match(a.data[0], w.data[0]) # type: ignore
        else:
            return a.data[0] == w.data[0]

    for i in range(a.bound):
        left = a.data[i]
        right = w.data[i]
        if not isinstance(left, Array):
            if not isinstance(left, Array):
                if left != right:
                    return False
            else:
                return False
        elif not match(left, right):
            return False

    return True

def encode(shape: list[int], idx: int) -> list[int]:
    """
    Encode -- dyadic ⊤
    
    Returns the coordinate vector into shape corresponding to the 
    linear index idx into its ravel vector

    https://aplwiki.com/wiki/Encode

    Inverse of `decode()`

    >>> encode([24, 60, 60], 10_000)
    [2, 46, 40]
    """
    encoded: list[int] = []
    for axis in shape[::-1]:
        idx, loc = divmod(idx, axis)
        encoded.append(loc)
    return encoded[::-1]

def decode(shape: list[int], coords: list[int]) -> int:
    """
    Decode -- dyadic ⊥

    Evaluates `coords` in terms of the radix system defined by `shape`.
    
    Inverse of `encode()`

    https://aplwiki.com/wiki/Decode

    """
    pos = 0
    rnk = len(shape)
    for axis in range(rnk):
        if axis >= len(coords):
            return pos
        pos += coords[axis]
        if axis != rnk - 1:
            pos *= shape[axis+1]
    return pos

# Private helpers
def _list(a: Any) -> list:
    if isinstance(a, list):
        return a

    if isinstance(a, Array):
        return a.data

    return [a]

def _ravel_index(shape: list[int], coords: list[int]) -> int:
    for i, a in enumerate(coords):
        if a<0 or a>=shape[i]:
            raise IndexError
    return decode(shape, coords)

def enclose_if_simple(a: Any) -> Array:
    return a if isinstance(a, Array) else S(a)

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

