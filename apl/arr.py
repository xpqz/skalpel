import itertools
import math
from typing import Any, Iterator, Sequence
from bitarray import bitarray
from apl.errors import DomainError, RankError
from enum import Enum

class DataType(Enum):
    UINT1 = 0  # Boolean
    UINT8 = 1  # 8-bit unsigned int
    INT   = 2  # Any int
    CHAR  = 3  # UTF-8
    FLOAT = 4  # Any floating point
    CMPLX = 5  # Complex
    MIXED = 6  # Can we be mixed without also being nested? 1 2.3 'a'

    def __str__(self):
        return self.name

class ArrayType(Enum):
    FLAT   = 0
    NESTED = 1

    def __str__(self):
        return self.name

class Array:

    def __init__(self, shape: list[int], data_type: DataType, array_type: ArrayType, data: bitarray|bytearray|list) -> None:
        """
        Raw init -- does no validation or data conversion, bar calculating rank and bound.
        """
        self.shape = shape
        self.rank = len(shape)
        self.bound = math.prod(shape)
        self.data = data
        self.type = data_type
        self.array_type = array_type

    @classmethod
    def from_sequence(cls, shape: list[int], data_type: DataType, array_type: ArrayType, data: Sequence):
        """
        from_sequence builds an array from a sequence of things, including other arrays:

        1. Pick ×/shape elements from data (repeating, if necessary).
        2. Select an array provider (currently one of: bitarray, bytearray or list) based on data_type
        """
        d = list(itertools.islice(itertools.cycle(data), math.prod(shape))) # note: can be sub-Arrays
        if array_type == ArrayType.FLAT:
            if data_type == DataType.UINT1:
                return cls(shape, data_type, array_type, bitarray(d))
            elif data_type == DataType.UINT8:
                return cls(shape, data_type, array_type, bytearray(d))
        return cls(shape, data_type, array_type, d)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        if isinstance(self.data, (bitarray, bytearray)):
            data = [int(i) for i in self.data]  # Harmonise presentation of all non-list array providers
        elif self.array_type == ArrayType.FLAT and isinstance(self.data[0], str):
            data = f"'{''.join(self.data)}'"    # Join up list-of-char
        else:
            data = self.data

        if not self.shape:                      # Enclosed/scalar
            return f"<{data[0]}>"

        if self.rank == 1:                      # Vector
            return f"V({self.type}, {self.array_type}, {data})"      

        return f"A({self.shape}, {self.type}, {self.array_type}, {data})"

    def get(self, coords: Sequence[int]) -> Any:
        """
        Get item at coords.
        """
        idx = decode(self.shape, coords)
        return self.data[idx]        

    def to_list(self) -> list:
        """
        Return a list of unwrapped scalars.
        """
        if self.array_type == ArrayType.FLAT:
            if self.type in (DataType.UINT1, DataType.UINT8):
                return [int(i) for i in self.data]
            return list(self.data)

        for e in self.data:
            if all(isinstance(e, Array) for e in self.data):
                if any(not issimple(e) for e in self.data): # type: ignore
                    raise DomainError("DOMAIN ERROR")

        return [e.data[0] for e in self.data] # type: ignore

    def widen(self, kind: DataType) -> None:
        """
        Widen our data type (if needed and possible) to fit values of `kind`
        """
        if self.type == kind:
            return

        if self.type == DataType.UINT1:
            if kind == DataType.UINT8:
                self.data = bytearray(list(self.data))
                self.type = kind
            elif kind in {DataType.MIXED, DataType.INT, DataType.CMPLX}:
                self.data = list(self.data)
                self.type = kind

        elif self.type == DataType.UINT8 and kind != DataType.UINT1:
            self.data = list(self.data)
            self.type = kind

        elif self.type in {DataType.INT, DataType.FLOAT}:
            if kind in {DataType.MIXED, DataType.FLOAT, DataType.CMPLX}:
                self.type = kind

        elif self.type == DataType.CMPLX and kind == DataType.MIXED:
            self.type = kind


    def squeeze(self) -> None:
        """
        Squeeze self into the narrowest data type its values will allow. Try to un-mix mixed. Try to flatten nested.
        """
        # If each element is rank 0 and simple, we can flatten.
        if self.array_type == ArrayType.NESTED:
            if all(map(issimple, self.data)):
                self.data = [elem.data[0] for elem in self.data] # type: ignore
                self.array_type = ArrayType.FLAT

        # Can we un-mix?
        if self.array_type == ArrayType.FLAT:
            if self.type == DataType.MIXED:
                types = [type(e) for e in self.data]
                if all(map(lambda i:i==complex, types)):
                    self.type = DataType.CMPLX
                elif all(map(lambda i:i == int, types)):
                    self.type = DataType.INT
                elif all(map(lambda i:i == float, types)):
                    self.type = DataType.FLOAT
                elif all(map(lambda i:i == str, types)):
                    self.type = DataType.CHAR

            # Try to squeeze bit width
            if self.type in {DataType.INT, DataType.UINT8}:
                max_val = max(self.data)
                min_val = min(self.data)

                if min_val in [0, 1] and max_val in [0, 1]:
                    self.data = bitarray(self.data)
                    self.data_type = DataType.UINT1
                elif min_val in range(256) and max_val in range(256):
                    self.data = bytearray(self.data)
                    self.data_type = DataType.UINT8

    def mutate(self, idx: 'Array', vals: 'Array') -> None:
        """
        Only for a subset of indexing ranks for now
        """
        if idx.shape != vals.shape:
            raise RankError("RANK ERROR")

        if idx.type not in {DataType.UINT1, DataType.UINT8, DataType.INT}: 
            raise DomainError("DOMAIN ERROR")

        if idx.array_type != ArrayType.FLAT: # Dyalog allows nested, but we don't for now
            raise DomainError("DOMAIN ERROR")

        self.widen(vals.type) # ensure self's data type fits the new values

        valc = 0
        for c in idx.data:
            if c >= len(self.data) or c < 0:
                raise IndexError
            self.data[c] = vals.data[valc] # pick values in ravel order
            valc += 1

    def at(self, idx: 'Array') -> 'Array':
        """
        Only for a subset of indexing ranks for now. Should use `get()` above
        """
        if idx.type not in {DataType.UINT1, DataType.UINT8, DataType.INT}: 
            raise DomainError("DOMAIN ERROR")

        if idx.array_type != ArrayType.FLAT: # Dyalog allows nested, but we don't for now
            raise DomainError(f"DOMAIN ERROR: indexing expression must be flat")

        data = [self.data[c] for c in idx.data]
        return Array.from_sequence(idx.shape, self.type, self.array_type, data)

def A(shape: list[int], items: Sequence) -> Array:
    """
    Convenience 'constructor', which will ensure that simple scalars are
    converted to rank-0 arrays, and that non-enclosed higher-ranked elements
    are enclosed.

    It will not enclose further a single simple scalar, so:

    >>> A([], [S(5)])
    <5>

    It will, however, enclose highers:

    >>> A([3], [V([1, 1]), V([1, 2]), V([1, 3])])
    V(MIXED, NESTED, [<V(UINT1, FLAT, [1, 1])>, <V(UINT8, FLAT, [1, 2])>, <V(UINT8, FLAT, [1, 3])>])
    """
    if (
        shape == [] and 
        len(items) == 1 and 
        isinstance(items[0], Array) and 
        issimple(items[0])
    ):
        return items[0]

    data = [
        e if isinstance(e, Array) and e.shape == [] else S(e) 
        for e in items
    ]

    return Array.from_sequence(shape, DataType.MIXED, ArrayType.NESTED, data) # type: ignore

def Aflat(shape: list[int], data: Sequence) -> Array:
    """
    Given a sequence of scalars, make a flat array of the narrowest data type
    into which it will fit. If data contains other arrays, Aflat will throw
    a ValueError.
    """
    if any(isinstance(e, Array) for e in data):
        raise ValueError

    if any(isinstance(e, complex) for e in data):
        return Array.from_sequence(shape, DataType.CMPLX, ArrayType.FLAT, data)

    if all(isinstance(e, str) for e in data):
        return Array.from_sequence(shape, DataType.CHAR, ArrayType.FLAT, data)

    if all(isinstance(e, int) for e in data):
        max_val = max(data)
        min_val = min(data)

        if min_val in [0, 1] and max_val in [0, 1]:
            return Array.from_sequence(shape, DataType.UINT1, ArrayType.FLAT, data)

        if min_val in range(256) and max_val in range(256):
            return Array.from_sequence(shape, DataType.UINT8, ArrayType.FLAT, data)

        return Array.from_sequence(shape, DataType.INT, ArrayType.FLAT, data)

    if all(isinstance(e, float) for e in data):
        return Array.from_sequence(shape, DataType.FLOAT, ArrayType.FLAT, data)

    return Array.from_sequence(shape, DataType.MIXED, ArrayType.FLAT, data)

def S(data: Any) -> Array:
    """
    Create a scalar.
    """
    if isinstance(data, Sequence):
        raise RankError

    if isinstance(data, Array) and issimple(data):
        return data

    if isinstance(data, Array):
        return Array.from_sequence([], data.type, ArrayType.NESTED, [data])

    if isinstance(data, int):
        if data in range(2):
            return Array.from_sequence([], DataType.UINT1, ArrayType.FLAT, [data])
        if data in range(256):
            return Array.from_sequence([], DataType.UINT8, ArrayType.FLAT, [data])
        return Array.from_sequence([], DataType.INT, ArrayType.FLAT, [data])

    if isinstance(data, float):
        return Array.from_sequence([], DataType.FLOAT, ArrayType.FLAT, [data])

    if isinstance(data, str) and len(data) == 1:
        return Array.from_sequence([], DataType.CHAR, ArrayType.FLAT, [data])
    else:
        raise ValueError

def V(data: Sequence) -> Array:
    """
    Convenience vector constructor. Try to squeeze into smallest data width 
    if not nested.
    """
    flattened = []
    for e in data:
        if isinstance(e, Array):
            if issimple(e):
                flattened.append(e.data[0])
            else:
                return A([len(data)], data)
        else:
            flattened.append(e)
    return Aflat([len(flattened)], flattened)
    
def isscalar(a: Any) -> bool:
    if isinstance(a, Array):
        return not a.shape
    return True

def issimple(a: Any) -> bool:
    if not isinstance(a, Array):
        return True
    return isscalar(a) and not isinstance(a.data[0], Array)

def isnested(a: Any) -> bool:
    if not isinstance(a, Array):
        return False
    return a.array_type == ArrayType.NESTED

def enclose(arr: Array) -> Array:
    return Array.from_sequence([], arr.type, ArrayType.NESTED, [arr])

def disclose(arr: Any) -> Array:
    if isinstance(arr, Array):
        if issimple(arr):
            return arr
        return arr.data[0] # type: ignore
    return S(arr) # make an array-y scalar from an actual scalar
        
def kcells(arr: Array, k: int) -> Iterator[Array]:
    if k == 0:
        yield arr

    if k > arr.rank:
        raise RankError("RANK ERROR")

    if arr.rank == k:
        yield enclose(arr)

    # Shape and bound of result
    rsh = arr.shape[:arr.rank-k]
    rbnd = math.prod(rsh)

    # Shape and bound of each cell
    csh = arr.shape[arr.rank-k:]
    cbnd = math.prod(csh)

    for cell in range(rbnd):
        data = arr.data[cell*cbnd:(cell+1)*cbnd]
        yield Array.from_sequence(csh, arr.type, arr.array_type, data) # type: ignore

def rows_flat_matrix(arr: Array) -> Iterator:
    """
    Iterate over the rows in a flat array of rank 2
    """
    if arr.rank != 2:
        raise RankError("RANK ERROR")
    if arr.array_type != ArrayType.FLAT:
        raise DomainError("DOMAIN ERROR")

    for row in range(arr.shape[0]):
        yield arr.data[row*arr.shape[1]:(row+1)*arr.shape[1]]

def encode(shape: Sequence[int], idx: int) -> Sequence[int]:
    """
    encode returns the coordinate vector into shape corresponding
    to the linear index idx into its ravel vector

    >>> encode([24, 60, 60], 10_000)
    [2, 46, 40]
    """
    encoded: list[int] = []
    for axis in shape[::-1]:
        idx, loc = divmod(idx, axis)
        encoded.append(loc)
    return encoded[::-1]

def decode(shape: Sequence[int], coords: Sequence[int]) -> int:
    """
    Convert coords from the basis given by shape to ravel location.

    >>> decode([2, 2], [0, 0])
    0

    >>> decode([2, 2], [1, 1])
    3

    # Mixed radix: how many seconds are 2 hours, 46 minutes and 40 seconds?
    >>> decode([24, 60, 60], [2, 46, 40]) 
    10000

    # What is binary 1101 in decimal?
    >>> decode([2, 2, 2, 2], [1, 1, 0, 1]) 
    13
    """
    pos = 0
    rank = len(shape)
    for axis in range(rank):
        if axis >= len(coords):
            return pos
        pos += coords[axis]
        if axis != rank - 1:
            pos *= shape[axis+1]
    return pos

def coords(shape: Sequence[int], IO: int = 0) -> Iterator[Sequence[int]]:
    """
    Generator. Step through the space defined by the shape, generating 
    each coordinate vector in turn.

    >>> list(coords([2, 3]))
    [[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2]]
    """
    rank = len(shape)
    bound = math.prod(shape)
    offsets = [IO for _ in range(rank)]
    coords = [0 for _ in range(rank)]
    yield coords[:]
    for idx in range(1, bound):
        axis = rank - 1
        coords[axis] += 1
        while axis>0 and coords[axis] == shape[axis] + offsets[axis]:
            coords[axis] = 0
            coords[axis-1] += 1
            axis -= 1
        yield coords[:]

def index_cell(arr: Array, ind: Sequence[int]) -> Array:
    """
    Collapse the first rank-len(index) axes at a single index along each of them.

    Beginnings of squad.

    >>> import math
    >>> shape = [3, 4, 5]
    >>> a = Aflat(shape, range(math.prod(shape)))
    >>> index_cell(a, [1])
    A([4, 5], UINT8, FLAT, [20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39])
    """
    if not ind or len(ind) > arr.rank:
        raise RankError("RANK ERROR")

    # Shape and bound of cell
    rank = arr.rank - len(ind)
    csh = arr.shape[len(ind):]
    cbnd = math.prod(csh)

    # Fix the indices given by ind in our shape
    rr = [0 for _ in range(len(arr.shape))]
    for i, e in enumerate(ind):
        rr[i] = e

    # Find the start of this cell
    pos = decode(arr.shape, rr)
    data = arr.data[pos:pos+cbnd]

    return Array.from_sequence(csh, arr.type, arr.array_type, data) # type: ignore
    
def match(alpha: Array, omega: Array) -> bool:
    if not isnested(alpha) and not isnested(omega) and alpha.shape == omega.shape:
        if alpha.type == omega.type:
            return alpha.data == omega.data
        return all(alpha.data[i] == omega.data[i] for i in range(alpha.bound))

    if alpha.shape != omega.shape:
        return False

    if not alpha.shape: # Simple scalar, or enclosed something
        if isinstance(alpha.data[0], Array):
            return match(alpha.data[0], omega.data[0]) # type: ignore
        else:
            return alpha.data[0] == omega.data[0]

    for i in range(alpha.bound):
        a = disclose(alpha.data[i])
        b = disclose(omega.data[i])
        if not match(a, b):
            return False
    return True

if __name__ == "__main__":
    # To run the doctests (verbosely), do 
    #
    # python arr.py -v
    #
    # See: https://docs.python.org/3/library/doctest.html
    import doctest
    doctest.testmod()