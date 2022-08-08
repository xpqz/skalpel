from itertools import chain
from math import prod
from typing import Any, Iterator, Optional, TypeAlias, Union

SimpleScalar: TypeAlias = int|float

class LengthError(Exception):
    pass

class LimitError(Exception):
    pass

class NYIError(Exception):
    pass

class RankError(Exception):
    pass

class ValueError(Exception):
    pass

class Array:
    def __init__(self, shape: list[int], data: Union[list[SimpleScalar],'Array', SimpleScalar]) -> None:
        if len(shape) > 5:
            raise LimitError("Rank of resultant array would exceed maximum permitted")
        self.shape = shape
        self.bound = prod(self.shape)
        self.rank = len(self.shape)
        self.data = data

    def __repr__(self):
        return str(self)
        
    def __str__(self):
        if self.rank == 0:
            return str(self.data[0])
        if self.rank == 1:
            return f"Vector({self.data})"
        return f"Array({self.shape}, {self.data})"

    def get(self, coords: list[int]) -> Any:
        """
        Get item at coords.

        >>> Array([2, 2], [1, 9, 4, 8]).get([1, 0])
        4
        """
        if self.rank == 0:
            raise RankError

        idx = decode(self.shape, coords)
        return self.data[idx] # type: ignore

def Vector(data: list[SimpleScalar]) -> Array:
    return Array([len(data)], data)

def Scalar(data: SimpleScalar) -> Array:
    return Array([], [data])

def enclose(a: Array) -> Array:
    if not a.shape and not isinstance(a.data, Array): # Simple scalar
        return a
    return Array([], a)

def disclose(a: Array) -> Array:
    """
    Should we map disclose to take first here?
    """
    if isinstance(a.data, Array):
        return a.data
    return a

def encode(shape: list[int], idx: int) -> list[int]:
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

def decode(shape: list[int], coords: list[int]) -> int:
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
        pos += coords[axis]
        if axis != rank - 1:
            pos *= shape[axis+1]
    return pos

def coords(shape: list[int], IO: int = 0) -> Iterator[list[int]]:
    """
    Generator. Step through the space defined by the shape, generating 
    each coordinate vector in turn.

    >>> list(coords([2, 3]))
    [[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2]]
    """
    rank = len(shape)
    bound = prod(shape)
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

def select(arr: Array, cells: Optional[list[int]] = None) -> Array:
    """
    (towards) sane indexing: select ← ⌷⍨∘⊃⍨⍤0 99

    Return major cells. No enclosed left yet, or reach index.

    If the last argument isn't given we return all major cells.

    >> select(Array([2, 2], [1, 2, 3, 4]))
    Array([2, 2], [1, 2, 3, 4])

    >> select(Array([2, 2], [1, 2, 3, 4]), [0])
    Array([2], [1, 2])

    >>> select(Vector([1, 2, 3]), [1])
    2

    """
    if not arr.shape and not isinstance(arr.data, Array): # Simple scalar
        return arr

    first = arr.shape[0]
    size = arr.bound//first
    if cells is None:
        cells = range(first)

    selection = list(chain(*(arr.data[cell*size: (cell+1)*size] for cell in cells)))

    # The new shape is that of our source, but with the leading axis being the number of
    # requested cells, except if this is one, in which case we drop the leading
    # axis completely.
    newshape = arr.shape[:]
    newshape[0] = len(cells)
    if newshape[0] == 1:
        newshape = newshape[1:]

    # if not newshape:
    #     return Scalar(selection[0])
    
    return Array(newshape, selection) # type: ignore

if __name__ == "__main__":
    # To run the doctests (verbosely), do 
    #
    # python arr.py -v
    #
    # See: https://docs.python.org/3/library/doctest.html
    import doctest
    doctest.testmod()