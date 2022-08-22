import itertools
import math
from typing import Any, Iterator, Sequence

from apl.errors import DomainError, LimitError, RankError

class Array:
    def __init__(self, shape: list[int], data: Sequence) -> None:
        if len(shape) > 16:
            raise LimitError("LIMIT ERROR: Rank of resultant array would exceed maximum permitted")
        self.shape = shape
        self.bound = math.prod(self.shape)
        self.rank = len(self.shape)
        self.data = list(itertools.islice(itertools.cycle(data), self.bound))

    def to_list(self) -> list:
        """
        Return a list of unwrapped scalars.
        """
        for e in self.data:
            if not issimple(e):
                raise DomainError("DOMAIN ERROR")

        return [e.data[0] for e in self.data]
        
    def __repr__(self):
        return str(self)
        
    def __str__(self):
        if self.rank == 0:
            return f"Scalar({self.data[0]})"
        if self.rank == 1:
            return f"Vector({self.data})"
        return f"Array({self.shape}, {self.data})"

    def get(self, coords: Sequence[int]) -> Any:
        """
        Get item at coords.

        >>> A([2, 2], [1, 9, 4, 8]).get([1, 0]).data[0]
        4
        """
        if self.rank == 0:
            raise RankError('RANK ERROR: cannot index scalars')

        idx = decode(self.shape, coords)
        return self.data[idx]

    def kcells(self, k: int) -> 'Array':
        """
        Dyalog's docs on k-cells:
    
        K-Cells

        A rank-k cell or k-cell of an array are terms used to describe a sub-array 
        on the trailling k axes of the array. Negative k is interpreted as r+k where r is 
        the rank of the array, and is used to describe a sub-array on the leading |k 
        axes of an array.

        If X is a 3-dimensional array of shape 2 3 4, the 1-cells are its 6 rows each 
        of 4 elements; and its 2-cells are its 2 matrices each of shape 3 4. Its 3-cells 
        is the array in its entirety. Its 0-cells are its individual elements.
        
        See: https://aplwiki.com/wiki/Cell

        Squad indexes k-cells:

            A ← 3 4 5 6 7⍴1   ⍝ A is a rank 5 array
            ⍴1 2⌷A            ⍝ Get the 3-cell at index 1 2
            ┌→────┐
            │5 6 7│
            └~────┘

        The kcells method essentially partitions the shape. If we ask for the 3-cells of
        the above array A with a shape [3, 4, 5, 6, 7], we get back an array with the
        shape [3, 4], where each element (the cells) are arrays of shape [5, 6, 7]:

        >>> Array([3, 4, 5, 6, 7], [1]).kcells(3).shape
        [3, 4]

        >>> Array([3, 4, 5, 6, 7], [1]).kcells(3).data[0].shape
        [5, 6, 7]

        Negative cells TODO
        """
        if k == 0:
            return self

        if k > self.rank:
            raise RankError("RANK ERROR")

        if self.rank == k:
            return S(self)

        # Shape and bound of result
        rsh = self.shape[:self.rank-k]
        rbnd = math.prod(rsh)

        # Shape and bound of each cell
        csh = self.shape[self.rank-k:]
        cbnd = math.prod(csh)

        return Array(rsh, [
            Array(csh, self.data[cell*cbnd:(cell+1)*cbnd])
            for cell in range(rbnd)
        ])

    def index_cell(self, ind: Sequence[int]) -> 'Array':
        """
        Collapse the first rank-len(index) axes at a single index along each of them.

        Beginnings of squad.

        >>> import math
        >>> shape = [3, 4, 5]
        >>> a = A(shape, list(range(math.prod(shape))))
        >>> match(a.index_cell([1]), A([4, 5], [20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39])) # 1⌷a
        True
        """
        if not ind or len(ind) > self.rank:
            raise RankError("RANK ERROR")

        # Shape and bound of cell
        rank = self.rank - len(ind)
        # csh = self.shape[-rank:]
        # csh = self.shape[:-len(ind)]
        csh = self.shape[len(ind):]
        cbnd = math.prod(csh)

        # Fix the indices given by ind in our shape
        rr = [0 for _ in range(len(self.shape))]
        for i, e in enumerate(ind):
            rr[i] = e

        # Find the start of this cell
        pos = decode(self.shape, rr)
        data = self.data[pos:pos+cbnd]

        return A(csh, data) # NOTE: must use convenience constructor

def isscalar(a: Array) -> bool:
    return not a.shape

def issimple(a: Array) -> bool:
    return isscalar(a) and not isinstance(a.data[0], Array)

def disclose(a: Array) -> Array:
    if issimple(a):
        return a
    return a.data[0]

def isnested(a: Array) -> bool:
    if issimple(a): # I am but a simple, humble scalar
        return False

    if isscalar(a): # I am enclosed. Check payload
        return isnested(disclose(a))

    return not all(issimple(disclose(cell)) for cell in a.data)

def A(shape: list[int], items: Sequence) -> Array:
    """
    Convenience 'constructor', which will ensure that simple scalars are
    converted to rank-0 arrays, and that non-enclosed higher-ranked elements
    are enclosed.

    It will not enclose further a single simple scalar, so:

    >>> A([], [S(5)])
    Scalar(5)

    It will, however, enclose highers:

    >>> A([3], [V([1, 1]), V([1, 2]), V([1, 3])])
    Vector([Scalar(Vector([Scalar(1), Scalar(1)])), Scalar(Vector([Scalar(1), Scalar(2)])), Scalar(Vector([Scalar(1), Scalar(3)]))])
    """
    if shape == [] and len(items) == 1 and isinstance(items[0], Array) and issimple(items[0]):
        return items[0]

    return Array(shape, [
        e if isinstance(e, Array) and e.shape == [] else S(e) 
        for e in items
    ])

def V(data: Sequence) -> Array:
    return A([len(data)], data)

def S(item: Any) -> Array:
    """
    Convenience constructor for making a rank-0 array from a simple scalar.
    If the item is already a rank-0 array of a simple scalar, return it as-is.

    >>> S(5).data[0] == 5
    True

    >>> match(S(S(S(5))), S(5))
    True
    """
    if not isinstance(item, (int, float, complex, Array)): # Too many self-inflicted problems
        raise TypeError

    if isinstance(item, Array) and issimple(item):
        return item

    return Array([], [item])  # NOTE: can't call A() for infinite recursion reasons :)

def match(alpha: Array, omega: Array) -> bool:
    if alpha.shape != omega.shape:
        return False
    if not alpha.shape: # Simple scalar, or enclosed something
        if isinstance(alpha.data[0], Array):
            return match(alpha.data[0], omega.data[0])
        else:
            return alpha.data[0] == omega.data[0]
    for i in range(alpha.bound):
        if not match(alpha.data[i], omega.data[i]):
            return False
    return True

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

if __name__ == "__main__":
    # To run the doctests (verbosely), do 
    #
    # python arr.py -v
    #
    # See: https://docs.python.org/3/library/doctest.html
    import doctest
    doctest.testmod()