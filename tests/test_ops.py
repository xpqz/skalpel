from apl.arr import Array, Vector
from apl.ops import reduce_first, reduce

class TestReduce:
    def test_reduce_first(self):
        a = reduce_first('+', None, None, Array([2, 2], [1, 2, 3, 4]))
        assert a.data == [4, 6]

    def test_reduce_last(self):
        a = reduce('+', None, None, Array([2, 2], [1, 2, 3, 4]))
        assert a.data == [3, 7]

    def test_reduce_vector(self):
        a = reduce('+', None, None, Vector([1, 2, 3, 4]))
        assert a.data == [10]

    def test_reduce_vector_non_assoc(self):
        a = reduce('-', None, None, Vector([1, 2, 3, 4]))
        assert a.data == [-2]
