from apl.arr import Array, Vector
from apl.ops import reduce

class TestReduce:
    def test_reduce_first(self):
        a = reduce(Array([2, 2], [1, 2, 3, 4]), 0, lambda x, y:x+y).data
        assert a == [4, 6]

    def test_reduce_last(self):
        a = reduce(Array([2, 2], [1, 2, 3, 4]), 1, lambda x, y:x+y).data
        assert a == [3, 7]

    def test_reduce_vector(self):
        a = reduce(Vector([1, 2, 3, 4]), 0, lambda x, y:x+y).data
        assert a == [10]

    def test_reduce_vector_non_assoc(self):
        a = reduce(Vector([1, 2, 3, 4]), 0, lambda x, y:x-y).data
        assert a == [-2]

    def test_reduce_vector_builtin(self):
        a = reduce(Vector([1, 2, 3, 4]), 0, '+').data
        assert a == [10]

