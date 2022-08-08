from math import prod
from apl.arr import Array, Vector, disclose, enclose, select

class TestSelect:
    def test_select_row(self):
        data = Array([2, 2], [1, 2, 3, 4])
        s = select(data, [1])
        assert s.rank == 1
        assert s.data == [3, 4]

    def test_select_scalar(self):
        data = Vector([1, 2, 3, 4])
        s = select(data, [1])
        assert s.rank == 0
        assert s.data == [2]

    def test_select_multiple_scalars(self):
        data = Vector([1, 2, 3, 4])
        s = select(data, [1, 2])
        assert s.rank == 1
        assert s.data == [2, 3]

    def test_select_all(self):
        data = Array([2, 2], [1, 2, 3, 4])
        s = select(data)
        assert s.rank == 2
        assert s.data == [1, 2, 3, 4]

    def test_select_layer(self):
        shape = [2, 3, 4]
        bound = prod(shape)
        data = Array(shape, list(range(bound)))
        s = select(data, [1])
        assert s.shape == [3, 4]
        assert s.data == [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]


class TestEnclose:
    def test_enclose(self):
        a = Array([2, 2], [1, 2, 3, 4])
        b = enclose(a)
        assert not b.shape
        assert b.rank == 0

    def test_enclose_scalar(self):
        a = Array([], 5)
        assert a == enclose(a)

class TestDisclose:
    def test_disclose(self):
        a = Array([2, 2], [1, 2, 3, 4])
        assert disclose(enclose(a)) == a

    


