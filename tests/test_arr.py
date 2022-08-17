from math import prod
import apl.arr as arr

class TestIndexCell:
    def test_select_row(self):
        data = arr.A([2, 2], [1, 2, 3, 4])
        s = data.index_cell([1])
        assert arr.match(s, arr.V([3, 4]))

    def test_select_scalar(self):
        data = arr.V([1, 2, 3, 4])
        s = data.index_cell([1])
        assert arr.match(s, arr.S(2))

    def test_select_layer(self):
        shape = [2, 3, 4]
        bound = prod(shape)
        data = arr.A(shape, list(range(bound)))
        s = data.index_cell([1])
        assert arr.match(s, arr.A([3, 4], [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]))

class TestKCells:
    def test_3_cell(self):
        a = arr.A([3, 4, 5, 6, 7], [1]).kcells(3)
        assert a.shape == [3, 4]
        assert a.data[0].shape == [5, 6, 7]

class TestEncodeDecode:
    def test_simple_encode(self):
        assert [2, 46, 40] == arr.encode([24, 60, 60], 10_000)

    def test_decode(self):
        assert 0 == arr.decode([2, 2], [0, 0])
        assert 3 == arr.decode([2, 2], [1, 1])
        assert 10000 == arr.decode([24, 60, 60], [2, 46, 40]) 
        assert 13 == arr.decode([2, 2, 2, 2], [1, 1, 0, 1]) 

class TestCoords:
    def test_coords_simple(self):
        assert list(arr.coords([2, 3])) == [[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2]]
        


    


