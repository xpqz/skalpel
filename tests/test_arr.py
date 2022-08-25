from math import prod
import pytest
import apl.arr as arr

class TestIndexCell:
    def test_select_row(self):
        data = arr.A([2, 2], [1, 2, 3, 4])
        s = arr.index_cell(data, [1])
        assert arr.match(s, arr.V([3, 4]))

    def test_select_scalar(self):
        data = arr.V([1, 2, 3, 4])
        s = arr.index_cell(data, [1])
        assert arr.match(s, arr.S(2))

    def test_select_layer(self):
        shape = [2, 3, 4]
        bound = prod(shape)
        data = arr.A(shape, list(range(bound)))
        s = arr.index_cell(data, [1])
        assert arr.match(s, arr.A([3, 4], [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]))

class TestKCells:
    def test_3_cell(self):
        a = list(arr.kcells(arr.A([3, 4, 5, 6, 7], [1]), 3))
        assert len(a) == 12
        assert a[0].shape == [5, 6, 7]

class TestEncodeDecode:
    def test_simple_encode(self):
        assert [2, 46, 40] == arr.encode([24, 60, 60], 10_000)

    @pytest.mark.parametrize("test_input,expected", [
        (([2, 2], [0, 0]),                 0),
        (([2, 2], [1, 1]),                 3),
        (([24, 60, 60], [2, 46, 40]),  10000),
        (([2, 2, 2, 2], [1, 1, 0, 1]),    13),
    ])
    def test_decode(self, test_input, expected):
        assert expected == arr.decode(*test_input)

class TestCoords:
    def test_coords_simple(self):
        assert list(arr.coords([2, 3])) == [[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2]]
    
class TestIsNested:
    @pytest.mark.parametrize("test_input,expected", [
        (arr.S(1),                         False), # 1
        (arr.V([1, 2, 3]),                 False), # 1 2 3
        (arr.Aflat([2, 2], [1, 2, 3, 4]),  False), # 2 2⍴1 2 3 4
        (arr.V([1, arr.V([1, 2])]),        True),  # 1 (1 2)
        (arr.S(arr.V([1, arr.V([1, 2])])), True),  # ⊂1 (1 2)
    ])
    def test_is_nested(self, test_input, expected):
        assert expected == arr.isnested(test_input)
        


    


