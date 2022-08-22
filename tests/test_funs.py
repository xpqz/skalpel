from math import prod

from apl.arr import Array, A, V, S, match
from apl.voc import pervade, transpose, reduce_first


class TestPervade:
    def test_mat_plus_scalar(self):
        plus = pervade(lambda x, y:x+y)
        a = A([3, 2], [54, 7, 88, 4, 73, 3, 1])
        b = S(12)
        result = plus(a, b)
        assert match(result, A([3, 2], [66, 19, 100, 16, 85, 15]))

    def test_scalar_plus_mat(self):
        plus = pervade(lambda x, y:x+y)
        a = A([3, 2], [54, 7, 88, 4, 73, 3, 1])
        b = S(12)
        result = plus(b, a)
        assert match(result, A([3, 2], [66, 19, 100, 16, 85, 15]))

    def test_vec_plus_vec(self):
        plus = pervade(lambda x, y:x+y)
        a = V([54, 7, 88, 4, 73, 3, 1])
        b = V([12, 8, 11, 7, 21, 7, 9])
        result = plus(b, a)
        assert match(result, V([66, 15, 99, 11, 94, 10, 10]))

    def test_non_simple(self):
        plus = pervade(lambda x, y:x+y)
        a = A([3, 2], [54, 7, A([2, 2], [1]), 4, 73, 3])
        b = S(12)
        result = plus(b, a)
        expected = A([3, 2], [66, 19, A([2, 2], [13]), 16, 85, 15])
        assert match(result, expected)


class TestTranspose:
    def test_monadic_transpose_empty(self):
        a = A([0], [])
        assert a.data == []
        b = transpose([], a)
        assert b.data == []

    def test_monadic_transpose_rank1(self):
        a = A([4], [1, 2, 3, 4])
        b = transpose([], a)
        assert match(b, V([1, 2, 3, 4]))

    def test_monadic_transpose_rank2(self):
        shape = [2, 2]
        total = prod(shape)
        a = A(shape, list(range(total)))
        b = transpose([], a)
        assert match(b, A([2, 2], [0, 2, 1, 3]))

    def test_dyadic_transpose(self):
        shape = [2, 3, 4]
        total = prod(shape)
        a = A(shape, list(range(total)))
        b = transpose([2, 0, 1], a)
        assert match(b, A([3, 4, 2], [0,12,1,13,2,14,3,15,4,16,5,17,6,18,7,19,8,20,9,21,10,22,11,23]))
        
    def test_dyadic_transpose_diagonal(self):
        shape = [3, 3]
        total = prod(shape)
        a = A(shape, list(range(total)))
        b = transpose([0, 0], a)
        assert match(b, V([0, 4, 8]))

class TestReduce:
    def test_reduce_first(self):
        r = reduce_first('+', None, None, A([2, 2], [1, 2, 3, 4]))
        assert match(r, V([4, 6]))

# class TestSquad:
#     def test_squad_single_cell(self):
#         a = squad(alpha=Vector([1]), omega=Array([2, 2], [1, 2, 3, 4]))
#         assert a.rank == 1
#         assert a.data == [3, 4]

#     def test_squad_recursive(self):
#         cells = Vector([1, 1])
#         data = Array([2, 2], [1, 2, 3, 4])
#         a = squad(alpha=cells, omega=data)
#         assert a.rank == 0
#         assert a.data == [4]
        
#     def test_squad_higher_rank(self):
#         cells = Vector([1, 2])
#         data = Array([2, 3, 4], list(range(2*3*4)))
#         a = squad(alpha=cells, omega=data)
#         assert a.rank == 1
#         assert a.data == [20, 21, 22, 23]
