from math import prod
from apl.funs import transpose, squad, rho, apply_scalar
from apl.arr import Array, Vector, Scalar

class TestTranspose:
    def test_monadic_transpose_empty(self):
        a = Array([0], [])
        assert a.data == []
        b = transpose(omega=a)
        assert b.data == []

    def test_monadic_transpose_rank1(self):
        a = Array([4], [1, 2, 3, 4])
        assert a.data == [1, 2, 3, 4]
        b = transpose(omega=a)
        assert b.data == [1, 2, 3, 4]

    def test_monadic_transpose_rank2(self):
        shape = [2, 2]
        total = prod(shape)
        a = Array(shape, list(range(total)))
        assert a.data == [0, 1, 2, 3]
        b = transpose(omega=a)
        assert b.data == [0, 2, 1, 3]

    def test_dyadic_transpose(self):
        shape = [2, 3, 4]
        total = prod(shape)
        a = Array(shape, list(range(total)))
        b = transpose(alpha=Vector([2, 0, 1]), omega=a)
        assert b.data == [0,12,1,13,2,14,3,15,4,16,5,17,6,18,7,19,8,20,9,21,10,22,11,23]
        
    def test_dyadic_transpose_diagonal(self): # unlikely to work
        shape = [3, 3]
        total = prod(shape)
        a = Array(shape, list(range(total)))
        b = transpose(alpha=Vector([0, 0]), omega=a)
        assert b.data == [0, 4, 8]

class TestSquad:
    def test_squad_single_cell(self):
        a = squad(alpha=Vector([1]), omega=Array([2, 2], [1, 2, 3, 4]))
        assert a.rank == 1
        assert a.data == [3, 4]

    def test_squad_recursive(self):
        cells = Vector([1, 1])
        data = Array([2, 2], [1, 2, 3, 4])
        a = squad(alpha=cells, omega=data)
        assert a.rank == 0
        assert a.data == [4]
        
    def test_squad_higher_rank(self):
        cells = Vector([1, 2])
        data = Array([2, 3, 4], list(range(2*3*4)))
        a = squad(alpha=cells, omega=data)
        assert a.rank == 1
        assert a.data == [20, 21, 22, 23]
    
class TestRho:
    def test_reshape_matrix(self):
        reshaped = rho(alpha=Vector([5, 5]), omega=Array([3, 3], [3, 2, 0, 8, 6, 5, 4, 7, 1])).data
        assert reshaped == [3, 2, 0, 8, 6, 5, 4, 7, 1, 3, 2, 0, 8, 6, 5, 4, 7, 1, 3, 2, 0, 8, 6, 5, 4]

    def test_reshape_scalar(self):
        reshaped = rho(alpha=Vector([2, 2]), omega=Scalar(5)).data
        assert reshaped == [5, 5, 5, 5]

    def test_monadic(self):
        shape = rho(omega=Array([2, 2], [1, 2, 3, 4])).data
        assert shape == [2, 2]

class TestApplyScalar:
    def test_sum_scalars(self):
        assert apply_scalar(Scalar(5), Scalar(7), '+').data == [12]

    def test_sum_arrays(self):
        assert [4, 4, 4, 4] == apply_scalar(Array([2, 2], [1, 2, 3, 4]), Array([2, 2], [3, 2, 1, 0]), '+').data