from math import prod

import apl.arr as arr
import apl.skalpel as f

class TestPervade:
    def test_mat_plus_scalar(self):
        plus = f.pervade(lambda x, y:x+y)
        a = arr.Aflat([3, 2], [54, 7, 88, 4, 73, 3, 1])
        b = arr.S(12)
        result = plus(a, b)
        assert arr.match(result, arr.Aflat([3, 2], [66, 19, 100, 16, 85, 15]))

    def test_scalar_plus_mat(self):
        plus = f.pervade(lambda x, y:x+y)
        a = arr.Aflat([3, 2], [54, 7, 88, 4, 73, 3, 1])
        b = arr.S(12)
        result = plus(b, a)
        assert arr.match(result, arr.Aflat([3, 2], [66, 19, 100, 16, 85, 15]))

    def test_vec_plus_vec(self):
        plus = f.pervade(lambda x, y:x+y)
        a = arr.V([54, 7, 88, 4, 73, 3, 1])
        b = arr.V([12, 8, 11, 7, 21, 7, 9])
        result = plus(b, a)
        assert arr.match(result, arr.V([66, 15, 99, 11, 94, 10, 10]))

    def test_non_simple(self):
        plus = f.pervade(lambda x, y:x+y)
        a = arr.A([3, 2], [54, 7, arr.Aflat([2, 2], [1]), 4, 73, 3])
        b = arr.S(12)
        result = plus(b, a)
        expected = arr.A([3, 2], [66, 19, arr.Aflat([2, 2], [13]), 16, 85, 15])
        assert arr.match(result, expected)


class TestTranspose:
    def test_monadic_transpose_empty(self):
        a = arr.A([0], [])
        assert a.data == []
        b = f.transpose([], a)
        assert b.data == []

    def test_monadic_transpose_rank1(self):
        a = arr.A([4], [1, 2, 3, 4])
        b = f.transpose([], a)
        assert arr.match(b, arr.V([1, 2, 3, 4]))

    def test_monadic_transpose_rank2(self):
        shape = [2, 2]
        total = prod(shape)
        a = arr.A(shape, list(range(total)))
        b = f.transpose([], a)
        assert arr.match(b, arr.A([2, 2], [0, 2, 1, 3]))

    def test_dyadic_transpose(self):
        shape = [2, 3, 4]
        total = prod(shape)
        a = arr.A(shape, list(range(total)))
        b = f.transpose([2, 0, 1], a)
        assert arr.match(b, arr.A([3, 4, 2], [0,12,1,13,2,14,3,15,4,16,5,17,6,18,7,19,8,20,9,21,10,22,11,23]))
        
    def test_dyadic_transpose_diagonal(self):
        shape = [3, 3]
        total = prod(shape)
        a = arr.A(shape, list(range(total)))
        b = f.transpose([0, 0], a)
        assert arr.match(b, arr.V([0, 4, 8]))

class TestReduce:
    def test_reduce_first(self):
        r = f.reduce_first('+', None, None, arr.A([2, 2], [1, 2, 3, 4]), None, None)
        assert arr.match(r, arr.V([4, 6]))

class TestBitwise:
    def test_or(self):
        bwo = f.pervade(lambda x, y:x|y)
        a = arr.V([1, 0, 1, 0])
        b = arr.V([0, 1, 0, 1])
        c = bwo(a, b)
        assert arr.match(c, arr.V([1, 1, 1, 1]))

class TestReplicate:
    def test_compress_flat(self):
        a = arr.V([1, 0, 1, 0, 1])
        b = arr.V([301, 32, 76, 87, 1])
        c = f.replicate(a, b)
        assert arr.match(c, arr.V([301, 76, 1]))

    def test_compress_nested(self):
        a = arr.V([1, 0, 1, 0, 1])
        b = arr.V([301, 32, arr.V([76, 43, 1]), 87, 1])
        c = f.replicate(a, b)
        assert arr.match(c, arr.V([301, arr.V([76, 43, 1]), 1]))

    def test_replicate_flat(self):
        a = arr.V([2, 0, 3, 0, 1])
        b = arr.V([301, 32, 76, 87, 1])
        c = f.replicate(a, b)
        assert arr.match(c, arr.V([301, 301, 76, 76, 76, 1]))

    def test_replicate_nested(self):
        a = arr.V([2, 0, 3, 0, 1])
        b = arr.V([arr.V([76, 43, 1]), 32, 76, 87, 1])
        c = f.replicate(a, b)
        assert arr.match(c, arr.V([arr.V([76, 43, 1]), arr.V([76, 43, 1]), 76, 76, 76, 1]))

class TestWithout:
    def test_without(self):
        a = arr.V([1, 2, 3, 4, 5])
        b = f.without(a, arr.V([2, 4]))
        assert arr.match(b, arr.V([1, 3, 5]))

class TestWhere:
    def test_where_vector(self):
        a = arr.V([1, 0, 0, 1, 0, 1, 0])
        b = f.where(a)
        assert arr.match(b, arr.V([0, 3, 5]))

    def test_where_arr(self):
        a = arr.Aflat([2, 2], [1, 1, 1, 0])
        b = f.where(a)
        assert 'V(MIXED, NESTED, [<V(INT, FLAT, [0, 0])>, <V(INT, FLAT, [0, 1])>, <V(INT, FLAT, [1, 0])>])' == str(b)

class TestEach:
    def test_each(self):
        a = arr.V([arr.V([1, 0]), arr.V([1, 0, 1]), arr.V([1, 0, 1, 0])])
        b = f.each('≢', None, None, a, None, None)
        assert 'V(MIXED, NESTED, [<2>, <3>, <4>])' == str(b)
