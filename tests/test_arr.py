from math import prod

import pytest

import apl.arr as arr
from apl.errors import RankError

class TestIndexing:
    def test_simple(self):
        a = arr.V([1, 2, 3, 4])
        assert arr.match(a.at(arr.S(1)), arr.S(2))

    def test_simple2(self):
        a = arr.V([1, 2, 3, 4])
        idx = arr.V([1, 0])
        elems = a.at(idx)
        assert arr.match(elems, arr.V([2, 1]))

    def test_simple3(self):
        """
        a←2 2⍴1 2 3 4
        a[(1 0)(0 1)]
        ┌→──┐
        │3 2│
        └~──┘
        """
        a = arr.Array([2, 2], [
            1, 2,
            3, 4
        ])
        idx = arr.V([arr.V([1, 0]), arr.V([0, 1])])
        elems = a.at(idx)
        assert arr.match(elems, arr.V([3, 2]))

    def test_simple4(self):
        a = arr.V([1, 2, 3, 4])
        idx = arr.Array([2, 2], [1, 0, 1, 0])
        elems = a.at(idx)
        assert arr.match(elems, arr.Array([2, 2], [
            2, 1,
            2, 1
        ]))

    def test_hi_rank1(self):
        """
        a←3 3⍴⍳9
        a[;2]
        ┌→────┐
        │2 5 8│
        └~────┘
        """
        a = arr.Array([3, 3], list(range(9)))
        elems = a.rect([[],[2]])
        assert arr.match(elems, arr.V([2, 5, 8]))

    def test_hi_rank2(self):
        """
        a←3 3⍴⍳9
        a[1 2;]
        ┌→────┐
        ↓3 4 5│
        │6 7 8│
        └~────┘
        """
        a = arr.Array([3, 3], list(range(9)))
        elems = a.rect([[1, 2], []])
        assert arr.match(elems, arr.Array([2, 3], [
            3, 4, 5, 
            6, 7, 8
        ]))

class TestPick:
    def test_pick_zilde(self):
        """
        ⍬⊃3
        
        3
        """
        a = arr.S(3)
        picked = a.pick(arr.Array([0], []))
        expected = arr.S(3)

        assert arr.match(picked, expected)

    def test_pick_simple(self):
        """
        2⊃'pick'
        
        'c'
        """
        v = arr.V('pick')
        picked = v.pick(arr.S(2))
        expected = arr.S('c')

        assert arr.match(picked, expected)

    def test_pick_nested(self):
        """
        1⊃'foo' 'bar'

        'bar'
        """
        a = arr.V([arr.V('foo'), arr.V('bar')])
        picked = a.pick(arr.S(1))
        expected = arr.V('bar')

        assert arr.match(picked, expected)

    def test_pick_vector_idx(self):
        """
        1 2⊃'foo' 'bar'

        'r'
        """
        a = arr.V([arr.V('foo'), arr.V('bar')])
        picked = a.pick(arr.V([1, 2]))
        expected = arr.S('r')

        assert arr.match(picked, expected)

    def test_pick_hirank(self):
        """
        (⊂1 0)⊃2 2⍴'abcd'

        'c'
        """
        a = arr.Array([2, 2], list('abcd'))
        picked = a.pick(arr.S(arr.V([1, 0])))
        expected = arr.S('c')
        assert arr.match(picked, expected)

# 2⊃'pick' ←→ 'c'
# (⊂1 0)⊃2 2⍴'abcd' ←→ 'c'
# 1⊃'foo' 'bar' ←→ 'bar'
# # 1 2⊃'foo' 'bar' ←→ 'r'

class TestKCells:
    def test_0_cells(self):
        shape = [3, 4, 5]
        a = arr.Array(shape, list(range(prod(shape))))
        cells = list(a.kcells(0))
        assert len(cells) == prod(shape)

    def test_1_cells(self):
        shape = [3, 4, 5]
        a = arr.Array(shape, list(range(prod(shape))))
        cells = list(a.kcells(1))
        assert len(cells) == prod(shape[:-1])

    def test_2_cells(self):
        shape = [3, 4, 5]
        a = arr.Array(shape, list(range(prod(shape))))
        cells = list(a.kcells(2))
        assert len(cells) == prod(shape[:-2])

class TestMajorCells:
    def test_major_cells_scalar(self):
        a = arr.S(4)
        assert not list(a.major_cells())

    def test_major_cells_vector(self):
        v = arr.V([0, 1, 2, 3, 4])
        for i, cell in enumerate(v.major_cells()):
            assert arr.match(cell, arr.S(i))

    def test_major_cells_matrix(self):
        m = arr.Array([2, 2], [
            1, 2,
            3, 4
        ])
        rows = list(m.major_cells())
        assert arr.match(rows[0], arr.V([1, 2]))
        assert arr.match(rows[1], arr.V([3, 4]))

    def test_major_cells_array(self):
        shape = [2, 2, 3]
        a = arr.Array(shape, list(range(prod(shape))))
        mats = list(a.major_cells())
        assert len(mats) == shape[0]
        assert arr.match(mats[0], arr.Array([2, 3], [
            0, 1, 2,
            3, 4, 5
        ]))
        assert arr.match(mats[1], arr.Array([2, 3], [
            6,  7,  8,
            9, 10, 11
        ]))

class TestProtElement:
    def test_prot_element_numeric_scalar(self):
        for elem in (0, 6, 9.2, -4.5, complex(8.3, 6.5)):
            zero = arr.Array.prot(arr.S(elem))
            assert arr.match(zero, arr.S(0))

    def test_prot_element_numeric_scalar_array(self):
        for elem in (0, 6, 9.2, -4.5, complex(8.3, 6.5)):
            zero = arr.Array.prot(arr.S(elem))
            assert arr.match(zero, arr.S(0))

    def test_prot_element_numeric_vector(self):
        zero = arr.Array.prot(arr.V([0, 6, 9.2, -4.5, complex(8.3, 6.5)]))
        assert arr.match(zero, arr.S(0))

    def test_prot_element_nested(self):
        """
        nested ← ⊂(1 2 3)'jel'4
        ⊃0⍴nested
        ┌→────────────────┐
        │ ┌→────┐ ┌→──┐   │
        │ │0 0 0│ │   │ 0 │
        │ └~────┘ └───┘   │
        └∊────────────────┘
        """
        nested = arr.S(arr.V([arr.V([1, 2, 3]), arr.V(list('jel')), 4]))
        p = arr.Array.prot(nested)
        expected = arr.V([arr.V([0, 0, 0]), arr.V(list('   ')), 0])
        assert arr.match(p, expected)

class TestTake:
    def test_scalar_overtake(self):
        """
        2↑1
        """
        x = arr.S(1)
        e = x.take(arr.V([2]))
        assert arr.match(e, arr.V([1, 0]))

    def test_simple_take(self):
        x = arr.V([1, 2, 3, 4, 5])
        e = x.take(arr.V([2]))
        assert arr.match(e, arr.V([1, 2]))

    def test_charvec_take(self):
        x = arr.V('hello world')
        e = x.take(arr.V([5]))
        assert arr.match(e, arr.V('hello'))

    def test_charvec_overtake(self):
        x = arr.V('hello')
        e = x.take(arr.V([7]))
        assert arr.match(e, arr.V('hello  '))

    def test_charvec_negative_overtake(self):
        x = arr.V('hello')
        e = x.take(arr.V([-7]))
        assert arr.match(e, arr.V('  hello'))

    def test_simple_negative_take(self):
        x = arr.V([1, 2, 3, 4, 5])
        e = x.take(arr.V([-2]))
        assert arr.match(e, arr.V([4, 5]))

    def test_simple_overtake(self):
        x = arr.V([1, 2])
        e = x.take(arr.V([5]))
        assert arr.match(e, arr.V([1, 2, 0, 0, 0]))

    def test_simple_negative_overtake(self):
        x = arr.V([1, 2])
        e = x.take(arr.V([-5]))
        assert arr.match(e, arr.V([0, 0, 0, 1, 2]))

    def test_take_matrix_1_axis(self):
        x = arr.Array([2, 2], [1, 2, 3, 4])
        e = x.take(arr.V([1]))
        assert arr.match(e, arr.Array([1, 2], [
            1, 2
        ]))
    
    def test_take_matrix_1_axis_neg(self):
        x = arr.Array([2, 2], [
            1, 2,
            3, 4
        ])
        e = x.take(arr.V([-1]))
        assert arr.match(e, arr.Array([1, 2], [
            3, 4
        ]))

    def test_take_matrix_2_axes(self):
        x = arr.Array([2, 2], [
            1, 2,
            3, 4
        ])
        e = x.take(arr.V([1, 1]))
        assert arr.match(e, arr.Array([1, 1], [1]))

    def test_take_matrix_2_axes_neg(self):
        x = arr.Array([2, 2], [
            1, 2,
            3, 4
        ])
        e = x.take(arr.V([-1, 1]))
        assert arr.match(e, arr.Array([1, 1], [3]))

    def test_take_zero_yields_prototype(self):
        """
        x ← ⊂'de'(3 4 5)
        e ← 0 ↑ x           ⍝ An empty array based on x
        ⊃e                  ⍝ Disclosing gets the prototype
        ┌──┬─────┐
        │  │0 0 0│
        └──┴─────┘
        """
        x = arr.S(arr.V([arr.V(list('de')), arr.V([3, 4, 5])]))
        e = x.take(arr.V([0]))
        assert arr.match(e, arr.Array([0], [arr.V([' ', ' ']), arr.V([0, 0, 0])]))

    def test_overtake_yields_prototype(self):
        """
        x ← ⊂'de'(3 4 5)
        e ← 2 ↑ x
        ┌→──────────────────────────────────┐
        │ ┌→─────────────┐ ┌→─────────────┐ │
        │ │ ┌→─┐ ┌→────┐ │ │ ┌→─┐ ┌→────┐ │ │
        │ │ │de│ │3 4 5│ │ │ │  │ │0 0 0│ │ │
        │ │ └──┘ └~────┘ │ │ └──┘ └~────┘ │ │
        │ └∊─────────────┘ └∊─────────────┘ │
        └∊──────────────────────────────────┘
        """
        x = arr.S(arr.V([arr.V(list('de')), arr.V([3, 4, 5])]))
        e = x.take(arr.V([2]))
        expected = arr.V([arr.V([arr.V(list('de')), arr.V([3, 4, 5])]), arr.V([arr.V(list('  ')), arr.V([0, 0, 0])])])
        assert arr.match(e, expected)

    def test_take_two_rows(self):
        """
        2↑3 3⍴⍳9
        ┌→────┐
        ↓0 1 2│
        │3 4 5│
        └~────┘
        """
        a = arr.Array([3, 3], list(range(9)))
        e = a.take(arr.S(2))
        expected = arr.Array([2, 3], [
            0, 1, 2,
            3, 4, 5
        ])
        assert arr.match(e, expected)

    def test_matrix_overtake(self):
        """
        4↑3 3⍴⍳9
        ┌→────┐
        ↓0 1 2│
        │3 4 5│
        │6 7 8│
        │0 0 0│
        └~────┘
        """
        a = arr.Array([3, 3], list(range(9)))
        e = a.take(arr.S(4))
        expected = arr.Array([4, 3], [
            0, 1, 2,
            3, 4, 5,
            6, 7, 8, 
            0, 0, 0,
        ])
        assert arr.match(e, expected)

class TestMix:
    def test_mix_zilde(self):
        """
        ↑⍬
        ┌⊖┐
        │0│
        └~┘
        """
        assert arr.match(arr.Array([0], []).mix(), arr.Array([0], []))

    def test_mix_scalar(self):
        """
        ↑5
        5
        """
        assert arr.match(arr.S(5).mix(), arr.S(5))

    def test_mix_vector(self):
        """
        ↑5 6 7
        ┌→────┐
        │5 6 7│
        └~────┘
        """
        assert arr.match(arr.V([5, 6, 7]).mix(), arr.V([5, 6, 7]))

    def test_mix_equal_lengths(self):
        """
        ↑'abc' 'def' 'ghi'
        ┌→──┐
        ↓abc│
        │def│
        │ghi│
        └───┘
        """
        strs = arr.V([arr.V('abc'), arr.V('def'), arr.V('ghi')])
        expected = arr.Array([3, 3], list('abcdefghi'))
        mixed = strs.mix()
        assert arr.match(mixed, expected)

    def test_mix_unequal_lengths(self):
        """
        ↑'abc' 'defg' 'hijkl'
        ┌→────┐
        ↓abc  │
        │defg │
        │hijkl│
        └─────┘
        """
        strs = arr.V([arr.V('abc'), arr.V('defg'), arr.V('hijkl')])
        expected = arr.Array([3, 5], list('abc  defg hijkl'))
        mixed = strs.mix()
        assert arr.match(mixed, expected)

    def test_mix_nested_contains_scalar(self):
        """
        ↑(1 2)3
        ┌→──┐
        ↓1 2│
        │3 0│
        └~──┘
        """
        v = arr.V([arr.V([1, 2]), 3])
        expected = arr.Array([2, 2], [1, 2, 3, 0])
        mixed = v.mix()
        assert arr.match(mixed, expected)

    def test_messy(self):
        """
        ↑2 2⍴1(1 1 2⍴3 4)(5 6)(2 0⍴0)
        ┌┌┌┌→──┐
        ↓↓↓↓1 0│
        ││││0 0│
        ││││   │
        ││││   │
        ││││3 4│
        ││││0 0│
        ││││   │
        ││││   │
        ││││   │
        ││││5 6│
        ││││0 0│
        ││││   │
        ││││   │
        ││││0 0│
        ││││0 0│
        └└└└~──┘
        """
        # 2 2⍴1(1 1 2⍴3 4)(5 6)(2 0⍴0)
        a = arr.Array([2, 2], [1, arr.Array([1, 1, 2], [3, 4]), arr.V([5, 6]), arr.Array([2, 0], [0])])
        mixed = a.mix()
        expected = arr.Array([2, 2, 1, 2, 2], [1, 0, 0, 0, 3, 4, 0, 0, 5, 6, 0, 0, 0, 0, 0, 0])
        assert arr.match(mixed, expected)

class TestDrop:
    def test_drop_vector(self):
        v = arr.V([1, 2, 3, 4])
        dropped = v.drop(arr.S(2))
        expected = arr.V([3, 4])
        assert arr.match(dropped, expected)

    def test_drop_row_from_matrix(self):
        a = arr.Array([2, 2], [1, 2, 3, 4])
        dropped = a.drop(arr.S(1))
        expected = arr.Array([1, 2], [3, 4])
        assert arr.match(dropped, expected)

    def test_drop_elem_drop_row_from_matrix(self):
        a = arr.Array([2, 2], [1, 2, 3, 4])
        dropped = a.drop(arr.V([1, 1]))
        expected = arr.Array([1, 1], [4])
        assert arr.match(dropped, expected)

    def test_drop_elem_drop_row_from_matrix_neg(self):
        a = arr.Array([2, 2], [1, 2, 3, 4])
        dropped = a.drop(arr.V([-1, -1]))
        expected = arr.Array([1, 1], [1])
        assert arr.match(dropped, expected)

    def test_drop_to_empty_yields_zilde(self):
        a = arr.S(0)
        dropped = a.drop(arr.S(1))
        expected = arr.Array([0], [])
        assert arr.match(dropped, expected)

    def test_drop_000(self):
        """
        1 2↓3 ⍝ 0 0⍴0
        ┌⊖┐
        ⌽0│
        └~┘
        """
        a = arr.S(3)
        dropped = a.drop(arr.V([1, 2]))
        expected = arr.Array([0, 0], [])
        assert arr.match(dropped, expected)

    def test_drop_zilde(self):
        """
        ⍬↓0
        
        0
        """
        a = arr.S(0)
        dropped = a.drop(arr.Array([0], []))
        expected = arr.S(0)
        assert arr.match(dropped, expected)

class TestSplit:
    def test_split_arr(self):
        a = arr.Array([3, 3], list('abcdefghi'))
        expected = arr.V([arr.V('abc'), arr.V('def'), arr.V('ghi')])
        split = a.split()
        assert arr.match(split, expected)

    def test_split_arr2(self):
        shape = [3, 3, 3]
        a = arr.Array(shape, list(range(prod(shape))))
        expected = arr.Array([3, 3], [
            arr.V([ 0,  1,  2]), arr.V([ 3,  4,  5]), arr.V([ 6,  7,  8]),
            arr.V([ 9, 10, 11]), arr.V([12, 13, 14]), arr.V([15, 16, 17]), 
            arr.V([18, 19, 20]), arr.V([21, 22, 23]), arr.V([24, 25, 26])
        ])
        split = a.split()
        assert arr.match(split, expected)

class TestFoldr:
    def test_foldr_add_vector(self):
        v = arr.V(list(range(10_000)))
        total = v.foldr(lambda a, w: a+w)
        assert arr.match(total, arr.S(49995000))

    def test_foldr_subtract_vector(self):
        v = arr.V(list(range(10)))
        total = v.foldr(lambda a, w: a-w)
        assert arr.match(total, arr.S(-5))

    def test_foldr_sum_mat_trailling(self):
        shape = [3, 3, 3]
        a = arr.Array(shape, list(range(prod(shape))))
        summed = a.foldr(lambda a, w: a+w)
        expected = arr.Array([3, 3], [
             3, 12, 21,
            30, 39, 48,
            57, 66, 75
        ])
        assert arr.match(summed, expected)

    def test_foldr_sum_mat_leading(self):
        shape = [3, 3, 3]
        a = arr.Array(shape, list(range(prod(shape))))
        summed = a.foldr(lambda a, w: a+w, axis=0)
        expected = arr.Array([3, 3], [
            27, 30, 33,
            36, 39, 42,
            45, 48, 51
        ])
        assert arr.match(summed, expected)

    def test_foldr_sum_mat_middle(self):
        shape = [3, 3, 3]
        a = arr.Array(shape, list(range(prod(shape))))
        summed = a.foldr(lambda a, w: a+w, axis=1)
        expected = arr.Array([3, 3], [
             9, 12, 15,
            36, 39, 42,
            63, 66, 69
        ])
        assert arr.match(summed, expected)

class TestLaminate:
    def test_laminate_arr_scalar(self):
        """
        a←2 3⍴1 2 3 4 5 6
        a⍪1
        ┌→────┐
        ↓1 2 3│
        │4 5 6│
        │1 1 1│
        └~────┘
        """
        a = arr.Array([2, 3], [
            1, 2, 3, 
            4, 5, 6
        ])
        b = a.laminate(arr.S(1))
        expected = arr.Array([3, 3], [
            1, 2, 3, 
            4, 5, 6, 
            1, 1, 1
        ])
        assert arr.match(b, expected)

    def test_length_error(self):
        """
        a←2 3⍴1 2 3 4 5 6
        a⍪1 1
        """
        a = arr.Array([2, 3], [
            1, 2, 3, 
            4, 5, 6
        ])
        with pytest.raises(arr.LengthError):
            a.laminate(arr.V([1, 1]))

    def test_laminate_arr_vec(self):
        """
        a←2 3⍴1 2 3 4 5 6
        a⍪7 8 9
        ┌→────┐
        ↓1 2 3│
        │4 5 6│
        │7 8 9│
        └~────┘
        """
        a = arr.Array([2, 3], [
            1, 2, 3, 
            4, 5, 6
        ])
        b = a.laminate(arr.V([7, 8, 9]))
        expected = arr.Array([3, 3], [
            1, 2, 3, 
            4, 5, 6, 
            7, 8, 9
        ])
        assert arr.match(b, expected)

    def test_laminate_arr_arr(self):
        """
        a←2 3⍴'abcdef'
        b←3 3⍴'ghijklmno'
        a⍪b
        ┌→──┐
        ↓abc│
        │def│
        │ghi│
        │jkl│
        │mno│
        └───┘
        """
        a = arr.Array([2, 3], list('abcdef'))
        b = arr.Array([3, 3], list('ghijklmno'))
        a_lam_b = a.laminate(b)
        expected = arr.Array([5, 3], list('abcdefghijklmno'))
        assert arr.match(a_lam_b, expected)

class TestTable:
    def test_table_scalar(self):
        s = arr.S(3)
        tabled = s.table()
        expected = arr.Array([1, 1], [s.data[0]])
        assert arr.match(tabled, expected)

    def test_table_vector(self):
        v = arr.V([3, 1, 4])
        tabled = v.table()
        expected = arr.Array([3, 1], v.data)
        assert arr.match(tabled, expected)

    def test_table_hirank(self):
        """
        a←2 3 4 2⍴⎕A
        (⍪a)≡2 24⍴a
        """
        a = arr.Array.fill([2, 3, 4, 2], list('ABCDEFGHIJKLMNOPQRSTUVWXY'))
        tabled = a.table()
        expected = a.reshape([2, 24])
        assert arr.match(tabled, expected)

class TestMutate:
    def test_mutate_vector(self):
        a = arr.V([1, 0, 0, 1, 0, 1])
        idx = arr.V([1, 2])
        vals = arr.V([1, 1])
        a.mutate(idx, vals)
        assert arr.match(a, arr.V([1, 1, 1, 1, 0, 1]))

    def test_mutate_vector_scalar_extension(self):
        """
        a←⍳5⋄a[1 3]←7⋄a

        ┌→────────┐
        │0 7 2 7 4│
        └~────────┘
        """
        v = arr.V([0, 1, 2, 3, 4])
        idx = arr.V([1, 3])
        val = arr.S(7)
        v.mutate(idx, val)
        assert arr.match(v, arr.V([0, 7, 2, 7, 4]))

    def test_mutate_matrix(self):
        a = arr.Array([3, 2], [
            1, 0, 
            0, 1, 
            0, 1
        ])

        idx = arr.V([arr.V([1, 1]), arr.V([0, 0])])
        vals = arr.V([-1, 9])
        a.mutate(idx, vals)
        assert arr.match(a, arr.Array([3, 2], [9, 0, 0, -1, 0, 1]))

    def test_mutate_matrix_index_error(self):
        a = arr.Array([3, 2], [
            1, 0, 
            0, 1, 
            0, 1
        ])

        idx = arr.V([arr.V([1, 2]), arr.V([0, 0])])  # [1, 2] is out of bounds
        vals = arr.V([-1, 9])

        with pytest.raises(IndexError):
            a.mutate(idx, vals)

class TestContains:
    def test_contains_vector_scalar_true(self):
        """
        a ← 1 2 3 4 5
        b ← 3
        b ∊ a
        1
        """
        a = arr.V([1, 2, 3, 4, 5])
        b = arr.S(3)
        assert arr.match(a.contains(b), arr.S(1))

    def test_contains_vector_scalar_false(self):
        """
        a ← 1 2 3 4 5
        b ← 9
        b ∊ a
        0
        """
        a = arr.V([1, 2, 3, 4, 5])
        b = arr.S(9)
        assert arr.match(a.contains(b), arr.S(0))

    def test_contains_matrix_vector(self):
        """
        a ← 3 2⍴(1 2)(1 3)4(1 2 3 4)(9 8)'hello'
        b ← (1 2)(1 2 3 4)5
        b∊a
        ┌→────┐
        │1 1 0│
        └~────┘
        """
        a = arr.Array([3, 2], [
            arr.V([1, 2]), arr.V([1, 3]), 
            4,             arr.V([1, 2, 3, 4]),
            arr.V([9, 8]), arr.V('hello'), 

        ])
        b = arr.V([arr.V([1, 2]), arr.V([1, 2, 3, 4]), 5])
        found = a.contains(b)

        assert arr.match(found, arr.V([1, 1, 0]))

class TestIndexOf:
    def test_index_of_rank_error(self):
        """
        3⍳1 2 3 4 5
         ^
        RANK ERROR
        """
        three = arr.S(3)
        with pytest.raises(RankError):
            three.index_of(arr.V([1, 2, 3, 4, 5]))

    def test_index_of_scalar(self):
        """
        1 2 3 4 5⍳3
 
        2
        """
        v = arr.V([1, 2, 3, 4, 5])
        io = v.index_of(arr.S(3))
        expected = arr.S(2)
        assert arr.match(io, expected)

    def test_index_of_non_present(self):
        """
        1 2 3 4 5⍳9
 
        5
        """
        v = arr.V([1, 2, 3, 4, 5])
        io = v.index_of(arr.S(9))
        expected = arr.S(5)
        assert arr.match(io, expected)

    def test_index_of_vector_vector(self):
        """
        1 2 3 4 5⍳2 4
        ┌→──┐
        │1 3│
        └~──┘
        """
        v = arr.V([1, 2, 3, 4, 5])
        io = v.index_of(arr.V([2, 4]))
        expected = arr.V([1, 3])
        assert arr.match(io, expected)

    def test_index_of_matrix(self):
        """
        (3 2⍴1 2 3 4 5 6)⍳5 6
 
        2
        """
        m = arr.Array([3, 2], [1, 2, 3, 4, 5, 6])
        io = m.index_of(arr.V([5, 6]))
        expected = arr.S(2)
        assert arr.match(io, expected)

    def test_index_of_matrix_matrix(self):
        """
        (3 2⍴1 2 3 4 5 6)⍳2 2⍴3 4 5 6
        ┌→──┐
        │1 2│
        └~──┘
        """
        m = arr.Array([3, 2], [1, 2, 3, 4, 5, 6])
        io = m.index_of(arr.Array([2, 2], [3, 4, 5, 6]))
        expected = arr.V([1, 2])
        assert arr.match(io, expected)

class TestReplicate:
    def test_scalar_replicate(self):
        a = arr.V([1, 2, 3])
        i = arr.S(3)
        e = arr.V([1, 1, 1, 2, 2, 2, 3, 3, 3])
        assert arr.match(a.replicate(i), e)

    def test_compress(self):
        a = arr.V([1, 2, 3, 4, 5, 6, 7])
        i = arr.V([1, 0, 1, 0, 1, 0, 1])
        e = arr.V([1, 3, 5, 7])
        assert arr.match(a.replicate(i), e)

    def test_replicate(self):
        a = arr.V([1, 2, 3])
        i = arr.V([2, 0, 3])
        e = arr.V([1, 1, 3, 3, 3])
        assert arr.match(a.replicate(i), e)

    def test_replicate_hirank_error(self):
        a = arr.V([1, 2, 3, 4])
        i = arr.Array([2, 2], [2, 0, 3, 1])

        with pytest.raises(arr.RankError):
            a.replicate(i)

class TestWithout:
    def test_without(self):
        """
        1 2 3 4 5 6 7~1 3 5 7
        ┌→────┐
        │2 4 6│
        └~────┘
        """
        a = arr.V([1, 2, 3, 4, 5, 6, 7])
        i = arr.V([1, 3, 5, 7])
        e = arr.V([2, 4, 6])
        r = a.without(i)
        assert arr.match(r, e)

    def test_without_letters(self):
        """
        1 2 3 4 5 6 7~1 3 5 7
        ┌→────┐
        │2 4 6│
        └~────┘
        """
        a = arr.V("mississippi")
        i = arr.V("sp")
        e = arr.V("miiii")
        r = a.without(i)
        assert arr.match(r, e)

    def test_without_nested(self):
        """
        'ab' 'cd' 'ad'~⊂'ab'
        """
        a = arr.V([arr.V('ab'), arr.V('ac'), arr.V('ad')])
        i = arr.S(arr.V('ab'))
        r = a.without(i)
        e = arr.V([arr.V('ac'), arr.V('ad')])
        assert arr.match(r, e)

class TestEnlist:
    def test_enlist_zilde(self):
        """
        ∊⍬
        ┌⊖┐
        │0│
        └~┘
        """
        a = arr.Array([0], [])
        assert arr.match(a.enlist(), a)

    def test_enlist_simple_scalar(self):
        """
        ∊2
        ┌→┐
        │2│
        └~┘
        """
        a = arr.S(2)
        assert arr.match(a.enlist(), arr.V([2]))

    def test_enlist_simple_vector(self):
        a = arr.V([1, 2, 3, 4, 5, 6, 7])
        assert arr.match(a.enlist(), a)

    def test_enlist_matrix(self):
        a = arr.Array([2, 3], [
            1, 2, 3,
            4, 5, 6
        ])
        assert arr.match(a.enlist(), arr.V([1, 2, 3, 4, 5, 6]))

    def test_enlist_nested(self):
        a = arr.Array([2, 3], [
            arr.V([1, 2]), arr.V([3,  4]), arr.V([ 5,  6]),
            arr.V([7, 8]), arr.V([9, 10]), arr.V([11, 12])
        ])
        assert arr.match(a.enlist(), arr.V([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]))

    def test_enlist_enclosures(self):
        """
        ∊(⊂1 2)(⊂1 2)
        ┌→──────┐
        │1 2 1 2│
        └~──────┘
        """
        v = arr.V([arr.S(arr.V([1, 2])), arr.S(arr.V([1, 2]))])
        r = v.enlist()
        e = arr.V([1, 2, 1, 2])
        assert arr.match(r, e)

class TestWhere:
    def test_where_scalar(self):
        """
        ⍸1
        ┌→────┐
        │ ┌⊖┐ │
        │ │0│ │
        │ └~┘ │
        └∊────┘
        """
        s = arr.S(1)
        r = s.where()
        e = arr.V([arr.Array([0], [])])
        assert arr.match(r, e)

    def test_where_zeros(self):
        """
        ⍸0 0 0 0
        ┌⊖┐
        │0│
        └~┘
        """
        v = arr.V([0, 0, 0, 0])
        r = v.where()
        e = arr.Array([0], [])
        assert arr.match(r, e)

    def test_where_vector(self):
        """
        ⍸0 1 0 1
        ┌→──┐
        │1 3│
        └~──┘
        """
        v = arr.V([0, 1, 0, 1])
        r = v.where()
        e = arr.V([1, 3])
        assert arr.match(r, e)

    def test_where_matrix(self):
        """
        ⍸2 2⍴0 1 0 1
        ┌→────────────┐
        │ ┌→──┐ ┌→──┐ │
        │ │0 1│ │1 1│ │
        │ └~──┘ └~──┘ │
        └∊────────────┘
        """
        v = arr.Array([2, 2], [0, 1, 0, 1])
        r = v.where()
        e = arr.V([arr.V([0, 1]), arr.V([1, 1])])
        assert arr.match(r, e)

class TestTranspose:
    def test_monadic_transpose_empty(self):
        a = arr.Array([0], [])
        r = a.transpose()
        e = arr.Array([0], [])
        assert arr.match(r, e)

    def test_monadic_transpose_rank1(self):
        a = arr.V([1, 2, 3, 4])
        r = a.transpose()
        e = arr.V([1, 2, 3, 4])
        assert arr.match(r, e)

    def test_monadic_transpose_rank2(self):
        shape = [2, 2]
        total = prod(shape)
        a = arr.Array(shape, list(range(total)))
        r = a.transpose()
        e = arr.Array([2, 2], [0, 2, 1, 3])
        assert arr.match(r, e)

    def test_dyadic_transpose(self):
        shape = [2, 3, 4]
        total = prod(shape)
        a = arr.Array(shape, list(range(total)))
        r = a.transpose([2, 0, 1])
        e = arr.Array([3, 4, 2], [0,12,1,13,2,14,3,15,4,16,5,17,6,18,7,19,8,20,9,21,10,22,11,23])
        assert arr.match(r, e)
        
    def test_dyadic_transpose_diagonal(self):
        shape = [3, 3]
        total = prod(shape)
        a = arr.Array(shape, list(range(total)))
        r = a.transpose([0, 0])
        e = arr.V([0, 4, 8])
        assert arr.match(r, e)

    def test_dyadic_transpose_zero_vector(self):
        """
        0⍉1 2
        ┌→──┐
        │1 2│
        └~──┘
        """
        v = arr.V([1, 2])
        vv = v.transpose([0])
        expected = arr.V([1, 2])
        assert arr.match(v, vv)

class TestIndexGen:
    def test_index_gen_scalar(self):
        length = 10
        r = arr.S(length).index_gen()
        e = arr.V(list(range(length)))
        assert arr.match(r, e)

    def test_index_gen_2d(self):
        """
        ⍳2 2
        ┌→────────────┐
        ↓ ┌→──┐ ┌→──┐ │
        │ │0 0│ │0 1│ │
        │ └~──┘ └~──┘ │
        │ ┌→──┐ ┌→──┐ │
        │ │1 0│ │1 1│ │
        │ └~──┘ └~──┘ │
        └∊────────────┘
        """
        r = arr.V([2, 2]).index_gen()
        e = arr.Array([2, 2], [arr.V([0, 0]), arr.V([0, 1]), arr.V([1, 0]), arr.V([1, 1])])
        assert arr.match(r, e)

class TestEncode:
    def test_encode(self):
        b = arr.encode([2, 2, 2], 5)
        assert b == [1, 0, 1]

    def test_encode_mixed_radix(self):
        b = arr.encode([24, 60, 60], 10_000)
        assert b == [2, 46, 40]

class TestDecode:
    def test_decode(self):
        s = arr.decode([2, 2, 2, 2], [1, 1, 0, 1])
        assert s == 13

    def test_decode_mixed_radix(self):
        s = arr.decode([24, 60, 60], [2, 46, 40])
        assert s == 10_000

class TestGrade:
    def test_grade_down_vector(self):
        v = arr.V([97, 62, 23, 11, 83, 73, 8, 35, 68, 21, 38, 47, 70, 65, 85, 0, 64, 76, 87, 80])
        permutation = v.grade()
        ordered = v.at(arr.V(list(permutation)))
        assert ordered.data == sorted(v.data)

    def test_grade_up_vector(self):
        v = arr.V([97, 62, 23, 11, 83, 73, 8, 35, 68, 21, 38, 47, 70, 65, 85, 0, 64, 76, 87, 80])
        permutation = v.grade(reverse=True)
        v_ordered = v.at(arr.V(list(permutation)))
        assert v_ordered.data == sorted(v.data, reverse=True)

    def test_grade_down_matrix(self):
        m = arr.Array([2, 2], [9, 3, 5, 1])
        permutation = m.grade()
        m_ordered = m.rect([list(permutation), []])
        assert m_ordered.data == [5, 1, 9, 3]

    def test_grade_char_matrix(self):
        """
        ⍋3 3⍴'BOBALFZAK'
        ┌→────┐
        │1 0 2│
        └~────┘
        """
        m = arr.Array([3, 3], list('BOBALFZAK'))
        permutation = m.grade()
        assert permutation == [1, 0, 2]
