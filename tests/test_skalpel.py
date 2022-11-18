import operator

import pytest

from apl.errors import DomainError, LengthError, RankError
import apl.arr as arr
from apl.parser import Parser
from apl.skalpel import each, pervade, mpervade, reduce_first, run, TYPE, encode, decode, bool_not, rank, power
from apl.stack import Stack

def run_code(src):
    parser = Parser()
    ast = parser.parse(src)
    code = ast.emit()
    env = {}
    stack = Stack()
    run(code, env, 0, stack)
    return stack.stack[stack.stackptr]

class TestPervade:
    def test_mat_plus_scalar(self):
        plus = pervade(operator.add)
        a = arr.Array([3, 2], [54, 7, 88, 4, 73, 3, 1])
        b = arr.S(12)
        result = plus(a, b)
        assert arr.match(result, arr.Array([3, 2], [66, 19, 100, 16, 85, 15]))

    def test_scalar_plus_mat(self):
        plus = pervade(operator.add)
        a = arr.Array([3, 2], [54, 7, 88, 4, 73, 3, 1])
        b = arr.S(12)
        result = plus(b, a)
        assert arr.match(result, arr.Array([3, 2], [66, 19, 100, 16, 85, 15]))

    def test_vec_plus_vec(self):
        plus = pervade(operator.add)
        a = arr.V([54, 7, 88, 4, 73, 3, 1])
        b = arr.V([12, 8, 11, 7, 21, 7, 9])
        result = plus(b, a)
        assert arr.match(result, arr.V([66, 15, 99, 11, 94, 10, 10]))

    def test_non_simple(self):
        plus = pervade(operator.add)
        a = arr.Array([3, 2], [54, 7, arr.Array([2, 2], [1, 1, 1, 1]), 4, 73, 3])
        b = arr.S(12)
        result = plus(b, a)
        expected = arr.Array([3, 2], [66, 19, arr.Array([2, 2], [13, 13, 13, 13]), 16, 85, 15])
        assert arr.match(result, expected)

    def test_nested_left_scalar_right(self):
        """
        (1 2 (3 4))1(2 3⍴1)+1
        ┌→────────────────────────┐
        │ ┌→──────────┐   ┌→────┐ │
        │ │     ┌→──┐ │ 2 ↓2 2 2│ │
        │ │ 2 3 │4 5│ │   │2 2 2│ │
        │ │     └~──┘ │   └~────┘ │
        │ └∊──────────┘           │
        └∊────────────────────────┘
        """
        plus = pervade(operator.add)
        alpha = arr.V([arr.V([1, 2, arr.V([3, 4])]), 1, arr.Array.fill([2, 3], [1])])
        omega = arr.S(1)
        result = plus(alpha, omega)
        expected = arr.V([arr.V([2, 3, arr.V([4, 5])]), 2, arr.Array.fill([2, 3], [2])])
        assert arr.match(result, expected)

    def test_scalar_left_nested_right(self):
        """
        1+(1 2 (3 4))1(2 3⍴1)
        ┌→────────────────────────┐
        │ ┌→──────────┐   ┌→────┐ │
        │ │     ┌→──┐ │ 2 ↓2 2 2│ │
        │ │ 2 3 │4 5│ │   │2 2 2│ │
        │ │     └~──┘ │   └~────┘ │
        │ └∊──────────┘           │
        └∊────────────────────────┘
        """
        plus = pervade(operator.add)
        alpha = arr.S(1)
        omega = arr.V([arr.V([1, 2, arr.V([3, 4])]), 1, arr.Array.fill([2, 3], [1])])
        result = plus(alpha, omega)
        expected = arr.V([arr.V([2, 3, arr.V([4, 5])]), 2, arr.Array.fill([2, 3], [2])])
        assert arr.match(result, expected)

    def test_right_enclosed_is_extended(self):
        """
        1 2 3+⊂100 200
        ┌→──────────────────────────────┐
        │ ┌→──────┐ ┌→──────┐ ┌→──────┐ │
        │ │101 201│ │102 202│ │103 203│ │
        │ └~──────┘ └~──────┘ └~──────┘ │
        └∊──────────────────────────────┘
        """
        alpha = arr.V([1, 2, 3])
        omega = arr.S(arr.V([100, 200]))
        expected = arr.V([arr.V([101, 201]), arr.V([102, 202]), arr.V([103, 203])])
        plus = pervade(operator.add)
        result = plus(alpha, omega)
        assert arr.match(result, expected)

    def test_left_enclosed_is_extended(self):
        """
        (⊂100 200)+1 2 3
        ┌→──────────────────────────────┐
        │ ┌→──────┐ ┌→──────┐ ┌→──────┐ │
        │ │101 201│ │102 202│ │103 203│ │
        │ └~──────┘ └~──────┘ └~──────┘ │
        └∊──────────────────────────────┘
        """
        alpha = arr.S(arr.V([100, 200]))
        omega = arr.V([1, 2, 3])
        expected = arr.V([arr.V([101, 201]), arr.V([102, 202]), arr.V([103, 203])])
        plus = pervade(operator.add)
        result = plus(alpha, omega)
        assert arr.match(result, expected)

    def test_monadic_pervade(self):
        """
        -4(1 2 3)1j2
        
        ¯4(¯1 ¯2 ¯3)¯1j¯2
        """
        v = arr.V([4, arr.V([1, 2, 3]), complex(1, 2)])
        negate = mpervade(lambda o: -o)
        result = negate(v)
        expected = arr.V([-4, arr.V([-1, -2, -3]), complex(-1, -2)])
        assert arr.match(result, expected)

    def test_both_singletons(self):
        """
        (1 1⍴2)+1 1 1⍴3
        ┌┌→┐
        ↓↓5│
        └└~┘
        """
        alpha = arr.Array([1, 1], [2])
        omega = arr.Array([1, 1, 1], [3])
        expected = arr.Array([1, 1, 1], [5])
        plus = pervade(operator.add)
        result = plus(alpha, omega)
        assert arr.match(result, expected)

    def test_left_singleton(self):
        """
        (1 1⍴2)+2 2⍴3
        ┌→──┐
        ↓5 5│
        │5 5│
        └~──┘
        """
        alpha = arr.Array([1, 1], [2])
        omega = arr.Array([2, 2], [3, 3, 3, 3])
        expected = arr.Array([2, 2], [5, 5, 5, 5])
        plus = pervade(operator.add)
        result = plus(alpha, omega)
        assert arr.match(result, expected)

    def test_right_singleton(self):
        """
        (2 2⍴3) + 1 1⍴2
        ┌→──┐
        ↓5 5│
        │5 5│
        └~──┘
        """
        alpha = arr.Array([2, 2], [3, 3, 3, 3])
        omega = arr.Array([1, 1], [2])
        expected = arr.Array([2, 2], [5, 5, 5, 5])
        plus = pervade(operator.add)
        result = plus(alpha, omega)
        assert arr.match(result, expected)

    def test_unequal_lengths_raises(self):
        alpha = arr.V([3, 3, 3, 3])
        omega = arr.V([1, 1])
        plus = pervade(operator.add)
        with pytest.raises(LengthError):
            plus(alpha, omega)

class TestRank:
    def test_rank_derives_monad(self):

        omega = arr.Array([2, 3, 4], [
            36, 99, 20,  5,
            63, 50, 26, 10,
            64, 90, 68, 98,

            66, 72, 27, 74,
            44,  1, 46, 62,
            48,  9, 81, 22,
        ])

        matrices = rank('⊂', arr.S(2), None, omega, {}, Stack())

        expected_m = arr.V([
            arr.Array([3, 4], [36, 99, 20, 5, 63, 50, 26, 10, 64, 90, 68, 98]),
            arr.Array([3, 4], [66, 72, 27, 74, 44, 1, 46, 62, 48, 9, 81, 22])
        ])

        assert arr.match(matrices, expected_m)

        rows = rank('⊂', arr.S(1), None, omega, {}, Stack())

        expected_r = arr.Array([2, 3], [
            arr.V([36, 99, 20, 5]), arr.V([63, 50, 26, 10]), arr.V([64, 90, 68, 98]),
            arr.V([66, 72, 27, 74]), arr.V([44,  1, 46, 62]), arr.V([48,  9, 81, 22]),
        ])

        assert arr.match(rows, expected_r)

    def test_rank_derives_dyad_1_0(self):

        alpha = arr.Array([2, 2], [
            1, 2,
            3, 4
        ])

        omega = arr.V([9, 5])
        rowsum = rank('+', arr.V([1, 0]), alpha, omega, {}, Stack())

        expected = arr.Array([2, 2], [
            10, 11,
            8, 9
        ])

        assert arr.match(rowsum, expected)

    def test_rank_derives_dyad_1_1(self):
        """
        (2 2⍴1 2 3 4)+⍤1 1⊢5 9
        ┌→───┐
        ↓6 11│
        │8 13│
        └~───┘
        """
        alpha = arr.Array([2, 2], [
            1, 2,
            3, 4
        ])

        omega = arr.V([5, 9])
        rowsum = rank('+', arr.V([1, 1]), alpha, omega, {}, Stack())

        expected = arr.Array([2, 2], [
            6, 11,
            8, 13
        ])

        assert arr.match(rowsum, expected)

    def test_rank_hirank(self):
        """
        9 5 (+⍤1 0) 2 3⍴1 2 3 4 5 6
        ┌┌→────┐
        ↓↓10  6│
        ││11  7│
        ││12  8│
        ││     │
        ││13  9│
        ││14 10│
        ││15 11│
        └└~────┘
        """
        alpha = arr.V([9, 5])
        omega = arr.Array([2, 3], [
            1, 2, 3,
            4, 5, 6,
        ])
        sums = rank('+', arr.V([1, 0]), alpha, omega, {}, Stack())
        expected = arr.Array([2, 3, 2], [
            10, 6, 
            11, 7, 
            12, 8, 
            
            13, 9, 
            14, 10, 
            15, 11
        ])

        assert arr.match(sums, expected)

class TestReduce:
    def test_reduce_first(self):
        r = reduce_first('+', None, None, arr.Array([2, 2], [1, 2, 3, 4]), None, None)
        assert arr.match(r, arr.V([4, 6]))

class TestPower:
    def test_power_for_loop(self):
        """
        1 (+⍣5) 0     
    
        5
        """
        alpha = arr.S(1)
        omega = arr.S(0)
        count = arr.S(5)
        total = power('+', count, alpha, omega, {}, Stack())
        expected = arr.S(5)

        assert arr.match(total, expected)

    def test_power_split_twice(self):
        """
        ⊢cube←2 2 2⍴⎕a
        ┌┌→─┐
        ↓↓AB│
        ││CD│
        ││  │
        ││EF│
        ││GH│
        └└──┘
        (↓⍣2) cube
        ┌→────────────────────────────┐
        │ ┌→──────────┐ ┌→──────────┐ │
        │ │ ┌→─┐ ┌→─┐ │ │ ┌→─┐ ┌→─┐ │ │
        │ │ │AB│ │CD│ │ │ │EF│ │GH│ │ │
        │ │ └──┘ └──┘ │ │ └──┘ └──┘ │ │
        │ └∊──────────┘ └∊──────────┘ │
        └∊────────────────────────────┘
        """
        cube = arr.Array([2, 2, 2], list('ABCDEFGH'))
        count = arr.S(2)
        split2 = power('↓', count, None, cube, {}, Stack())
        expected = arr.V([
            arr.V([arr.V(['A', 'B']), arr.V(['C', 'D'])]), 
            arr.V([arr.V(['E', 'F']), arr.V(['G', 'H'])])
        ])

        assert arr.match(split2, expected)

class TestDecode:
    def test_decode_left_scalar(self):
        """
        2 ⊥ 1 1 0 1

        13
        """
        alpha = arr.S(2)
        omega = arr.V([1, 1, 0, 1])
        expected = arr.S(13)
        assert arr.match(decode(alpha, omega), expected)

    def test_decode_right_scalar(self):
        """
        2 2 2⊥1
        
        7
        """
        alpha = arr.V([2, 2, 2])
        omega = arr.S(1)
        expected = arr.S(7)
        assert arr.match(decode(alpha, omega), expected)

    def test_decode_vector_vector(self):
        """
        24 60 60 ⊥ 2 46 40

        10000
        """
        alpha = arr.V([24, 60, 60])
        omega = arr.V([2, 46, 40])
        expected = arr.S(10_000)
        assert arr.match(decode(alpha, omega), expected)

    def test_decode_hirank_alpha_omega(self):
        """
        Decode is really doing an inner product. 

        (4 3⍴1 1 1 2 2 2 3 3 3 4 4 4)⊥3 8⍴0 0 0 0 1 1 1 1 0 0 1 1 0 0 1 1 0 1 0 1 0 1 0 1
        ┌→──────────────────┐
        ↓0 1 1 2  1  2  2  3│
        │0 1 2 3  4  5  6  7│
        │0 1 3 4  9 10 12 13│
        │0 1 4 5 16 17 20 21│
        └~──────────────────┘
        """
        alpha = arr.Array([4, 3], [
            1, 1, 1,
            2, 2, 2,
            3, 3, 3,
            4, 4, 4,
        ])

        omega = arr.Array([3, 8], [
            0, 0, 0, 0, 1, 1, 1, 1,
            0, 0, 1, 1, 0, 0, 1, 1,
            0, 1, 0, 1, 0, 1, 0, 1,
        ])

        expected = arr.Array([4, 8], [
            0, 1, 1, 2,  1,  2,  2,  3,
            0, 1, 2, 3,  4,  5,  6,  7,
            0, 1, 3, 4,  9, 10, 12, 13, 
            0, 1, 4, 5, 16, 17, 20, 21,
        ])

        result = decode(alpha, omega)
        assert arr.match(result, expected)

    def test_decode_raises_rank_error(self):
        alpha = arr.Array([4, 2], [
            1, 1,
            2, 2,
            3, 3,
            4, 4,
        ])
        omega = arr.Array([3, 8], [
            0, 0, 0, 0, 1, 1, 1, 1,
            0, 0, 1, 1, 0, 0, 1, 1,
            0, 1, 0, 1, 0, 1, 0, 1,
        ])

        with pytest.raises(RankError):
            decode(alpha, omega)

    def test_decode_extend_left_last_axis_is_1(self):
        """
        (4 1⍴1 2 3 4)⊥3 8⍴0 0 0 0 1 1 1 1 0 0 1 1 0 0 1 1 0 1 0 1 0 1 0 1
        ┌→──────────────────┐
        ↓0 1 1 2  1  2  2  3│
        │0 1 2 3  4  5  6  7│
        │0 1 3 4  9 10 12 13│
        │0 1 4 5 16 17 20 21│
        └~──────────────────┘
        """
        alpha = arr.Array([4, 1], [
            1,
            2,
            3,
            4,
        ])
                
        omega = arr.Array([3, 8], [
            0, 0, 0, 0, 1, 1, 1, 1,
            0, 0, 1, 1, 0, 0, 1, 1,
            0, 1, 0, 1, 0, 1, 0, 1,
        ])

        expected = arr.Array([4, 8], [
            0, 1, 1, 2,  1,  2,  2,  3,
            0, 1, 2, 3,  4,  5,  6,  7,
            0, 1, 3, 4,  9, 10, 12, 13, 
            0, 1, 4, 5, 16, 17, 20, 21,
        ])

        result = decode(alpha, omega)
        assert arr.match(result, expected)

    def test_decode_extend_right_first_axis_is_1(self):
        """
        (4 1⍴1 2 3 4)⊥1 8⍴0 0 0 0 1 1 1 1
        ┌→──────────────────┐
        ↓0 0 0 0  3  3  3  3│
        │0 0 0 0  7  7  7  7│
        │0 0 0 0 13 13 13 13│
        │0 0 0 0 21 21 21 21│
        └~──────────────────┘
        """
        alpha = arr.Array([4, 3], [
            1, 1, 1,
            2, 2, 2,
            3, 3, 3,
            4, 4, 4,
        ])

        omega = arr.Array([1, 8], [
            0, 0, 0, 0, 1, 1, 1, 1,
        ])

        expected = arr.Array([4, 8], [
            0, 0, 0, 0,  3,  3,  3,  3,
            0, 0, 0, 0,  7,  7,  7,  7,
            0, 0, 0, 0, 13, 13, 13, 13,
            0, 0, 0, 0, 21, 21, 21, 21,
        ])

        result = decode(alpha, omega)
        assert arr.match(result, expected)

    def test_decode_vec_left_hirank_right(self):
        """
        1760 3 12⊥3 3⍴1 1 1 2 0 3 0 1 8

        60 37 80
        """
        alpha = arr.V([1760, 3, 12])

        omega = arr.Array([3, 3], [
            1, 1, 1,
            2, 0, 3,
            0, 1, 8
        ])
        expected = arr.V([60, 37, 80])

        result = decode(alpha, omega)
        assert arr.match(result, expected)

    def test_scalar_left_hirank_right(self):
        """
        2⊥3 8⍴0 0 0 0 1 1 1 1 0 0 1 1 0 0 1 1 0 1 0 1 0 1 0 1
        
        0 1 2 3 4 5 6 7
        """
        alpha = arr.S(2)

        omega = arr.Array([3, 8], [
            0, 0, 0, 0, 1, 1, 1, 1,
            0, 0, 1, 1, 0, 0, 1, 1,
            0, 1, 0, 1, 0, 1, 0, 1,
        ])

        expected = arr.V([0, 1, 2, 3, 4, 5, 6, 7])
        result = decode(alpha, omega)
        assert arr.match(result, expected)

class TestEncode:
    def test_encode_scalar_left(self):
        """
        10⊤5 15 125

        5 5 5
        """
        alpha = arr.S(10)
        omega = arr.V([5, 15, 125])
        expected = arr.V([5, 5, 5])
        result = encode(alpha, omega)
        assert arr.match(result, expected)

class TestBitwise:
    def test_or(self):
        bwo = pervade(lambda x, y:x|y)
        a = arr.V([1, 0, 1, 0])
        b = arr.V([0, 1, 0, 1])
        c = bwo(a, b)
        assert arr.match(c, arr.V([1, 1, 1, 1]))

    def test_not_flat(self):
        """
        ~3 3⍴1 0 1  0 1 0  1 1 1
        ┌→────┐
        ↓0 1 0│
        │1 0 1│
        │0 0 0│
        └~────┘
        """
        a = arr.Array([3, 3], [
            1, 0, 1,
            0, 1, 0,
            1, 1, 1,
        ])
        
        bitflipped = bool_not(a)

        expected =  arr.Array([3, 3], [
            0, 1, 0,
            1, 0, 1,
            0, 0, 0,
        ])

        assert arr.match(bitflipped, expected)

    def test_not_nested(self):
        """
        ~(1 0 1) (0 1 0) (1 1 1)
        ┌→────────────────────────┐
        │ ┌→────┐ ┌→────┐ ┌→────┐ │
        │ │0 1 0│ │1 0 1│ │0 0 0│ │
        │ └~────┘ └~────┘ └~────┘ │
        └∊────────────────────────┘
        """
        v = arr.V([arr.V([1, 0, 1]), arr.V([0, 1, 0]), arr.V([1, 1, 1])])
        
        bitflipped = bool_not(v)

        expected =  arr.V([arr.V([0, 1, 0]), arr.V([1, 0, 1]), arr.V([0, 0, 0])])

        assert arr.match(bitflipped, expected)

    def test_not_non_boolean_raises(self):
        """
        ~1 2 3
        
        DOMAIN ERROR
            ~1 2 3
            ∧
        """
        v = arr.V([1, 2, 3])
        
        with pytest.raises(DomainError):
            bool_not(v)

class TestEach:
    def test_each(self):
        a = arr.V([arr.V([1, 0]), arr.V([1, 0, 1]), arr.V([1, 0, 1, 0])])
        b = each('≢', None, None, a, None, None)
        assert arr.match(b, arr.V([2, 3, 4]))

class TestRun:
    def test_arith(self):
        src = "1+2"
        result = run_code(src)
        assert arr.match(result.payload, arr.S(3))

    def test_mop_deriving_monad(self):
        src = "+⌿1 2 3 4 5"
        result = run_code(src)
        assert arr.match(result.payload, arr.S(15))
        
    def test_mop_deriving_dyad(self):
        src = "1 +⍨ 2"
        result = run_code(src)
        assert arr.match(result.payload, arr.S(3))

    def test_diamond(self):
        src = "v←⍳99⋄s←+⌿v"
        parser = Parser()
        ast = parser.parse(src)
        code = ast.emit()
        env = {}
        stack = Stack()
        run(code, env, 0, stack)
        assert 'v' in env
        assert 's' in env
        s = env['s'].payload
        assert isinstance(s, arr.Array)
        assert s.shape == []
        assert s.data[0] == 4851

    def test_dop_deriving_dyad(self):
        src = "1 2 3 ⌊⍥≢ 1 2 3 4"
        result = run_code(src)
        assert arr.match(result.payload, arr.S(3))

    def test_inline_assign(self):
        src = "a×a←2 5"
        result = run_code(src)
        assert arr.match(result.payload, arr.V([4, 25]))
    
class TestDfn:
    def test_inline_call(self):
        src = "1 {⍺+⍵} 2"
        result = run_code(src)
        assert arr.match(result.payload, arr.S(3))

    def test_nested(self):
        src = "1 {⍵ {⍺+⍵} ⍺} 2"
        result = run_code(src)
        assert arr.match(result.payload, arr.S(3))

    def test_gets(self):
        src = "A ← {⍺+⍵}"
        parser = Parser()
        ast = parser.parse(src)
        code = ast.emit()
        env = {}
        stack = Stack()
        run(code, env, 0, stack)
        assert 'A' in env
        assert env['A'].kind == TYPE.dfn

    def test_apply_fref(self):
        src = "Add←{⍺+⍵}⋄1 Add 2"
        result = run_code(src)
        assert arr.match(result.payload, arr.S(3))

    def test_dfn_operand_inline(self):
        src = '{⍺+⍵}/1 2 3 4'
        result = run_code(src)
        assert arr.match(result.payload, arr.S(10))
        
    def test_dfn_ref_operand(self):
        src = "Add←{⍺+⍵}⋄Add/1 2 3 4"
        result = run_code(src)
        assert arr.match(result.payload, arr.S(10))

    def test_early_return(self):
        src = "3 {a←⍺ ⋄ b←⍵>a ⋄ a+b ⋄ a-b ⋄ a×b ⋄ 2 2⍴a a a a} 7"
        result = run_code(src)
        assert arr.match(result.payload, arr.S(4))

    def test_dfn_index(self):
        src = "{1 2 3[⍵]}1"
        result = run_code(src)
        assert arr.match(result.payload, arr.S(2))

class TestOperator:
    def test_dfn_reduce(self):
        """
        {⍺+⍵}/1 2
        3
        """
        src = "{⍺+⍵}/1 2"
        result = run_code(src)
        assert arr.match(result.payload, arr.S(3))

    def test_each_primitive(self):
        src = '≢¨(1 2 3)(1 2)(1 2 3 4)'
        result = run_code(src)
        assert arr.match(result.payload, arr.V([3, 2, 4]))

    def test_each_dfn_inline(self):
        src = '{≢⍵}¨(1 2 3)(1 2)(1 2 3 4)'
        result = run_code(src)
        assert arr.match(result.payload, arr.V([3, 2, 4]))

    def test_each_dfn_ref(self):
        src = 'A←{≢⍵}⋄A¨(1 2 3)(1 2)(1 2 3 4)'
        result = run_code(src)
        assert arr.match(result.payload, arr.V([3, 2, 4]))

    def test_each_matrix(self):
        src = '{≢⍵}¨2 2⍴(1 2 3)(1 2)(1 2 3 4)(1 2 3)'
        result = run_code(src)
        assert arr.match(result.payload, arr.Array([2, 2], [3, 2, 4, 3]))

    def test_over_primitives(self):
        src = "1 2 3 ⌊⍥≢ 1 2 3 4"
        result = run_code(src)
        assert arr.match(result.payload, arr.S(3))

    def test_over_dfn_left(self):
        src = "1 2 3 {⍺⌊⍵}⍥≢ 1 2 3 4"
        result = run_code(src)
        assert arr.match(result.payload, arr.S(3))

    def test_over_dfn_left_right(self):
        src = "1 2 3 {⍺⌊⍵}⍥{≢⍵} 1 2 3 4"
        result = run_code(src)
        assert arr.match(result.payload, arr.S(3))

    def test_rank_vector_operand(self):
        src = "(2 2⍴1 2 3 4)(+⍤1 1)9 5"
        result = run_code(src)
        assert arr.match(result.payload, arr.Array([2, 2], [10, 7, 12, 9]))

class TestIndexing:
    def test_indexed_gets1(self):
        src = "a ← 1 2 3 4 ⋄ a[1] ← 99 ⋄ a"
        result = run_code(src)
        assert arr.match(result.payload, arr.V([1, 99, 3, 4]))

    def test_indexed_gets2(self):
        src = "a←1 2 3 4⋄a[2 2⍴1 2]←2 2⍴9 8 7 6⋄a"
        result = run_code(src)
        assert arr.match(result.payload, arr.V([1, 7, 6, 4]))

    def test_indexed_read1(self):
        src = "a←1 2 3 4⋄a[2 2⍴1 2]"
        result = run_code(src)
        assert arr.match(result.payload, arr.Array([2, 2], [2, 3, 2, 3]))

    def test_indexed_read_high_rank(self):
        src = "a←2 2⍴1 2 3 4⋄a[(1 0)(0 0)]"
        result = run_code(src)
        assert arr.match(result.payload, arr.V([3, 1]))

    def test_indexed_read_high_rank_enclose(self):
        src = "a←2 2⍴1 2 3 4⋄a[⊂1 0]"
        result = run_code(src)
        assert arr.match(result.payload, arr.S(3))

    def test_indexed_gets_high_rank(self):
        src = "a←2 2⍴1 2 3 4⋄a[(1 0)(0 0)]←9 8⋄a"
        result = run_code(src)
        assert arr.match(result.payload, arr.Array([2, 2], [8, 2, 9, 4]))

    def test_indexed_gets_high_rank_enclose(self):
        src = "a←2 2⍴1 2 3 4⋄a[⊂1 0]←9⋄a"
        result = run_code(src)
        assert arr.match(result.payload, arr.Array([2, 2], [1, 2, 9, 4]))

    def test_nested_without(self):
        src = "'ab' 'cd' 'ad'~⊂'cd'"
        result = run_code(src)
        assert arr.match(result.payload, arr.V([arr.V("ab"), arr.V("ad")]))

    def test_index_literal_vector(self):
        src = "1 2 3[1]"
        result = run_code(src)
        assert arr.match(result.payload, arr.S(2))

    def test_empty_index_vector(self):
        src = "1 2 3[⍬]"
        result = run_code(src)
        assert arr.match(result.payload, arr.Array([0], []))

class TestSystemArrays:
    def test_zilde(self):
        src = "⍬"
        result = run_code(src)
        assert arr.match(result.payload, arr.Array([0], []))

    def test_zilde2(self):
        src = "⍬≡0⍴0"
        result = run_code(src)
        assert arr.match(result.payload, arr.S(1))

    def test_zilde3(self):
        src = '(2 3⍴⍬)≡2 3⍴0'
        result = run_code(src)
        assert arr.match(result.payload, arr.S(1))

    def test_zilde4(self):
        src = '(⍴⍬)≡,0'
        result = run_code(src)
        assert arr.match(result.payload, arr.S(1))

    def test_d(self):
        src = "⎕D≡0 1 2 3 4 5 6 7 8 9"
        result = run_code(src)
        assert arr.match(result.payload, arr.S(1))

class TestCharVec:
    def test_empty(self):
        src = "(⍴'')≡⍴⍬"
        result = run_code(src)
        assert arr.match(result.payload, arr.S(1))


