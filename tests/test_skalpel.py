import apl.arr as arr
from apl.parser import Parser
from apl.skalpel import each, pervade, reduce_first, run, TYPE
from apl.stack import Stack

def run_code(src):
    parser = Parser()
    ast = parser.parse(src)
    code = ast.emit()
    env = {}
    stack = Stack()
    run(code, env, 0, stack)
    return stack.stack[0]

class TestPervade:
    def test_mat_plus_scalar(self):
        plus = pervade(lambda x, y:x+y)
        a = arr.Array([3, 2], [54, 7, 88, 4, 73, 3, 1])
        b = arr.S(12)
        result = plus(a, b)
        assert arr.match(result, arr.Array([3, 2], [66, 19, 100, 16, 85, 15]))

    def test_scalar_plus_mat(self):
        plus = pervade(lambda x, y:x+y)
        a = arr.Array([3, 2], [54, 7, 88, 4, 73, 3, 1])
        b = arr.S(12)
        result = plus(b, a)
        assert arr.match(result, arr.Array([3, 2], [66, 19, 100, 16, 85, 15]))

    def test_vec_plus_vec(self):
        plus = pervade(lambda x, y:x+y)
        a = arr.V([54, 7, 88, 4, 73, 3, 1])
        b = arr.V([12, 8, 11, 7, 21, 7, 9])
        result = plus(b, a)
        assert arr.match(result, arr.V([66, 15, 99, 11, 94, 10, 10]))

    def test_non_simple(self):
        plus = pervade(lambda x, y:x+y)
        a = arr.Array([3, 2], [54, 7, arr.Array([2, 2], [1, 1, 1, 1]), 4, 73, 3])
        b = arr.S(12)
        result = plus(b, a)
        expected = arr.Array([3, 2], [66, 19, arr.Array([2, 2], [13, 13, 13, 13]), 16, 85, 15])
        assert arr.match(result, expected)

class TestReduce:
    def test_reduce_first(self):
        r = reduce_first('+', None, None, arr.Array([2, 2], [1, 2, 3, 4]), None, None)
        assert arr.match(r, arr.V([4, 6]))

class TestBitwise:
    def test_or(self):
        bwo = pervade(lambda x, y:x|y)
        a = arr.V([1, 0, 1, 0])
        b = arr.V([0, 1, 0, 1])
        c = bwo(a, b)
        assert arr.match(c, arr.V([1, 1, 1, 1]))

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
        src = "v←⍳99 ⋄ s←+⌿v"
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

class TestOperator:
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


