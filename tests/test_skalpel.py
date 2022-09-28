from apl.arr import Array, Aflat, S, V, match
from apl.parser import Parser
from apl.skalpel import run, TYPE
from apl.stack import Stack

class TestRun:
    def test_arith(self):
        src = "1+2"
        parser = Parser()
        ast = parser.parse(src)
        code = ast.emit()
        env = {}
        stack = Stack()
        run(code, env, 0, stack)
        result = stack.stack[0]
        assert match(result.payload, S(3))

    def test_mop_deriving_monad(self):
        src = "+⌿1 2 3 4 5"
        parser = Parser()
        ast = parser.parse(src)
        code = ast.emit()
        env = {}
        stack = Stack()
        run(code, env, 0, stack)
        result = stack.stack[0]
        assert match(result.payload, S(15))
        
    def test_mop_deriving_dyad(self):
        src = "1 +⍨ 2"
        parser = Parser()
        ast = parser.parse(src)
        code = ast.emit()
        env = {}
        stack = Stack()
        run(code, env, 0, stack)
        result = stack.stack[0]
        assert match(result.payload, S(3))

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
        assert isinstance(s, Array)
        assert s.shape == []
        assert s.data[0] == 4851

    def test_dop_deriving_dyad(self):
        src = "1 2 3 ⌊⍥≢ 1 2 3 4"
        parser = Parser()
        ast = parser.parse(src)
        code = ast.emit()
        env = {}
        stack = Stack()
        run(code, env, 0, stack)
        result = stack.stack[0]
        assert match(result.payload, S(3))
    
class TestDfn:
    def test_inline_call(self):
        src = "1 {⍺+⍵} 2"
        parser = Parser()
        ast = parser.parse(src)
        code = ast.emit()
        env = {}
        stack = Stack()
        run(code, env, 0, stack)
        result = stack.stack[0]
        assert match(result.payload, S(3))

    def test_nested(self):
        src = "1 {⍵ {⍺+⍵} ⍺} 2"
        parser = Parser()
        ast = parser.parse(src)
        code = ast.emit()
        env = {}
        stack = Stack()
        run(code, env, 0, stack)
        result = stack.stack[0]
        assert match(result.payload, S(3))

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
        parser = Parser()
        ast = parser.parse(src)
        code = ast.emit()
        env = {}
        stack = Stack()
        run(code, env, 0, stack)
        result = stack.stack[0]
        assert match(result.payload, S(3))

    def test_dfn_operand_inline(self):
        src = '{⍺+⍵}/1 2 3 4'
        parser = Parser()
        ast = parser.parse(src)
        code = ast.emit()
        env = {}
        stack = Stack()
        run(code, env, 0, stack)
        result = stack.stack[0]
        assert match(result.payload, S(10))
        
    def test_dfn_ref_operand(self):
        src = "Add←{⍺+⍵}⋄Add/1 2 3 4"
        parser = Parser()
        ast = parser.parse(src)
        code = ast.emit()
        env = {}
        stack = Stack()
        run(code, env, 0, stack)
        result = stack.stack[0]
        assert match(result.payload, S(10))

    def test_early_return(self):
        src = "3 {a←⍺ ⋄ b←⍵>a ⋄ a+b ⋄ a-b ⋄ a×b ⋄ 2 2⍴a a a a} 7"
        parser = Parser()
        ast = parser.parse(src)
        code = ast.emit()
        env = {}
        stack = Stack()
        run(code, env, 0, stack)
        result = stack.stack[0]
        assert match(result.payload, S(4))

class TestOperator:
    def test_each_primitive(self):
        src = '≢¨(1 2 3)(1 2)(1 2 3 4)'
        parser = Parser()
        ast = parser.parse(src)
        code = ast.emit()
        env = {}
        stack = Stack()
        run(code, env, 0, stack)
        result = stack.stack[0]
        assert match(result.payload, V([3, 2, 4]))

    def test_each_dfn_inline(self):
        src = '{≢⍵}¨(1 2 3)(1 2)(1 2 3 4)'
        parser = Parser()
        ast = parser.parse(src)
        code = ast.emit()
        env = {}
        stack = Stack()
        run(code, env, 0, stack)
        result = stack.stack[0]
        assert match(result.payload, V([3, 2, 4]))

    def test_each_dfn_ref(self):
        src = 'A←{≢⍵}⋄A¨(1 2 3)(1 2)(1 2 3 4)'
        parser = Parser()
        ast = parser.parse(src)
        code = ast.emit()
        env = {}
        stack = Stack()
        run(code, env, 0, stack)
        result = stack.stack[0]
        assert match(result.payload, V([3, 2, 4]))

    def test_each_matrix(self):
        src = '{≢⍵}¨2 2⍴(1 2 3)(1 2)(1 2 3 4)(1 2 3)'
        parser = Parser()
        ast = parser.parse(src)
        code = ast.emit()
        env = {}
        stack = Stack()
        run(code, env, 0, stack)
        result = stack.stack[0]
        assert match(result.payload, Aflat([2, 2], [3, 2, 4, 3]))

    def test_over_primitives(self):
        src = "1 2 3 ⌊⍥≢ 1 2 3 4"
        parser = Parser()
        ast = parser.parse(src)
        code = ast.emit()
        env = {}
        stack = Stack()
        run(code, env, 0, stack)
        result = stack.stack[0]
        assert match(result.payload, S(3))

    def test_over_dfn_left(self):
        src = "1 2 3 {⍺⌊⍵}⍥≢ 1 2 3 4"
        parser = Parser()
        ast = parser.parse(src)
        code = ast.emit()
        env = {}
        stack = Stack()
        run(code, env, 0, stack)
        result = stack.stack[0]
        assert match(result.payload, S(3))

    def test_over_dfn_left_right(self):
        src = "1 2 3 {⍺⌊⍵}⍥{≢⍵} 1 2 3 4"
        parser = Parser()
        ast = parser.parse(src)
        code = ast.emit()
        env = {}
        stack = Stack()
        run(code, env, 0, stack)
        result = stack.stack[0]
        assert match(result.payload, S(3))

class TestIndexing:
    def test_indexed_gets1(self):
        src = "a ← 1 2 3 4 ⋄ a[1] ← 99 ⋄ a"
        parser = Parser()
        ast = parser.parse(src)
        code = ast.emit()
        env = {}
        stack = Stack()
        run(code, env, 0, stack)
        result = stack.stack[0]
        assert match(result.payload, V([1, 99, 3, 4]))

    def test_indexed_gets2(self):
        src = "a←1 2 3 4⋄a[2 2⍴1 2]←2 2⍴9 8 7 6⋄a"
        parser = Parser()
        ast = parser.parse(src)
        code = ast.emit()
        env = {}
        stack = Stack()
        run(code, env, 0, stack)
        result = stack.stack[0]
        assert match(result.payload, V([1, 7, 6, 4]))

    def test_indexed_read1(self):
        src = "a←1 2 3 4⋄a[2 2⍴1 2]"
        parser = Parser()
        ast = parser.parse(src)
        code = ast.emit()
        env = {}
        stack = Stack()
        run(code, env, 0, stack)
        result = stack.stack[0]
        assert match(result.payload, Aflat([2, 2], [2, 3, 2, 3]))