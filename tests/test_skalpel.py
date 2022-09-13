from apl.arr import Array, S, match
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
        src = "a ← {⍺+⍵}"
        parser = Parser()
        ast = parser.parse(src)
        code = ast.emit()
        env = {}
        stack = Stack()
        run(code, env, 0, stack)
        assert 'a' in env
        assert env['a'].kind == TYPE.dfn
