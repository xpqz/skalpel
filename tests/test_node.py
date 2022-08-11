from apl.node import Node
from apl.parser import Parser
from apl.skalpel import CMD

class TestNode:
    def test_arith(self):
        src = "1+2"
        parser = Parser()
        ast = parser.parse(src)
        code = ast.emit()
        assert code[0][0] == CMD.push
        assert code[1][0] == CMD.push
        assert code[2][0] == CMD.call

    def test_mop_deriving_monad(self):
        src = "+⌿1 2 3 4 5"
        parser = Parser()
        ast = parser.parse(src)
        code = ast.emit()
        instr = [line[0] for line in code]
        assert instr == [
            CMD.push, CMD.push, CMD.push, CMD.push, CMD.push, CMD.vec, CMD.call,
        ]
        
    def test_mop_deriving_dyad(self):
        src = "1 +⍨ 2"
        parser = Parser()
        ast = parser.parse(src)
        code = ast.emit()
        assert code[0][0] == CMD.push
        assert code[1][0] == CMD.push
        assert code[2][0] == CMD.call

    def test_gets(self):
        src = "var←99"
        parser = Parser()
        ast = parser.parse(src)
        code = ast.emit()
        assert code[0][0] == CMD.push
        assert code[1][0] == CMD.set

    def test_diamond(self):
        src = "v←⍳99 ⋄ s←+⌿v"
        parser = Parser()
        ast = parser.parse(src)
        print(ast)
        code = ast.emit()
        instr = [line[0] for line in code]
        assert instr == [
            CMD.push, CMD.call, CMD.set, CMD.get, CMD.call, CMD.set,
        ]

    def test_sys(self):
        src = "⎕IO←0"
        parser = Parser()
        ast = parser.parse(src)
        code = ast.emit()
        assert code[0][0] == CMD.push
        assert code[1][0] == CMD.set

    def test_dop_deriving_dyad(self):
        src = "1 2 3 ⌊⍥≢ 1 2 3 4"
        parser = Parser()
        ast = parser.parse(src)
        code = ast.emit()
        instr = [line[0] for line in code]
        assert instr == [
            CMD.push, CMD.push, CMD.push, CMD.vec, CMD.push, CMD.push, CMD.push, CMD.push, CMD.vec, CMD.call,  
        ]
    
    
    