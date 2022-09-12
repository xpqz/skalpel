from apl.parser import Parser
from apl.skalpel import INSTR

class TestNode:
    def test_arith(self):
        src = "1+2"
        parser = Parser()
        ast = parser.parse(src)
        code = ast.emit()
        assert code[0][0] == INSTR.psh
        assert code[1][0] == INSTR.psh
        assert code[2][0] == INSTR.dya

    def test_mop_deriving_monad(self):
        src = "+⌿1 2 3 4 5"
        parser = Parser()
        ast = parser.parse(src)
        code = ast.emit()
        instr = [line[0] for line in code]
        assert instr == [
            INSTR.psh, INSTR.psh, INSTR.psh, INSTR.psh, INSTR.psh, INSTR.vec, INSTR.mon,
        ]
        
    def test_mop_deriving_dyad(self):
        src = "1 +⍨ 2"
        parser = Parser()
        ast = parser.parse(src)
        code = ast.emit()
        assert code[0][0] == INSTR.psh
        assert code[1][0] == INSTR.psh
        assert code[2][0] == INSTR.dya

    def test_gets(self):
        src = "var←99"
        parser = Parser()
        ast = parser.parse(src)
        code = ast.emit()
        assert code[0][0] == INSTR.psh
        assert code[1][0] == INSTR.set

    def test_diamond(self):
        src = "v←⍳99 ⋄ s←+⌿v"
        parser = Parser()
        ast = parser.parse(src)
        print(ast)
        code = ast.emit()
        instr = [line[0] for line in code]
        assert instr == [
            INSTR.psh, INSTR.mon, INSTR.set, INSTR.get, INSTR.mon, INSTR.set,
        ]

    def test_sys(self):
        src = "⎕IO←0"
        parser = Parser()
        ast = parser.parse(src)
        code = ast.emit()
        assert code[0][0] == INSTR.psh
        assert code[1][0] == INSTR.set

    def test_dop_deriving_dyad(self):
        src = "1 2 3 ⌊⍥≢ 1 2 3 4"
        parser = Parser()
        ast = parser.parse(src)
        code = ast.emit()
        instr = [line[0] for line in code]
        assert instr == [
            INSTR.psh, INSTR.psh, INSTR.psh, INSTR.vec, INSTR.psh, INSTR.psh, INSTR.psh, INSTR.psh, INSTR.vec, INSTR.dya,  
        ]

    def test_dfn_call(self):
        src = '1 {⍺+⍵} 2'
        parser = Parser()
        ast = parser.parse(src)
        code = ast.emit()
        instr = [line[0] for line in code]
        assert instr == [
            INSTR.psh, INSTR.psh, INSTR.get, INSTR.get, INSTR.dya, INSTR.dfn, INSTR.dya
        ]

    def test_gets_dfn(self):
        src = 'a←{⍺+⍵}'
        parser = Parser()
        ast = parser.parse(src)
        code = ast.emit()
        instr = [line[0] for line in code]
        assert instr == [
            INSTR.get, INSTR.get, INSTR.dya, INSTR.dfn, INSTR.set
        ]

    def test_nested_dfn(self):
        src = 'a←{⍵ {⍺+⍵} ⍺}'
        parser = Parser()
        ast = parser.parse(src)
        code = ast.emit()
        instr = [line[0] for line in code]
        assert instr == [
            INSTR.get, INSTR.get, INSTR.get, INSTR.get, INSTR.dya, INSTR.dfn, INSTR.dya, INSTR.dfn, INSTR.set
        ]
    
    def test_dfn_instr_count(self):
        src = '{-⍉2 2⍴⍺ ⍵}'
        parser = Parser()
        ast = parser.parse(src)
        code = ast.emit()
        assert code[-1][1] == len(code) - 1
    
    