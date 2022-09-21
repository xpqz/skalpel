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
            INSTR.psh, INSTR.psh, INSTR.psh, INSTR.psh, INSTR.psh, INSTR.vec, INSTR.psh, INSTR.mon,
        ]
        
    def test_mop_deriving_dyad(self):
        src = "1 +⍨ 2"
        parser = Parser()
        ast = parser.parse(src)
        code = ast.emit()
        assert code[0][0] == INSTR.psh
        assert code[1][0] == INSTR.psh
        assert code[2][0] == INSTR.psh
        assert code[3][0] == INSTR.dya

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
            INSTR.psh, INSTR.mon, INSTR.set, INSTR.get, INSTR.psh, INSTR.mon, INSTR.set,
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
            INSTR.psh, INSTR.psh, INSTR.psh, INSTR.vec, INSTR.psh, INSTR.psh, INSTR.psh, INSTR.psh, INSTR.vec, INSTR.psh, INSTR.psh, INSTR.dya,  
        ]

    def test_dfn_call(self):
        src = '1 {⍺+⍵} 2'
        parser = Parser()
        ast = parser.parse(src)
        code = ast.emit()
        instr = [line[0] for line in code]
        assert instr == [
            INSTR.psh, INSTR.psh, INSTR.dfn, INSTR.get, INSTR.get, INSTR.dya, INSTR.dya
        ]

    def test_dfn_operand_inline(self):
        src = '{⍺+⍵}/1 2 3 4'
        parser = Parser()
        ast = parser.parse(src)
        code = ast.emit()
        instr = [line[0] for line in code]
        assert instr == [
            INSTR.psh, INSTR.psh, INSTR.psh, INSTR.psh, INSTR.vec, INSTR.dfn, INSTR.get, INSTR.get, INSTR.dya, INSTR.mon
        ]
    

    def test_gets_dfn(self):
        src = 'A←{⍺+⍵}'
        parser = Parser()
        ast = parser.parse(src)
        code = ast.emit()
        instr = [line[0] for line in code]
        assert instr == [
            INSTR.dfn, INSTR.get, INSTR.get, INSTR.dya, INSTR.set
        ]

    def test_nested_dfn(self):
        src = 'A←{⍵ {⍺+⍵} ⍺}'
        parser = Parser()
        ast = parser.parse(src)
        code = ast.emit()
        instr = [line[0] for line in code]
        assert instr == [
            INSTR.dfn, INSTR.get, INSTR.get, INSTR.dfn, INSTR.get, INSTR.get, INSTR.dya, INSTR.dya, INSTR.set
        ]
    
    def test_dfn_instr_count(self):
        src = '{-⍉2 2⍴⍺ ⍵}'
        parser = Parser()
        ast = parser.parse(src)
        code = ast.emit()
        assert code[0][1] == len(code) - 1

    def test_apply_fref(self):
        src = "Add←{⍺+⍵}⋄1 Add 2"
        parser = Parser()
        ast = parser.parse(src)
        code = ast.emit()
        instr = [line[0] for line in code]
        assert instr == [
            INSTR.dfn, INSTR.get, INSTR.get, INSTR.dya, INSTR.set, INSTR.psh, INSTR.psh, INSTR.dya
        ]
        assert code[7][1] == 'Add' # call by reference

    def test_dfn_ref_operand(self):
        src = "A/1 2 3"
        parser = Parser()
        ast = parser.parse(src)
        code = ast.emit()
        instr = [line[0] for line in code]
        assert instr == [
            INSTR.psh, INSTR.psh, INSTR.psh, INSTR.vec, INSTR.psh, INSTR.mon
        ]    

    def test_dfn_ref_operand2(self):    
        src = "Add←{⍺+⍵}⋄Add/⍳8"
        parser = Parser()
        ast = parser.parse(src)
        code = ast.emit()
        instr = [line[0] for line in code]
        assert instr == [
            INSTR.dfn, INSTR.get, INSTR.get, INSTR.dya, INSTR.set, INSTR.psh, INSTR.mon, INSTR.psh, INSTR.mon
        ]   
    