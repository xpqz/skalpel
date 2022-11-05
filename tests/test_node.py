from apl.parser import Parser
from apl.skalpel import INSTR

def compile(src):
    parser = Parser()
    ast = parser.parse(src)
    code = ast.emit()
    return [line[0] for line in code]

class TestNode:
    def test_arith(self):
        instr = compile("1+2")
        assert instr == [INSTR.push, INSTR.push, INSTR.dya]

    def test_mop_deriving_monad(self):
        instr = compile("+⌿1 2 3 4 5")
        assert instr == [
            INSTR.push, INSTR.push, INSTR.push, INSTR.push, INSTR.push, INSTR.vec, INSTR.fun, INSTR.mon,
        ]
        
    def test_mop_deriving_dyad(self):
        instr = compile("1 +⍨ 2")
        assert instr == [
            INSTR.push, INSTR.push, INSTR.fun, INSTR.dya
        ]

    def test_gets(self):
        instr = compile("var←99")
        assert instr == [INSTR.push, INSTR.set]

    def test_diamond(self):
        instr = compile("v←⍳99 ⋄ s←+⌿v")
        assert instr == [
            INSTR.push, INSTR.mon, INSTR.set, INSTR.get, INSTR.fun, INSTR.mon, INSTR.set,
        ]

    def test_sys(self):
        instr = compile("⎕IO←0")
        assert instr == [INSTR.push, INSTR.set]

    def test_dop_deriving_dyad(self):
        instr = compile("1 2 3 ⌊⍥≢ 1 2 3 4")
        assert instr == [
            INSTR.push, 
            INSTR.push,
            INSTR.push,
            INSTR.push,
            INSTR.vec,
            INSTR.push,
            INSTR.push,
            INSTR.push,
            INSTR.vec,
            INSTR.fun,
            INSTR.fun,
            INSTR.dya
        ]

    def test_dfn_call(self):
        instr = compile('1 {⍺+⍵} 2')
        assert instr == [
            INSTR.push, INSTR.push, INSTR.dfn, INSTR.get, INSTR.get, INSTR.dya, INSTR.dya
        ]

    def test_dfn_operand_inline(self):
        instr = compile('{⍺+⍵}/1 2 3 4')
        assert instr == [
            INSTR.push, INSTR.push, INSTR.push, INSTR.push, INSTR.vec, INSTR.dfn, INSTR.get, INSTR.get, INSTR.dya, INSTR.mon
        ]
    

    def test_gets_dfn(self):
        instr = compile('A←{⍺+⍵}')
        assert instr == [
            INSTR.dfn, INSTR.get, INSTR.get, INSTR.dya, INSTR.set
        ]

    def test_nested_dfn(self):
        instr = compile('A←{⍵ {⍺+⍵} ⍺}')
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
            INSTR.dfn, INSTR.get, INSTR.get, INSTR.dya, INSTR.set, INSTR.push, INSTR.push, INSTR.dya
        ]
        assert code[7][1] == 'Add' # call by reference

    def test_dfn_ref_operand(self):
        instr = compile("A/1 2 3")
        assert instr == [
            INSTR.push, INSTR.push, INSTR.push, INSTR.vec, INSTR.fun, INSTR.mon
        ]    

    def test_dfn_ref_operand2(self):    
        instr = compile("Add←{⍺+⍵}⋄Add/⍳8")
        assert instr == [
            INSTR.dfn, INSTR.get, INSTR.get, INSTR.dya, INSTR.set, INSTR.push, INSTR.mon, INSTR.fun, INSTR.mon
        ]   
    
    def test_early_return(self):
        instr = compile("{a←⍺ ⋄ b←⍵>a ⋄ a+b ⋄ a-b ⋄ a×b ⋄ 2 2⍴a a a a}")
        assert instr == [
            INSTR.dfn, INSTR.get, INSTR.set, INSTR.get, INSTR.get, INSTR.dya, INSTR.set, INSTR.get, INSTR.get, INSTR.dya
        ] 

    def test_indexed_gets1(self):
        instr = compile("a[2]←9")
        assert instr == [
            INSTR.push,
            INSTR.push,
            INSTR.seti,
        ]

    def test_indexed_gets2(self):
        instr = compile("a[2 3 4]←9 8 7")
        assert instr == [
            INSTR.push,
            INSTR.push,
            INSTR.push,
            INSTR.vec,
            INSTR.push,
            INSTR.push,
            INSTR.push,
            INSTR.vec,
            INSTR.seti,
        ]

    def test_indexed_read1(self):
        instr = compile("a[2 3 4]")
        assert instr == [
            INSTR.push,
            INSTR.push,
            INSTR.push,
            INSTR.vec,
            INSTR.geti,
        ]

    def test_indexed_read_higher_rank(self):
        instr = compile("a←2 2⍴1 2 3 4⋄a[(1 0)(0 0)]")
        assert instr == [
            INSTR.push, 
            INSTR.push, 
            INSTR.push, 
            INSTR.push, 
            INSTR.vec, 
            INSTR.push, 
            INSTR.push, 
            INSTR.vec, 
            INSTR.dya, 
            INSTR.set, 
            INSTR.push, 
            INSTR.push, 
            INSTR.vec, 
            INSTR.push, 
            INSTR.push, 
            INSTR.vec, 
            INSTR.vec, 
            INSTR.geti
        ]

    def test_two_d_zilde(self):
        instr = compile('2 3⍴⍬')
        assert instr == [
            INSTR.get,
            INSTR.push,
            INSTR.push,
            INSTR.vec,
            INSTR.dya
        ]
