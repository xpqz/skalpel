from apl.parser import Parser

class TestParser:
    def test_comment_only(self):
        src = " ⍝ a comment "
        parser = Parser()
        ast = parser.parse(src)
        assert ast is None

    def test_empty(self):
        src = ""
        parser = Parser()
        ast = parser.parse(src)
        assert ast is None

    def test_whitespace(self):
        src = "         "
        parser = Parser()
        ast = parser.parse(src)
        assert ast is None

    def test_parser_arith(self):
        src = "1+2"
        parser = Parser()
        ast = parser.parse(src)
        assert 'CHNK[DYADIC(FUN(+), SCALAR(1), SCALAR(2))]' == str(ast)

    def test_mop_deriving_monad(self):
        src = "+⌿1 2 3 4 5"
        parser = Parser()
        ast = parser.parse(src)
        assert "CHNK[MONADIC(MOP('⌿', FUN(+)), VEC[SCALAR(1), SCALAR(2), SCALAR(3), SCALAR(4), SCALAR(5)])]" == str(ast)

    def test_mop_deriving_monad2(self):
        src = "+/1 2 3 4 5"
        parser = Parser()
        ast = parser.parse(src)
        assert "CHNK[MONADIC(MOP('/', FUN(+)), VEC[SCALAR(1), SCALAR(2), SCALAR(3), SCALAR(4), SCALAR(5)])]" == str(ast)

    def test_mop_deriving_dyad(self):
        src = "1 +⍨ 2"
        parser = Parser()
        ast = parser.parse(src)
        assert "CHNK[DYADIC(MOP('⍨', FUN(+)), SCALAR(1), SCALAR(2))]" == str(ast)

    def test_gets(self):
        src = "var←99"
        parser = Parser()
        ast = parser.parse(src)
        assert "CHNK[GETS(ID('var'), SCALAR(99))]" == str(ast)

    def test_diamond(self):
        src = "v←⍳99 ⋄ s←+⌿v"
        parser = Parser()
        ast = parser.parse(src)
        assert "CHNK[GETS(ID('v'), MONADIC(FUN(⍳), SCALAR(99))), GETS(ID('s'), MONADIC(MOP('⌿', FUN(+)), ID('v')))]" == str(ast)

    def test_sys(self):
        src = "⎕IO←0"
        parser = Parser()
        ast = parser.parse(src)
        assert "CHNK[GETS(ID('⎕IO'), SCALAR(0))]" == str(ast)

    def test_dop_deriving_dyad(self):
        src = "1 2 3 ⌊⍥≢ 1 2 3 4"
        parser = Parser()
        ast = parser.parse(src)
        assert "CHNK[DYADIC(DOP('⍥', FUN(⌊), FUN(≢)), VEC[SCALAR(1), SCALAR(2), SCALAR(3)], VEC[SCALAR(1), SCALAR(2), SCALAR(3), SCALAR(4)])]" == str(ast)
    
    def test_gets_gets(self):
        src = "a ← -b ← 3"
        parser = Parser()
        ast = parser.parse(src)
        assert "CHNK[GETS(ID('a'), MONADIC(FUN(-), GETS(ID('b'), SCALAR(3))))]" == str(ast)
    
    def test_dfn(self):
        src = "{⍺+⍵}"
        parser = Parser()
        ast = parser.parse(src)
        assert 'CHNK[DFN[DYADIC(FUN(+), ARG(⍺), ARG(⍵))]]' == str(ast)

    def test_gets_dfn(self):
        src = "a←{⍺+⍵}"
        parser = Parser()
        ast = parser.parse(src)
        assert "CHNK[GETS(ID('a'), DFN[DYADIC(FUN(+), ARG(⍺), ARG(⍵))])]" == str(ast)

    def test_gets_dfn_call(self):
        src = "a←3 {⍺+⍵} 1 2 3 4"
        parser = Parser()
        ast = parser.parse(src)
        assert "CHNK[GETS(ID('a'), DYADIC(DFN[DYADIC(FUN(+), ARG(⍺), ARG(⍵))], SCALAR(3), VEC[SCALAR(1), SCALAR(2), SCALAR(3), SCALAR(4)]))]" == str(ast)

    def test_gets_dfn_call2(self):
        src = "1 {⍺+⍵} 2"
        parser = Parser()
        ast = parser.parse(src)
        assert 'CHNK[DYADIC(DFN[DYADIC(FUN(+), ARG(⍺), ARG(⍵))], SCALAR(1), SCALAR(2))]' == str(ast)
