from apl.parser import Parser

class TestParser:
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
        assert "CHNK[GETS(ID('s'), MONADIC(MOP('⌿', FUN(+)), ID('v'))), GETS(ID('v'), MONADIC(FUN(⍳), SCALAR(99)))]" == str(ast)

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
    
    
    