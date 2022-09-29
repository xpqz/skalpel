import pytest 
from apl.parser import Parser

def parse(src):
    parser = Parser()
    ast = parser.parse(src)

    if ast is None:
        return None

    return str(ast)

class TestParser:
    def test_comment_only(self):
        assert parse(" ⍝ a comment ") is None

    def test_empty(self):
        assert parse("") is None

    def test_whitespace(self):
        assert parse("         ") is None

    def test_parser_arith(self):
        assert 'CHNK[DYADIC(FUN(+), SCALAR(1), SCALAR(2))]' == parse('1+2')

    def test_mop_deriving_monad(self):
        assert parse("+⌿1 2 3 4 5") == "CHNK[MONADIC(MOP('⌿', FUN(+)), VEC[SCALAR(1), SCALAR(2), SCALAR(3), SCALAR(4), SCALAR(5)])]"

    def test_mop_deriving_monad2(self):
        assert parse("+/1 2 3 4 5") == "CHNK[MONADIC(MOP('/', FUN(+)), VEC[SCALAR(1), SCALAR(2), SCALAR(3), SCALAR(4), SCALAR(5)])]"

    def test_mop_deriving_dyad(self):
        assert parse("1 +⍨ 2") == "CHNK[DYADIC(MOP('⍨', FUN(+)), SCALAR(1), SCALAR(2))]"

    def test_gets(self):
        assert parse("var←99") == "CHNK[GETS(ID('var'), SCALAR(99))]"

    def test_diamond(self):
        assert parse("v←⍳99 ⋄ s←+⌿v") == "CHNK[GETS(ID('v'), MONADIC(FUN(⍳), SCALAR(99))), GETS(ID('s'), MONADIC(MOP('⌿', FUN(+)), ID('v')))]"

    def test_sys(self):
        assert parse("⎕IO←0") == "CHNK[GETS(ID('⎕IO'), SCALAR(0))]"

    def test_dop_deriving_dyad(self):
        assert parse("1 2 3 ⌊⍥≢ 1 2 3 4") == "CHNK[DYADIC(DOP('⍥', FUN(⌊), FUN(≢)), VEC[SCALAR(1), SCALAR(2), SCALAR(3)], VEC[SCALAR(1), SCALAR(2), SCALAR(3), SCALAR(4)])]"
    
    def test_gets_gets(self):
        assert parse("a ← -b ← 3") == "CHNK[GETS(ID('a'), MONADIC(FUN(-), GETS(ID('b'), SCALAR(3))))]"
    
    def test_dfn(self):
        assert parse("{⍺+⍵}") == 'CHNK[DFN[DYADIC(FUN(+), ARG(⍺), ARG(⍵))]]'

    def test_gets_dfn_raises(self):
        src = 'a←{⍺+⍵}'
        parser = Parser()
        with pytest.raises(SyntaxError):
            parser.parse(src)

    def test_gets_dfn(self):
        assert parse("Add←{⍺+⍵}") == 'CHNK[GETS(FREF(Add), DFN[DYADIC(FUN(+), ARG(⍺), ARG(⍵))])]'

    def test_gets_dfn_call(self):
        assert parse("a←3 {⍺+⍵} 1 2 3 4") == "CHNK[GETS(ID('a'), DYADIC(DFN[DYADIC(FUN(+), ARG(⍺), ARG(⍵))], SCALAR(3), VEC[SCALAR(1), SCALAR(2), SCALAR(3), SCALAR(4)]))]"

    def test_gets_dfn_call2(self):
        assert parse("1 {⍺+⍵} 2") == 'CHNK[DYADIC(DFN[DYADIC(FUN(+), ARG(⍺), ARG(⍵))], SCALAR(1), SCALAR(2))]'

    def test_strand_or_call1(self):
        assert parse("1 A 2") == 'CHNK[DYADIC(FREF(A), SCALAR(1), SCALAR(2))]'

    def test_strand_or_call2(self):
        assert parse("1 a 2") == "CHNK[VEC[SCALAR(1), ID('a'), SCALAR(2)]]"

    def test_dfn_inline_operand(self):
        assert parse("{⍺+⍵}/⍳8") == "CHNK[MONADIC(MOP('/', DFN[DYADIC(FUN(+), ARG(⍺), ARG(⍵))]), MONADIC(FUN(⍳), SCALAR(8)))]"

    def test_dfn_ref_operand(self):
        assert parse("Add←{⍺+⍵}⋄Add⌿⍳8") == "CHNK[GETS(FREF(Add), DFN[DYADIC(FUN(+), ARG(⍺), ARG(⍵))]), MONADIC(MOP('⌿', FREF(Add)), MONADIC(FUN(⍳), SCALAR(8)))]"

class TestBracketIndexing:
    def test_bracket_index1(self):
        assert parse("a[2]←5") == "CHNK[GETS(IDX(ID('a'), SCALAR(2)), SCALAR(5))]"

    def test_bracket_index2(self):
        assert parse("a[2 2 3]←5 8 7") == "CHNK[GETS(IDX(ID('a'), VEC[SCALAR(2), SCALAR(2), SCALAR(3)]), VEC[SCALAR(5), SCALAR(8), SCALAR(7)])]"

    def test_bracket_index_read_name(self):
        d = parse("a[2 3]")
        assert d == "CHNK[IDX(ID('a'), VEC[SCALAR(2), SCALAR(3)])]"

    def test_bracket_index_read_vector_literal(self):
        d = parse("1 2 3 4[2]")
        assert d == 'CHNK[IDX(VEC[SCALAR(1), SCALAR(2), SCALAR(3), SCALAR(4)], SCALAR(2))]'

    def test_bracket_index_paren(self):
        d = parse("(1 2 3 4)[2]")
        assert d == 'CHNK[IDX(VEC[SCALAR(1), SCALAR(2), SCALAR(3), SCALAR(4)], SCALAR(2))]'

    def test_bracket_index_stranding(self):
        d = parse("a[1] b[2] c[3]")
        assert d == "CHNK[VEC[IDX(ID('a'), SCALAR(1)), IDX(ID('b'), SCALAR(2)), IDX(ID('c'), SCALAR(3))]]"

    def test_bracket_index_stranding2(self):
        d = parse("(a[1] b[2] c)[1]")
        assert d == "CHNK[IDX(VEC[IDX(ID('a'), SCALAR(1)), IDX(ID('b'), SCALAR(2)), ID('c')], SCALAR(1))]"

    def test_bracket_index_higher_rank(self):
        d = parse("a←2 2⍴1 2 3 4⋄a[(1 0)(0 0)]")
        assert d == "CHNK[GETS(ID('a'), DYADIC(FUN(⍴), VEC[SCALAR(2), SCALAR(2)], VEC[SCALAR(1), SCALAR(2), SCALAR(3), SCALAR(4)])), IDX(ID('a'), VEC[VEC[SCALAR(1), SCALAR(0)], VEC[SCALAR(0), SCALAR(0)]])]"