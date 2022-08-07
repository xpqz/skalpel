from apl.tokeniser import Tokeniser, TokenType

class TestTokeniser:
    def test_system_variable(self):
        src = "⎕IO←0"
        tokeniser = Tokeniser(src)
        tokens = tokeniser.lex()

        assert len(tokens) == 4
        assert tokens[1].kind == TokenType.NAME

    def test_numeric_vector(self):
        src = "1 2 3"
        tokeniser = Tokeniser(src)
        tokens = tokeniser.lex()

        assert len(tokens) == 4
        types = [t.kind for t in tokens]

        for i in types:
            assert i in [TokenType.SCALAR, TokenType.EOF]
    
    def test_character_vector(self):
        src = "'hello'"
        tokeniser = Tokeniser(src)
        tokens = tokeniser.lex()

        assert len(tokens) == 8
        types = [t.kind for t in tokens]
        assert types[1] == types[-1] == TokenType.SINGLEQUOTE
        
    def test_numeric_scalar(self):
        src = "7"
        tokeniser = Tokeniser(src)
        tokens = tokeniser.lex()

        assert len(tokens) == 2
        assert tokens[1].kind == TokenType.SCALAR
        assert tokens[1].tok == 7

    def test_character_scalar(self):
        src = "'h'"
        tokeniser = Tokeniser(src)
        tokens = tokeniser.lex()

        assert len(tokens) == 4
        assert tokens[2].kind == TokenType.SCALAR
        assert tokens[2].tok == 'h'

    def test_code(self):
        src = "var ← 1 2 3 4 5 6 7 8 9 ⋄ 3 3⍴var"
        tokeniser = Tokeniser(src)
        tokens = tokeniser.lex()
        assert tokens[0].kind == TokenType.EOF

        assert tokens[1].kind == TokenType.NAME
        assert tokens[1].tok == 'var'

        assert tokens[2].kind == TokenType.GETS
        assert tokens[2].tok == '←'

        assert tokens[3].kind == TokenType.SCALAR
        assert tokens[3].tok == 1

        assert tokens[12].kind == TokenType.DIAMOND
        assert tokens[12].tok == '⋄'

        assert tokens[13].kind == TokenType.SCALAR
        assert tokens[13].tok == 3

        assert tokens[15].kind == TokenType.FUN
        assert tokens[15].tok == '⍴'

        assert tokens[16].kind == TokenType.NAME
        assert tokens[16].tok == 'var'

    def test_code2(self):
        src = "×⍨ 4.5 - (4 ¯3 5.6)"
        tokeniser = Tokeniser(src)
        tokens = tokeniser.lex()

