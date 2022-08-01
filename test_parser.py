from aplparse import Parser
# import inspect

class TestAPLParser:
    def test_parser_arith(self):
        src = "1+2"
        parser = Parser()
        ast = parser.parse(src)
        print(ast)

    def test_op(self):
        src = "×⍨ 4.5 - (4 ¯3 5.6)"
        parser = Parser()
        ast = parser.parse(src)
        print(ast)

    def test_monadic_op_deriving_dyad(self):
        src = "1 +⍨ 2"
        parser = Parser()
        ast = parser.parse(src)
        print(ast)

    def test_gets(self):
        src = "var←99"
        parser = Parser()
        ast = parser.parse(src)
        print(ast)
    
    