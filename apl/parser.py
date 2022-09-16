from typing import Optional
from apl.errors import UnexpectedToken
from apl.node import Node, NodeType
from apl.tokeniser import Token, Tokeniser, TokenType

SCALARS = [TokenType.SCALAR, TokenType.NAME, TokenType.ALPHA, TokenType.OMEGA]
DYADIC_OPS = set('⍥@⍣⍤∘.⌺⍠') # FIXME: this should use Voc!!
MONADIC_OPS = set('\\/⌿⍀¨⍨')

class Parser:
    """
    Helpful resources for grammar experiments:

    https://mdkrajnak.github.io/ebnftest/
    https://www.bottlecaps.de/rr/ui

    chunk      ::= EOF statements
    statements ::= (statement DIAMOND)* statement
    statement  ::= ( ID GETS | vector function | function )* vector?
    function   ::= function MOP | function DOP f | f
    f          ::= FUN | LPAREN function RPAREN | dfn | fref
    dfn        ::= LBRACE statements RBRACE
    vector     ::= vector* ( scalar | ( LPAREN statement RPAREN ) )
    scalar     ::= INTEGER | FLOAT | ID | ALPHA | OMEGA
    """

    def __init__(self):
        self.source = ''
        self.tokens = []
        self.current_token = -1

    def token(self) -> Token:
        return self.tokens[self.current_token]  

    def peek(self) -> Token:
        return self.tokens[self.current_token-1]

    def peek_beyond(self, kinds: list[TokenType]) -> Token:
        pos = self.current_token-1
        while pos>0 and self.tokens[pos].kind in kinds:
            pos -= 1
        return self.tokens[pos]

    def eat_token(self) -> Token:
        tok = self.token()
        self.current_token -= 1
        return tok

    def expect_token(self, toktype: TokenType) -> Token:
        tok = self.eat_token()
        if tok.kind != toktype:
            raise UnexpectedToken(f"SYNTAX ERROR: expected {toktype}, found {tok.kind}")
        return tok

    def parse(self, chunk: str) -> Optional[Node]:
        self.source = chunk
        self.tokens = Tokeniser(self.source).lex()
        self.current_token = len(self.tokens) - 1
        if self.current_token == 0: # Just whitespace or comment
            return None
        return self.parse_chunk()

    def parse_chunk(self) -> Node:
        chunk = Node(NodeType.CHUNK, None, self.parse_statements())
        self.expect_token(TokenType.EOF)
        return chunk

    def parse_statements(self) -> list[Node]:
        statements = [self.parse_statement()]
        while self.token().kind == TokenType.DIAMOND:
            self.eat_token()
            statements.append(self.parse_statement())
        return statements[::-1] # NOTE: statements separated by diamond should be interpreted top to bottom

    def parse_statement(self) -> Node:
        """
        statement  ::= ( ID GETS | vector function | function )* vector?
        """
        if self.token().kind in SCALARS + [TokenType.RPAREN]:
            node = self.parse_vector()
        else:
            node = None

        while (kind := self.token().kind) in [TokenType.FUN, TokenType.OPERATOR, TokenType.GETS, TokenType.RBRACE, TokenType.FNAME]:
            if kind == TokenType.GETS:
                gets = self.expect_token(TokenType.GETS)
                if node is None: 
                    raise SyntaxError
                name = self.parse_identifier()
                if node.kind in {NodeType.DFN, NodeType.FUN} and name.kind != NodeType.FREF:
                    raise SyntaxError(f"SYNTAX ERROR: named functions must start with a capital letter: '{name.main_token.tok}'")
                node = Node(NodeType.GETS, gets, [name, node])
            else:
                fun = self.parse_function()
                if node is None:
                    node = fun
                else:
                    if self.token().kind in SCALARS+[TokenType.RPAREN]:
                        node = Node(NodeType.DYADIC, None, [fun, self.parse_vector(), node]) # Dyadic application of fun
                    else:
                        node = Node(NodeType.MONADIC, None, [fun, node])                    # Monadic application of fun
        return node  # type: ignore

    def parse_identifier(self) -> Node:
        tok = self.eat_token()
        if tok.kind == TokenType.NAME:
            return Node(NodeType.ID, tok)

        if tok.kind == TokenType.FNAME:
            return Node(NodeType.FREF, tok)

        raise SyntaxError(f"SYNTAX ERROR: expected a name, got {tok.tok}")

    def parse_function(self) -> Node:
        """
        function ::= function MOP | function DOP f | f
        """
        if self.token().tok in MONADIC_OPS:
            op = self.expect_token(TokenType.OPERATOR)
            return Node(NodeType.MOP, op, [self.parse_function()])
        else:
            fun = self.parse_f()
            if self.token().tok in DYADIC_OPS:
                op = self.expect_token(TokenType.OPERATOR)
                return Node(NodeType.DOP, op, [self.parse_function(), fun])
            return fun

    def parse_f(self) -> Node:
        """
        f ::= FUN | LPAREN function RPAREN | dfn | fref
        """
        tok = self.token().kind

        if tok == TokenType.FUN:
            return Node(NodeType.FUN, self.expect_token(TokenType.FUN))

        if tok == TokenType.RPAREN:
            self.expect_token(TokenType.RPAREN)
            fun = self.parse_function()
            self.expect_token(TokenType.LPAREN)
            return fun

        if tok == TokenType.RBRACE:
            return self.parse_dfn()

        if tok == TokenType.FNAME:
            return Node(NodeType.FREF, self.expect_token(TokenType.FNAME))

        raise SyntaxError("SYNTAX ERROR: expected a function")

    def parse_dfn(self) -> Node:
        self.expect_token(TokenType.RBRACE)
        statements = self.parse_statements()
        self.expect_token(TokenType.LBRACE)
        return Node(NodeType.DFN, None, statements)

    def parse_vector(self) -> Node:
        nodes = []
        while (kind := self.token().kind) in SCALARS + [TokenType.RPAREN]:
            if kind == TokenType.RPAREN:
                if self.peek_beyond([TokenType.RPAREN]).kind in SCALARS:
                    self.expect_token(TokenType.RPAREN)
                    nodes.append(self.parse_statement())
                    self.expect_token(TokenType.LPAREN)
                else:
                    break
            else:
                nodes.append(self.parse_scalar())
        if len(nodes) == 1:
            return nodes[0]
        
        return Node(NodeType.VECTOR, None, nodes[::-1]) 

    def parse_scalar(self) -> Node:
        if self.token().kind == TokenType.NAME:
            return self.parse_identifier()

        if self.token().kind in [TokenType.ALPHA, TokenType.OMEGA]:
            return Node(NodeType.ARG, self.eat_token())

        return Node(NodeType.SCALAR, self.expect_token(TokenType.SCALAR))
