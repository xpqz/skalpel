from typing import List
from aplex import Token, Tokeniser, TokenType
from aplast import Node, NodeType

ARRAY_START = [TokenType.RPAREN, TokenType.SCALAR, TokenType.NAME]

class UnexpectedToken(Exception):
    pass

class Parser:
    """
    chunk := EOF statements
    statements := (statement "⋄")* statement
    statement := ( ID "←" | array function | function )* array
    function := f | function mop
    mop := "⍨"
    f := "+" | "-" | "×" | "÷" | "⌈" | "⌊"
    array := scalar | ( "(" statement ")" | scalar )+
    scalar := INTEGER | FLOAT | ID
    """

    def __init__(self):
        self.source = ''
        self.tokens = []
        self.current_token = -1

    def token(self) -> Token:
        return self.tokens[self.current_token]  

    def peek(self) -> Token:
        return self.tokens[self.current_token-1]

    def eat_token(self) -> Token:
        tok = self.token()
        # print(f"Ate a {tok.kind} at pos {self.current_token} -> {self.current_token -1}")
        self.current_token -= 1
        return tok

    def expect_token(self, toktype: TokenType) -> Token:
        tok = self.eat_token()
        if tok.kind != toktype:
            raise UnexpectedToken(tok.kind)
        return tok

    def parse(self, chunk: str) -> Node:
        self.source = chunk
        self.tokens = Tokeniser(self.source).lex()
        self.current_token = len(self.tokens) - 1
        return self.parse_chunk()

    def parse_chunk(self) -> Node:
        chunk = Node(NodeType.CHUNK, None, self.parse_statements())
        self.expect_token(TokenType.EOF)
        return chunk

    def parse_statements(self) -> List[Node]:
        statements = [self.parse_statement()]
        while self.token().kind == TokenType.DIAMOND:
            self.eat_token()
            statements.append(self.parse_statement())
        return statements

    def parse_statement(self) -> Node:
        """statement := ( ID "←" | array function | function )* array"""
        node = self.parse_array()
        while self.token().kind in [TokenType.FUN, TokenType.OP, TokenType.GETS]:
            if self.token().kind == TokenType.GETS:
                gets = self.expect_token(TokenType.GETS)
                node = Node(NodeType.GETS, gets, [self.parse_identifier(), node])
            else:
                fun = self.parse_fun()
                if self.token().kind in ARRAY_START:
                    node = Node(NodeType.DYADIC, None, [fun, self.parse_array(), node]) # Dyadic application of fun
                else:
                    node = Node(NodeType.MONADIC, None, [fun, node])                    # Monadic application of fun
        return node

    def parse_identifier(self) -> Node:
        return Node(NodeType.ID, self.expect_token(TokenType.NAME))

    def parse_fun(self) -> Node:
        if self.token().kind == TokenType.OP:
            op = self.expect_token(TokenType.OP)
            return Node(NodeType.MOP, op, [self.parse_fun()])        
        return self.parse_simple_fun()

    def parse_simple_fun(self) -> Node:
        return Node(NodeType.FUN, self.expect_token(TokenType.FUN))

    def parse_array(self) -> Node:
        nodes = []
        while self.token().kind in ARRAY_START:
            if self.token().kind == TokenType.RPAREN:
                self.expect_token(TokenType.RPAREN)
                nodes.append(self.parse_statement())
                self.expect_token(TokenType.LPAREN)
            else:
                nodes.append(self.parse_scalar())
        if len(nodes) == 1:
            return nodes[0]
        
        return Node(NodeType.ARRAY, None, nodes[::-1]) 

    def parse_scalar(self) -> Node:
        return Node(NodeType.SCALAR, self.expect_token(TokenType.SCALAR))