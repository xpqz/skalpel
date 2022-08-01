from aplex import Token, Tokeniser, TokenType
from aplast import Node, NodeType

class UnexpectedToken(Exception):
    pass

class Parser:

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
        ast = self.parse_chunk()
        self.expect_token(TokenType.EOF)
        return ast

    def parse_chunk(self) -> Node:
        node = self.parse_expr()
        return node

    def parse_expr(self) -> Node:
        node = self.parse_array()
        while self.token().kind in [TokenType.FUN, TokenType.OP]:
            fun = self.parse_fun()
            if fun.kind == NodeType.DYAD:
                fun.rhs = node
                fun.lhs = self.parse_array()
            else:
                fun.rhs = node
            node = fun
        return node

    def parse_fun(self) -> Node:
        if self.token().kind == TokenType.OP:
            node = Node(NodeType.MOP, self.eat_token(), self.parse_fun(), None)
        else:
            node = self.parse_simple_fun()
        return node

    def parse_simple_fun(self) -> Node:
        next = self.peek()
        if next.kind in [TokenType.SCALAR, TokenType.RPAREN]:
            node = Node(NodeType.DYAD, self.eat_token(), None, None)
        else:
            node = Node(NodeType.MONAD, self.eat_token(), None, None)
        return node

    def parse_array(self) -> Node:
        nodes = []
        while self.token().kind in [TokenType.SCALAR, TokenType.RPAREN]:
            if self.token().kind == TokenType.RPAREN:
                self.expect_token(TokenType.RPAREN)
                nodes.append(self.parse_expr())
                self.expect_token(TokenType.LPAREN)
            else:
                nodes.append(self.parse_scalar())
        nodes = nodes[::-1]
        if len(nodes) == 1:
            return nodes[0]
        
        return Node(NodeType.ARRAY, None, nodes, None) 

    def parse_scalar(self) -> Node:
        return Node(NodeType.SCALAR, self.expect_token(TokenType.SCALAR), None, None)