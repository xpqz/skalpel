from enum import Enum, auto
from string import ascii_letters

from apl.errors import UnexpectedToken

alpha = '_abcdefghijklmnopqrstuvwxyz∆ABCDEFGHIJKLMNOPQRSTUVWXYZ⍙ÁÂÃÇÈÊËÌÍÎÏÐÒÓÔÕÙÚÛÝþãìðòõÀÄÅÆÉÑÖØÜßàáâäåæçèéêëíîïñóôöøùúûü'
funs = '⎕[]{}!&*+,-./<=>?\\^|~×÷↑→↓∊∣∧∨∩∪≠≡≢≤≥⊂⊃⊆⊖⊢⊣⊤⊥⌈⌊⌶⌷⌽⍉⍋⍎⍒⍕⍟⍪⍬⍱⍲⍳⍴⍷⍸○'

operators = '@⌸⌹⌺⍠⌿⍀∘⍠⍣⍤⍥⍨¨'

class TokenType(Enum):
    NAME = auto()
    SCALAR = auto()
    FUN = auto()
    OPERATOR = auto()
    EOF = auto()
    LPAREN = auto()
    RPAREN = auto()
    DIAMOND = auto()
    GETS = auto()
    SINGLEQUOTE = auto()

class Token:

    def __init__(self, kind: TokenType, tok: str|int|float):
        self.kind = kind
        self.tok = tok

    def __str__(self):
        return f"Token({self.kind}, {self.tok})"

class Tokeniser:
    def __init__(self, chunk: str):
        self.chunk = chunk
        self.pos = 0

    def getname(self) -> Token:
        tok = ''
        if self.chunk[self.pos] == '⎕': # System variables and constants, like ⎕IO
            tok = '⎕'
            self.pos += 1
        while self.pos < len(self.chunk) and (self.chunk[self.pos] in alpha or self.chunk[self.pos].isdigit()):
            tok += self.chunk[self.pos]
            self.pos += 1

        return Token(TokenType.NAME, tok)

    def getnum(self) -> Token:
        tok = ''
        start = self.pos
        negative = self.chunk[self.pos] == "¯"
        if negative:
            self.pos += 1

        while self.pos < len(self.chunk) and (self.chunk[self.pos] == '.' or self.chunk[self.pos].isdigit()
                                    or self.chunk[self.pos] == 'e'):
            tok += self.chunk[self.pos]
            self.pos += 1

        if '.' in tok:
            val = float(tok)
        else:
            val = int(tok)

        if negative:
            val *= -1

        return Token(TokenType.SCALAR, val)

    def peek(self) -> str:
        try:
            return self.chunk[self.pos+1]
        except IndexError:
            return ''

    def lex(self) -> list[Token]:
        tokens = [Token(TokenType.EOF, '<EOF>')]
        while self.pos < len(self.chunk):
            hd = self.chunk[self.pos]

            if hd.isspace():  # skip whitespace
                self.pos += 1
                continue

            if hd == '⍝':
                while self.pos < len(self.chunk) and self.chunk[self.pos] != "\n":
                    self.pos += 1
                continue

            if hd == "'":  # character scalar or character vector
                tokens.append(Token(TokenType.SINGLEQUOTE, "'"))
                self.pos += 1
                for ch in str(self.getname().tok):
                    tokens.append(Token(TokenType.SCALAR, ch))
                if self.chunk[self.pos] == "'":
                    tokens.append(Token(TokenType.SINGLEQUOTE, "'"))
                    self.pos += 1
                continue

            if hd == '¯' or hd.isdigit():  # numeric scalar
                tokens.append(self.getnum())
                continue

            if hd in operators:
                tokens.append(Token(TokenType.OPERATOR, hd))
                self.pos += 1
                continue

            if hd in alpha or (hd == '⎕' and self.peek() in ascii_letters):  # Name
                tokens.append(self.getname())
                continue

            if hd in funs: # Note: must come after the name check, as ⎕ can be both function and name
                tokens.append(Token(TokenType.FUN, hd))
                self.pos += 1
                continue

            if hd in "⋄\n":
                tokens.append(Token(TokenType.DIAMOND, hd))
                self.pos += 1
                continue

            if hd == "←":
                tokens.append(Token(TokenType.GETS, hd))
                self.pos += 1
                continue

            if hd == "(":
                tokens.append(Token(TokenType.LPAREN, hd))
                self.pos += 1
                continue

            if hd == ")":
                tokens.append(Token(TokenType.RPAREN, hd))
                self.pos += 1
                continue

            raise UnexpectedToken(f"Error: unknown symbol '{hd}'")

        return tokens
