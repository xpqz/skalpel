from enum import Enum, auto
from string import ascii_letters

from apl.errors import UnexpectedToken

alpha = '_abcdefghijklmnopqrstuvwxyz∆ABCDEFGHIJKLMNOPQRSTUVWXYZ⍙ÁÂÃÇÈÊËÌÍÎÏÐÒÓÔÕÙÚÛÝþãìðòõÀÄÅÆÉÑÖØÜßàáâäåæçèéêëíîïñóôöøùúûü'
funs = '⎕!&*+,-./<=>?\\^|~×÷↑→↓∊∣∧∨∩∪≠≡≢≤≥⊂⊃⊆⊖⊢⊣⊤⊥⌈⌊⌶⌷⌽⍉⍋⍎⍒⍕⍟⍪⍱⍲⍳⍴⍷⍸○'

operators = '@⌸⌹⌺⍠⌿⍀∘⍠⍣⍤⍥⍨¨/\\'

quadfuns = ('⎕UCS')

class TokenType(Enum):
    NAME = auto()
    FNAME = auto()
    SCALAR = auto()
    FUN = auto()
    OPERATOR = auto()
    EOF = auto()
    LPAREN = auto()
    RPAREN = auto()
    LBRACE = auto()
    RBRACE = auto()
    LBRACKET = auto()
    RBRACKET = auto()
    DIAMOND = auto()
    GETS = auto()
    ALPHA = auto()
    OMEGA = auto()
    COLON = auto()
    ZILDE = auto()
    SINGLEQUOTE = auto()

TOK: dict[str, TokenType] = {
    "⋄":  TokenType.DIAMOND,
    "\n": TokenType.DIAMOND,
    "←":  TokenType.GETS,
    "(":  TokenType.LPAREN,
    ")":  TokenType.RPAREN,
    "{":  TokenType.LBRACE,
    "}":  TokenType.RBRACE,
    "[":  TokenType.LBRACKET,
    "]":  TokenType.RBRACKET,
    ":":  TokenType.COLON,
    "⍺":  TokenType.ALPHA,
    "⍵":  TokenType.OMEGA,
    "⍬":  TokenType.ZILDE,
}

class Token:

    def __init__(self, kind: TokenType, tok: str|int|float|complex) -> None:
        self.kind = kind
        self.tok = tok

    def __str__(self) -> str:
        return f"Token({self.kind}, {self.tok})"

class Tokeniser:
    def __init__(self, chunk: str) -> None:
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

        if tok.upper() in quadfuns: 
            return Token(TokenType.FUN, tok.upper())

        if tok[0].isupper(): # Names starting with Upper Case Letter is considered a function reference
            return Token(TokenType.FNAME, tok)

        return Token(TokenType.NAME, tok)

    def get_cmplx(self, tok: str) -> Token:
        parts = tok.split('J')
        if len(parts) != 2:
            raise UnexpectedToken('SYNTAX ERROR: malformed complex scalar')
        (re, im) = parts
        try:
            cmplx = complex(float(re), float(im))
        except TypeError:
            raise UnexpectedToken('SYNTAX ERROR: malformed complex scalar')

        return Token(TokenType.SCALAR, cmplx)

    def getnum(self) -> Token:
        tok = ''
        start = self.pos

        while self.pos < len(self.chunk) and (self.chunk[self.pos] in '.¯' or self.chunk[self.pos].isdigit()
                                    or self.chunk[self.pos].lower() == 'e' or self.chunk[self.pos].lower() == 'j'):
            tok += self.chunk[self.pos].upper()
            self.pos += 1

        tok = tok.replace('¯', '-')
        if 'J' in tok:
            return self.get_cmplx(tok)

        if '.' in tok:
            val = float(tok)
        else:
            val = int(tok)

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
                while True:
                    try:
                        ch = self.chunk[self.pos]
                    except IndexError:
                        raise SyntaxError("SYNTAX ERROR: Unpaired quote")
                    if ch == "'":
                        if self.peek() != "'":
                            break
                        self.pos += 1
                    tokens.append(Token(TokenType.SCALAR, ch)) 
                    self.pos += 1
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

            try:
                t = TOK[hd]
            except KeyError:
                raise UnexpectedToken(f"Error: unknown symbol '{hd}'")

            tokens.append(Token(t, hd))
            self.pos += 1

        return tokens

