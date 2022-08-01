from enum import Enum, auto
from typing import List, Optional, Union, TypeAlias
from aplex import Token

class NodeType(Enum):
    MOP = auto()
    DOP = auto()
    DYAD = auto()
    MONAD = auto()
    ARRAY = auto()
    SCALAR = auto()

Branch: TypeAlias = Optional[Union[List['Node'], 'Node']]

class Node:
    def __init__(self, kind: NodeType, tok: Optional[Token], lhs: Branch, rhs: Branch):
        self.kind = kind
        self.main_token = tok
        self.lhs = lhs
        self.rhs = rhs

    def __str__(self):
        if self.kind == NodeType.SCALAR:
            return f"S('{self.main_token.tok}')"
        if self.kind == NodeType.DYAD:
            return f"D('{self.main_token.tok}', {self.lhs}, {self.rhs})"
        if self.kind == NodeType.MONAD:
            return f"M('{self.main_token.tok}', {self.rhs})"
        if self.kind == NodeType.DOP:
            return f"_D_('{self.main_token.tok}', {self.lhs}, {self.rhs})"
        if self.kind == NodeType.MOP:
            return f"_M('{self.main_token.tok}', {self.lhs}, {self.rhs})"
        if self.kind == NodeType.ARRAY:
            body = ''
            for sc in self.lhs:
                body += str(sc) + ", "
            return f"A({body})"
            


