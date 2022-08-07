from enum import Enum, auto
from typing import Optional, TypeAlias
from apl.tokeniser import Token

class NodeType(Enum):
    GETS = auto()
    ID = auto()
    MOP = auto()
    DOP = auto()
    FUN = auto()
    DYADIC = auto()
    MONADIC = auto()
    VECTOR = auto()
    SCALAR = auto()
    CHUNK = auto()

NodeList: TypeAlias = list['Node'] # type: ignore

class Node:
    def __init__(self, kind: NodeType, tok: Optional[Token], children: Optional[NodeList] = None):
        self.kind = kind
        self.main_token = tok
        self.children = children

    def add(self, node: 'Node') -> None:
        if self.children is None:
            self.children = [node]
        else:
            self.children.append(node)

    def __str__(self):
        if self.kind == NodeType.SCALAR:
            return f"SCALAR({self.main_token.tok})"
        if self.kind == NodeType.FUN:
            return f"FUN({self.main_token.tok})"
        if self.kind == NodeType.ID:
            return f"ID('{self.main_token.tok}')"
        if self.kind == NodeType.DYADIC:
            return f"DYADIC({self.children[0]}, {self.children[1]}, {self.children[2]})"
        if self.kind == NodeType.GETS:
            return f"GETS({self.children[0]}, {self.children[1]})"
        if self.kind == NodeType.MONADIC:
            return f"MONADIC({self.children[0]}, {self.children[1]})"
        if self.kind == NodeType.DOP:
            return f"DOP('{self.main_token.tok}', {self.children[0]}, {self.children[1]})"
        if self.kind == NodeType.MOP:
            return f"MOP('{self.main_token.tok}', {self.children[0]})"
        if self.kind == NodeType.VECTOR:
            body = []
            for sc in self.children:
                body.append(str(sc))
            return f"VEC[{', '.join(body)}]"
        if self.kind == NodeType.CHUNK:
            body = []
            for sc in self.children:
                body.append(str(sc))
            return f"CHNK[{', '.join(body)}]"
            


