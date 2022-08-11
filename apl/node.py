from enum import Enum, auto
from typing import Callable, Optional, TypeAlias

from apl.errors import ArityError, EmitError, ValueError
from apl.funs import Arity, FUNS, Signature
from apl.ops import derive, OPERATORS, Operator
from apl.skalpel import CMD
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
    code = []

    def __init__(self, kind: NodeType, tok: Optional[Token], children: Optional[NodeList] = None) -> None:
        self.kind = kind
        self.main_token = tok
        self.children = children

    def add(self, node: 'Node') -> None:
        if self.children is None:
            self.children = [node]
        else:
            self.children.append(node)

    def emit_monadic_function(self) -> Callable:
        assert self.kind in [NodeType.FUN, NodeType.MOP, NodeType.DOP]
        if self.kind == NodeType.FUN:
            try:
                fn: Signature = FUNS[self.main_token.tok]
            except KeyError:
                raise ValueError(f"function '{self.main_token.tok}' is not defined")
            if fn[0] is None:
                raise ArityError(f"function '{self.main_token.tok}' is not monadic")
            return fn[0]
        else:
            return self.emit_derived_monad()

    def emit_dyadic_function(self) -> Callable:
        assert self.kind in [NodeType.FUN, NodeType.MOP, NodeType.DOP]
        if self.kind == NodeType.FUN:
            try:
                fn: Signature = FUNS[self.main_token.tok]
            except KeyError:
                raise ValueError(f"function '{self.main_token.tok}' is not defined")
            if fn[1] is None:
                raise ArityError(f"function '{self.main_token.tok}' is not dyadic")
            return fn[1]
        return self.emit_derived_dyad()

    def emit_scalar(self) -> None:
        Node.code.append((CMD.push, self.main_token.tok))

    def emit_id(self) -> None:
        Node.code.append((CMD.get, self.main_token.tok))

    def emit_derived_monad(self) -> Callable:
        op_name = self.main_token.tok
        try:
            op: Operator = OPERATORS[op_name]
        except KeyError:
            raise ValueError(f"operator '{op_name}' not defined")

        if op.derives != Arity.MONAD:
            raise ValueError(f"operator '{op_name}' does not derive a monadic function")

        if op.left == Arity.MONAD:
            left = self.children[0].emit_monadic_function()
        else:
            left = self.children[0].emit_dyadic_function()

        if self.kind == NodeType.DOP:
            if op.right == Arity.MONAD:
                right = self.children[1].emit_monadic_function()
            else:
                right = self.children[1].emit_dyadic_function()
        else:
            right = None

        return derive(op.f, left, right, Arity.MONAD)

    def emit_derived_dyad(self) -> Callable:
        op_name = self.main_token.tok
        try:
            op: Operator = OPERATORS[op_name]
        except KeyError:
            raise ValueError(f"operator '{op_name}' not defined")

        if op.derives != Arity.DYAD:
            raise ValueError(f"operator '{op_name}' does not derive a dyadic function")

        if op.left == Arity.MONAD:
            left = self.children[0].emit_monadic_function()
        else:
            left = self.children[0].emit_dyadic_function()

        if self.kind == NodeType.DOP:
            if op.right == Arity.MONAD:
                right = self.children[1].emit_monadic_function()
            else:
                right = self.children[1].emit_dyadic_function()
        else:
            right = None

        return derive(op.f, left, right, Arity.DYAD)

    def emit_monadic_call(self) -> None:
        assert self.children[0].kind in [NodeType.FUN, NodeType.MOP, NodeType.DOP]
        self.children[1].emit()
        if self.children[0].kind == NodeType.FUN:
            fn = self.children[0].emit_monadic_function()
        else:
            fn = self.children[0].emit_derived_monad()

        Node.code.append((CMD.call, fn))

    def emit_dyadic_call(self) -> None:
        assert self.children[0].kind in [NodeType.FUN, NodeType.MOP, NodeType.DOP]
        self.children[1].emit()
        self.children[2].emit()
        if self.children[0].kind == NodeType.FUN:
            fn = self.children[0].emit_dyadic_function()
        else:
            fn = self.children[0].emit_derived_dyad()
        
        Node.code.append((CMD.call, fn))

    def emit_gets(self) -> None:
        assert self.children[0].kind == NodeType.ID
        self.children[1].emit()
        Node.code.append((CMD.set, self.children[0].main_token.tok))

    def emit_vector(self) -> None:
        for el in self.children:
            el.emit()
        Node.code.append((CMD.vec, len(self.children)))

    def emit_chunk(self) -> list:
        for sc in self.children:
            sc.emit()
        return Node.code

    def emit(self) -> Optional[list]:
        if self.kind == NodeType.CHUNK:
            return self.emit_chunk()
        elif self.kind == NodeType.DYADIC:
            self.emit_dyadic_call()
        elif self.kind == NodeType.GETS:
            self.emit_gets()
        elif self.kind == NodeType.ID:
            self.emit_id()
        elif self.kind == NodeType.MONADIC:
            self.emit_monadic_call()
        elif self.kind == NodeType.SCALAR:
            self.emit_scalar()
        elif self.kind == NodeType.VECTOR:
            self.emit_vector()
        else:
            raise EmitError('Unknown node type')
        return None

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