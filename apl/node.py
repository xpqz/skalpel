from enum import Enum, auto
from typing import Callable, Optional, TypeAlias

from apl.errors import ArityError, EmitError, NYIError
from apl.tokeniser import Token
from apl.skalpel import Arity, derive, INSTR, Voc

class NodeType(Enum):
    GETS = auto()
    ID = auto()
    FREF = auto()
    MOP = auto()
    DOP = auto()
    FUN = auto()
    DFN = auto()
    ARG = auto()
    DYADIC = auto()
    MONADIC = auto()
    VECTOR = auto()
    SCALAR = auto()
    CHUNK = auto()

NodeList: TypeAlias = list['Node'] # type: ignore

CALLABLE = {NodeType.FUN, NodeType.MOP, NodeType.DOP, NodeType.DFN, NodeType.FREF}

class Node:
    code: list[tuple] = []

    def __init__(self, kind: NodeType, tok: Optional[Token], children: Optional[NodeList] = None) -> None:
        self.kind = kind
        self.main_token = tok
        self.children = children

    def _mttok(self) -> str:
        if self.main_token is None or not isinstance(self.main_token.tok, str):
            raise EmitError('EMIT ERROR: expected a main_token of type str')
        else:
            return self.main_token.tok

    def add(self, node: 'Node') -> None:
        if self.children is None:
            self.children = [node]
        else:
            self.children.append(node)

    def emit_dfn(self) -> None:
        if self.kind == NodeType.FREF:
            fname = self._mttok()
            Node.code.append((INSTR.dfn, fname)) # reference to dfn
        else:        
            state = len(Node.code)
            Node.code.append((INSTR.dfn, None)) # Place holder
            if self.children is not None:
                for sc in self.children:
                    sc.emit()
            Node.code[state] = (INSTR.dfn, len(Node.code)-state-1)

    def monadic_function(self) -> Callable|str:
        assert self.kind in CALLABLE

        if self.kind in {NodeType.FUN, NodeType.FREF}:
            return self._mttok()

        # if self.kind == NodeType.FUN:
        #     return Voc.get_fn(self._mttok(), Arity.MONAD)

        # if self.kind == NodeType.FREF:
        #     return self._mttok()

        if self.kind == NodeType.DFN:
            raise NYIError

        return self.derived_monad()

    def dyadic_function(self) -> Callable|str:
        assert self.kind in CALLABLE

        if self.kind in {NodeType.FUN, NodeType.FREF}:
            return self._mttok()

        # if self.kind == NodeType.FUN:
        #     return Voc.get_fn(self._mttok(), Arity.DYAD)

        # if self.kind == NodeType.FREF:
        #     return self._mttok()

        if self.kind == NodeType.DFN:
            raise NYIError

        return self.derived_dyad()

    def emit_scalar(self) -> None:
        if self.main_token is None:
            raise EmitError('EMIT ERROR: main_token is undefined')
        Node.code.append((INSTR.psh, self.main_token.tok))

    def emit_id(self) -> None:
        if self.main_token is None:
            raise EmitError('EMIT ERROR: main_token is undefined')
        Node.code.append((INSTR.get, self.main_token.tok))

    def derived_monad(self) -> Callable:
        op_name = self._mttok()
        op = Voc.get_op(op_name)

        if self.children is None:
            raise EmitError('EMIT ERROR: node has no children')
    
        if op.derives != Arity.MONAD:
            raise ArityError(f"ARITY ERROR: operator '{op_name}' does not derive a monadic function")

        if op.left == Arity.MONAD:
            left = self.children[0].monadic_function()
        else:
            left = self.children[0].dyadic_function()

        if self.kind == NodeType.DOP:
            if op.right == Arity.MONAD:
                right = self.children[1].monadic_function()
            else:
                right = self.children[1].dyadic_function()
        else:
            right = None

        return derive(op.f, left, right, Arity.MONAD)  # type: ignore

    def derived_dyad(self) -> Callable:
        op_name = self._mttok()
        if self.children is None:
            raise EmitError('EMIT ERROR: node has no children')

        op = Voc.get_op(op_name)

        if op.derives != Arity.DYAD:
            raise ArityError(f"ARITY ERROR: operator '{op_name}' does not derive a dyadic function")

        if op.left == Arity.MONAD:
            left = self.children[0].monadic_function()
        else:
            left = self.children[0].dyadic_function()

        if self.kind == NodeType.DOP:
            if op.right == Arity.MONAD:
                right = self.children[1].monadic_function()
            else:
                right = self.children[1].dyadic_function()
        else:
            right = None

        return derive(op.f, left, right, Arity.DYAD) # type: ignore

    def emit_monadic_call(self) -> None:
        if self.children is None:
            raise EmitError('EMIT ERROR: node has no children')
        assert self.children[0].kind in CALLABLE

        self.children[1].emit()
        if self.children[0].kind == NodeType.DFN: # in-line dfn
            self.children[0].emit_dfn()
            Node.code.append((INSTR.mon, None))
            return 

        if self.children[0].kind in {NodeType.FUN, NodeType.FREF}:
            fn = self.children[0].monadic_function()
        else:
            fn = self.children[0].derived_monad()

        Node.code.append((INSTR.mon, fn))

    def emit_dyadic_call(self) -> None:
        if self.children is None:
            raise EmitError('EMIT ERROR: node has no children')
        assert self.children[0].kind in CALLABLE
        
        self.children[1].emit()
        self.children[2].emit()

        if self.children[0].kind == NodeType.DFN: # in-line dfn WRONG
            self.children[0].emit_dfn()
            Node.code.append((INSTR.dya, None))
            return 
        
        if self.children[0].kind in [NodeType.FUN, NodeType.FREF]:
            fn = self.children[0].dyadic_function()
        else:
            fn = self.children[0].derived_dyad()
        
        Node.code.append((INSTR.dya, fn))

    def emit_gets(self) -> None:
        if self.children is None:
            raise EmitError('EMIT ERROR: node has no children')
        assert self.children[0].kind in {NodeType.ID, NodeType.FREF}
        self.children[1].emit()
        Node.code.append((INSTR.set, self.children[0]._mttok()))

    def emit_vector(self) -> None:
        if self.children is None:                     # NOTE we need to handle ⍬ somehow
            raise EmitError('EMIT ERROR: node has no children')
        for el in self.children:
            el.emit()
        Node.code.append((INSTR.vec, len(self.children)))

    def emit_chunk(self) -> list:
        if self.children is None:
            raise EmitError('EMIT ERROR: node has no children')  # NOTE this should probably not be an error
        for sc in self.children:
            sc.emit()
        return Node.code

    def emit(self) -> Optional[list]: # type: ignore
        if self.kind == NodeType.CHUNK:
            Node.code = []
            return self.emit_chunk()
        elif self.kind == NodeType.DYADIC:
            self.emit_dyadic_call()
        elif self.kind == NodeType.GETS:
            self.emit_gets()
        elif self.kind in {NodeType.ID, NodeType.ARG}:
            self.emit_id()
        elif self.kind == NodeType.MONADIC:
            self.emit_monadic_call()
        elif self.kind == NodeType.SCALAR:
            self.emit_scalar()
        elif self.kind == NodeType.VECTOR:
            self.emit_vector()
        elif self.kind in {NodeType.DFN, NodeType.FREF}:
            self.emit_dfn()
        else:    
            raise EmitError(f'EMIT ERROR: Unknown node type: {self.kind}')
    
    def __str__(self):
        if self.kind == NodeType.ARG:
            return f"ARG({self.main_token.tok})"
        if self.kind == NodeType.SCALAR:
            return f"SCALAR({self.main_token.tok})"
        if self.kind == NodeType.FUN:
            return f"FUN({self.main_token.tok})"
        if self.kind == NodeType.FREF:
            return f"FREF({self.main_token.tok})"
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
        if self.kind == NodeType.DFN:
            body = []
            for sc in self.children:
                body.append(str(sc))
            return f"DFN[{', '.join(body)}]"
