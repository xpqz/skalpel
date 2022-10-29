from enum import Enum, auto
from typing import Optional, TypeAlias

from apl.errors import ArityError, EmitError
from apl.tokeniser import Token
from apl.skalpel import Arity, INSTR, Voc

class NodeType(Enum):
    GETS = auto()
    ID = auto()
    FREF = auto()
    MOP = auto()
    DOP = auto()
    FUN = auto()
    DFN = auto()
    ARG = auto()
    IDX = auto()
    DYADIC = auto()
    MONADIC = auto()
    VECTOR = auto()
    CHARVEC = auto()
    SCALAR = auto()
    SYSTEM = auto()
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
        if not self.main_token or not isinstance(self.main_token.tok, str):
            raise EmitError('EMIT ERROR: expected a main_token of type str')
        else:
            return self.main_token.tok

    def add(self, node: 'Node') -> None:
        if not self.children:
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
                    if sc.kind != NodeType.GETS: # First non-assignment is returned. TODO: guards
                        break
            Node.code[state] = (INSTR.dfn, len(Node.code)-state-1)

    def monadic_function(self) -> Optional[str]:
        assert self.kind in CALLABLE

        if self.kind in {NodeType.FUN, NodeType.FREF}:
            return self._mttok()

        if self.kind == NodeType.DFN:
            self.emit_dfn()
            return None

        return self.derived_monad()

    def dyadic_function(self) -> Optional[str]:
        assert self.kind in CALLABLE

        if self.kind in {NodeType.FUN, NodeType.FREF}:
            return self._mttok()

        if self.kind == NodeType.DFN: # Inline dfn
            self.emit_dfn()
            return None

        return self.derived_dyad()

    def emit_system(self) -> None:
        if not self.main_token or type(self.main_token.tok) != str:
            raise SyntaxError(f'SYNTAX ERROR: expected a system array')
        token = self.main_token.tok
        val = Voc.arrs.get(token)
        if not val:
            val = Voc.arrs.get(token.lower())
            if not val:
                raise ValueError(f'VALUE ERROR: unknown system array "{token}"')
        Node.code.append((INSTR.get, token.upper()))

    def emit_scalar(self) -> None:
        if not self.main_token:
            raise EmitError('EMIT ERROR: main_token is undefined')
        Node.code.append((INSTR.push, self.main_token.tok))

    def emit_idx(self) -> None:
        if not self.children:
            raise EmitError('EMIT ERROR: node has no children')
        (receiver, idx) = self.children
        idx.emit()
        if receiver.kind in {NodeType.ARG, NodeType.ID}:
            Node.code.append((INSTR.geti, receiver.main_token.tok)) # type: ignore
        elif receiver.kind == NodeType.VECTOR:
            receiver.emit_vector()
            Node.code.append((INSTR.geti, None))
        else:
            raise EmitError(f"EMIT ERROR: unexpected node {receiver.kind}")

    def emit_id(self) -> None:
        if not self.main_token:
            raise EmitError('EMIT ERROR: main_token is undefined')
        Node.code.append((INSTR.get, self.main_token.tok))

    def derived_monad(self) -> str: # op deriving monad
        op_name = self._mttok()
        op = Voc.get_op(op_name)

        if not self.children:
            raise EmitError('EMIT ERROR: node has no children')
    
        if op.derives != Arity.MONAD:
            raise ArityError(f"ARITY ERROR: operator '{op_name}' does not derive a monadic function")

        if self.kind == NodeType.DOP:
            if op.right == Arity.MONAD:
                right = self.children[1].monadic_function()
            else:
                right = self.children[1].dyadic_function()        
            Node.code.append((INSTR.fun, right))

        if op.left == Arity.MONAD:
            left = self.children[0].monadic_function()
        else:
            left = self.children[0].dyadic_function()

        if left: # Note: for in-line dfns we already have the dfn on the stack
            Node.code.append((INSTR.fun, left))

        return op_name

    def derived_dyad(self) -> str: # op deriving dyad
        op_name = self._mttok()
        if not self.children:
            raise EmitError('EMIT ERROR: node has no children')

        op = Voc.get_op(op_name)

        if op.derives != Arity.DYAD:
            raise ArityError(f"ARITY ERROR: operator '{op_name}' does not derive a dyadic function")

        if self.kind == NodeType.DOP:
            if op.right == Arity.MONAD:
                right = self.children[1].monadic_function()
            else:
                right = self.children[1].dyadic_function()
    
            if right:
                Node.code.append((INSTR.fun, right))

        if op.left == Arity.MONAD:
            left = self.children[0].monadic_function()
        else:
            left = self.children[0].dyadic_function()
    
        if left: # Note: for in-line dfns we already have the dfn on the stack
            Node.code.append((INSTR.fun, left))

        return op_name

    def emit_monadic_call(self) -> None:
        if not self.children:
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
        if not self.children:
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
        if not self.children:
            raise EmitError('EMIT ERROR: node has no children')
        assert self.children[0].kind in {NodeType.ID, NodeType.FREF, NodeType.IDX}

        (receiver, value) = self.children
        value.emit()

        if receiver.kind == NodeType.IDX: # Indexed gets
            (name, idx) = receiver.children # type: ignore
            idx.emit()
            Node.code.append((INSTR.seti, name._mttok()))
        else:
            Node.code.append((INSTR.set, receiver._mttok()))

    def emit_vector(self) -> None:
        for el in self.children: # type: ignore
            el.emit()
        if self.kind == NodeType.CHARVEC:
            Node.code.append((INSTR.cvec, len(self.children)))  # type: ignore
        else:            
            Node.code.append((INSTR.vec, len(self.children)))  # type: ignore

    def emit_chunk(self) -> list:
        if not self.children:
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
        elif self.kind == NodeType.SYSTEM:
            self.emit_system()
        elif self.kind == NodeType.MONADIC:
            self.emit_monadic_call()
        elif self.kind == NodeType.SCALAR:
            self.emit_scalar()
        elif self.kind in {NodeType.VECTOR, NodeType.CHARVEC}:
            self.emit_vector()
        elif self.kind == NodeType.IDX:
            self.emit_idx()
        elif self.kind in {NodeType.DFN, NodeType.FREF}:
            self.emit_dfn()
        else:    
            raise EmitError(f'EMIT ERROR: Unknown node type: {self.kind}')
    
    def __str__(self):
        if self.kind == NodeType.ARG:
            return f"ARG({self.main_token.tok})"
        if self.kind == NodeType.SYSTEM:
            return f"SYS({self.main_token.tok})"
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
        if self.kind == NodeType.IDX:
            return f"IDX({self.children[0]}, {self.children[1]})"
        if self.kind in {NodeType.VECTOR, NodeType.CHARVEC}:
            body = []
            for sc in self.children:
                body.append(str(sc))
            if self.kind == NodeType.CHARVEC:
                return f"CVEC[{', '.join(body)}]"
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
