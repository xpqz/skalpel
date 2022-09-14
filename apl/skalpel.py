from typing import Callable, Optional
from enum import Enum
from apl.arr import Array, S, V
from apl.stack import Stack

class INSTR(Enum):
    psh=0
    pop=1
    set=2
    get=3
    mon=4
    dya=5
    vec=6
    dfn=7
    fget=8

class TYPE(Enum):
    arr=0
    fun=1
    dfn=2

class Value:
    def __init__(self, payload: Array|Callable|list[tuple], kind:TYPE) -> None:
        self.payload = payload
        self.kind = kind


def run(code:list[tuple], env:dict[str, Value], ip:int, stack:Stack) -> None:
    while ip < len(code):
        (instr, arg) = code[ip]
        ip += 1
        if instr == INSTR.psh:
            stack.push([Value(S(arg), TYPE.arr)])
        elif instr == INSTR.set:
            env[arg] = stack.pop()[0]
        elif instr == INSTR.get:
            if arg not in env:
                raise ValueError('VALUE ERROR: Undefined name: "{arg}"')
            stack.push([env[arg]])
        elif instr == INSTR.dfn:
            if type(arg) == str: # by reference
                if arg not in env:
                    raise ValueError('VALUE ERROR: Undefined name: "{arg}"')
                f = env[arg]
                assert f.kind == TYPE.dfn # TODO: also TYPE.fun
                stack.push([f])
            else: # direct definition
                stack.push([Value(code[ip:ip+arg], TYPE.dfn)])
                ip += arg
        elif instr == INSTR.dya:
            if arg is None: # dfn
                dfn = stack.pop()[0]
                assert dfn is not None and dfn.kind == TYPE.dfn
            (alpha, omega) = stack.pop(2)
            assert alpha.kind == omega.kind == TYPE.arr
            if arg is None: 
                run(dfn.payload, {'⍺': alpha, '⍵': omega}, 0, stack) # type: ignore
            else:
                stack.push([Value(arg(alpha.payload, omega.payload), TYPE.arr)])
        elif instr == INSTR.mon:
            if arg is None: # dfn
                dfn = stack.pop()[0]
                assert dfn is not None and dfn.kind == TYPE.dfn
            omega = stack.pop()[0]
            assert omega.kind == TYPE.arr
            if arg is None:
                run(dfn.payload, {'⍵': omega}, 0, stack) # type: ignore
            else:
                stack.push([Value(arg(omega.payload), TYPE.arr)])
        elif instr == INSTR.vec:
            stack.push([Value(V([e.payload for e in stack.pop(arg)]), TYPE.arr)])    

