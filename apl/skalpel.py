from typing import Sequence
from enum import Enum
from apl.arr import Array, S, V
from apl.stack import Stack

class INSTR(Enum):
    psh=0
    pop=1
    set=2
    get=3
    mon=4
    dya=6
    vec=7
    dfn=8

def run(code:list[tuple], env:dict[str, Array], ip:int, stack:Stack, callstack:list[list[tuple]] = []) -> None:
    while ip < len(code):
        (instr, arg) = code[ip]
        ip += 1
        if instr == INSTR.psh:
            stack.push([S(arg)])
        elif instr == INSTR.set:
            env[arg] = stack.pop()[0]
        elif instr == INSTR.get:
            if arg not in env:
                raise ValueError('VALUE ERROR: Undefined name: "{arg}"')
            stack.push([env[arg]])
        elif instr == INSTR.dfn:
            callstack.append(code[ip:ip+arg])
            ip += arg
        elif instr == INSTR.dya:
            (alpha, omega) = stack.pop(2)
            if arg is None: # in-line dfn
                run(callstack.pop(), {'⍺': alpha, '⍵': omega}, 0, stack, callstack)
            else:
                stack.push([arg(alpha, omega)])
        elif instr == INSTR.mon:
            omega = stack.pop()[0]
            if arg is None: # in-line dfn
                run(callstack.pop(), {'⍵': omega}, 0, stack, callstack)
            else:
                stack.push([arg(omega)])
        elif instr == INSTR.vec:
            stack.push([V(stack.pop(arg))])    

