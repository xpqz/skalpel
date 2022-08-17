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

def run(code:Sequence, env:dict[str, Array], ip:int, stack:Stack) -> None:
    while ip < len(code):
        (instr, arg) = code[ip]
        ip += 1
        if instr == INSTR.psh:
            stack.push([S(arg)])
        elif instr == INSTR.set:
            env[arg] = stack.pop()[0]
        elif instr == INSTR.get:
            if arg not in env:
                raise ValueError('Undefined name: "{arg}"')
            stack.push([env[arg]])
        elif instr == INSTR.dya:
            (alpha, omega) = stack.pop(2)
            stack.push([arg(alpha, omega)])
        elif instr == INSTR.mon:
            stack.push([arg(stack.pop()[0])])
        elif instr == INSTR.vec:
            stack.push([V(stack.pop(arg))])    

