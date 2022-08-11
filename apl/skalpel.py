import types
from typing import Callable, Optional
from enum import Enum

class CMD(Enum):
    push=0
    pop=1
    set=2
    get=3
    call=4
    vec=5

class SYM(Enum):
    const=0
    var=1
    primitive=2
    fun=3

class ExpectedArity(Exception):
    pass 

class BuiltIn:
    def __init__(self, arity: int, op: Callable):
        if arity not in [1, 2]:
            raise ExpectedArity('A function needs an arity of 1 or 2')
        self.arity = arity
        self.op = op

class Fun:
    def __init__(self, arity: int, op: Callable):
        if arity not in [1, 2]:
            raise ExpectedArity('A function needs an arity of 1 or 2')
        self.arity = arity
        self.op = op

class Var:
    def __init__(self, value: int):
        self.value = value

 # code:bytecode,env:environment,ip:instruction counter,stack:stack
def vm(code, env, ip, stack):
    while ip<len(code):
        (cmd, arg) = code[ip]
        if cmd == CMD.push:
            stack.append(arg)
            ip += 1
        elif cmd == CMD.set:
            env[arg] = Var(stack.pop())
            ip += 1
        elif cmd == CMD.get:
            if arg not in env:
                raise ValueError('Undefined variable: "{arg}"')
            var = env[arg]
            if not isinstance(var, Var):
                raise ValueError('Expected a variable: "{arg}"')
            stack.append(env[arg].value)
            ip += 1
        elif cmd == CMD.call:
            if arg not in env:
                raise ValueError(f'Undefined function: "{arg}"')
            sym = env[arg]
            if not isinstance(sym, (BuiltIn, Fun)):
                raise ValueError(f'Not callable: "{arg}"')
            if sym.arity == 2:
                omega = stack.pop()
                alpha = stack.pop()                
                r = sym.op(alpha, omega)
            else:
                omega = stack.pop()
                r = sym.op(omega)
            stack.append(r)
            ip += 1
        else:
            raise ValueError(f'Unknown instruction: {cmd}')
    return stack.pop()

if __name__ == "__main__":
    symtab = {
        '+': BuiltIn(2, lambda x, y:x+y)
    }

    code = [(CMD.push, 5), (CMD.set, 'a'), (CMD.push, 7), (CMD.get, 'a'), (CMD.call, '+')]

    print(vm(code, symtab, 0, []))
            
