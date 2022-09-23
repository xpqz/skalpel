# type: ignore

import re

from pygments.lexers.apl import APLLexer
from prompt_toolkit.lexers import PygmentsLexer
from prompt_toolkit import PromptSession
from prompt_toolkit.shortcuts import clear

from apl.box import disp
from apl.parser import Parser
from apl.runtime import cmpx
from apl.skalpel import run
from apl.stack import Stack

def main():
    session = PromptSession()
    parser = Parser()
    env = {}
    
    print(f'Welcome to skalpel "APL". (c) 2022 xpqz, MIT LICENSE. C-d to quit')
    while True:
        compile_only = False
        style = 'apl'
        try:
            src = session.prompt('skalpel> ', lexer=PygmentsLexer(APLLexer))
        except KeyboardInterrupt:
            continue
        except EOFError:
            break

        if len(src) == 0:  # User hit return only
            continue

        if src == ')clear':
            env = {}
            clear()
            print('Environment reset')
            continue

        if m := re.match(r'^\s*\]compile\s+(.*)$', src):
            compile_only = True
            src = m.group(1)

        if m := re.match(r'^\s*\]py\s+(.*)$', src):
            style = 'python'
            src = m.group(1)

        if re.match(r'^\s*\]cmpx', src):
            candidates = [e for (i, e) in enumerate(src.split('"')) if i%2]
            print(cmpx(candidates))
            continue
        
        try:
            ast = parser.parse(src)
            if ast is None: # Whitespace or comments only
                continue
            code = ast.emit()
        except Exception as inst:
            print(inst) # Syntactic errors
            continue

        if compile_only:
            for i in code:
                instr = str(i[0])[6:].upper()
                print(f'{instr}  {i[1]}')
            continue

        stack = Stack()
        try:
            run(code, env, 0, stack)
        except Exception as inst: 
            print(inst) # User error
            continue        

        # If the stack isn't empty, show its final result
        if stack.stackptr == 0: # Single element
            result = stack.pop()[0]
            if style == 'python':
                print(f")py {result.payload}")
            else:
                try:
                    disp(result.payload)
                except:
                    print(f"* {result.payload}")

if __name__=="__main__":
    main()