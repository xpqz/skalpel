# type: ignore

from pygments.lexers.apl import APLLexer
from prompt_toolkit.lexers import PygmentsLexer
from prompt_toolkit import PromptSession
from prompt_toolkit.shortcuts import clear

from apl.box import disp
from apl.parser import Parser
from apl.skalpel import run
from apl.stack import Stack

def main():
    session = PromptSession()
    parser = Parser()
    env = {}
    style = 'apl'

    print(f'Welcome to skalpel "APL". (c) 2022 xpqz, MIT LICENSE. C-d to quit')
    while True:
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

        if src == ')py':
            style = 'python'
            continue

        if src == ')apl':
            style = 'apl'
            continue
        
        try:
            ast = parser.parse(src)
            if ast is None: # Whitespace or comments only
                continue
            code = ast.emit()
        except Exception as inst:
            print(inst) # Syntactic errors
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