# type: ignore

from pygments.lexers.apl import APLLexer
from prompt_toolkit.lexers import PygmentsLexer
from prompt_toolkit import PromptSession

from apl.arr import Array
from apl.box import box
from apl.parser import Parser
from apl.skalpel import run
from apl.stack import Stack

def main():
    session = PromptSession()
    parser = Parser()
    env = {}

    print(f'Welcome to skalpel "APL". (c) 2022 xpqz, MIT LICENSE. C-d to quit')
    while True:
        try:
            src = session.prompt('skalpel> ', lexer=PygmentsLexer(APLLexer))
        except KeyboardInterrupt:
            continue
        except EOFError:
            break

        if len(src) == 0:
            continue

        try:
            ast = parser.parse(src)
            if ast is None: # empty, whitespace only, or comment only
                continue
            code = ast.emit()
        except Exception as inst:
            print(inst)
            continue

        stack = Stack()
        run(code, env, 0, stack)

        # If the stack isn't empty, show its final result
        if stack.stackptr == 0: # one element
            result = stack.pop()[0]
            try:
                box(result)
            except:
                print(result)

if __name__=="__main__":
    main()