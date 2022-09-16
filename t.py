"""
Run the APL-level tests from the file found in 

    tests/apltests.py

These tests originate from 

    https://raw.githubusercontent.com/abrudz/ngn-apl/master/t.apl
"""
from apl.arr import Array, S, match
from apl.errors import SyntaxError, RankError
from apl.parser import Parser
from apl.skalpel import run, Value
from apl.stack import Stack

EXCEPTIONS = {
    'RANK ERROR': RankError,
}

def eval_apl(src: str) -> Array:
    parser = Parser()
    ast = parser.parse(src)
    if ast is None:
        raise SyntaxError

    code = ast.emit()
    if code is None:
        raise ValueError

    env:dict[str, Value] = {}
    stack = Stack()

    run(code, env, 0, stack)

    return stack.stack[0].payload # type: ignore


def compare(a: str, b: str) -> bool:
    try:
        result_a = eval_apl(a)
    except:
        return False

    try:
        result_b = eval_apl(b)
    except:
        return False

    return match(result_a, result_b)
    
def compare_throws(a: str, aplerror: Exception) -> bool:
    try:
        eval_apl(a)
    except aplerror: # type: ignore
        return True
    except:
        return False
    return False

def main() -> None:
    with open('tests/apltests.txt') as f:
    # with open('tests/aplbasic.txt') as f:
        tests = f.read().splitlines()

    failed = []
    succeeded = []
    for t in tests:
        # print(t)
        if " ←→ " in t:
            data = t.split(" ←→ ")
            if not compare(data[0], data[1]):
                # print(f"failed: {t}")
                failed.append(t)
            else:
                print(f"succeeded: {t}")
                succeeded.append(t)
        elif " !! " in t:
            data = t.split(" !! ")
            if data[1] not in EXCEPTIONS:
                # print(f"failed: {t}")
                failed.append(t)
            elif not compare_throws(data[0], EXCEPTIONS[data[1]]): # type: ignore
                # print(f"failed: {t}")
                failed.append(t)
            else:
                print(f"succeeded: {t}")
                succeeded.append(t)
        else:
            pass
            # print(f"Malformed test: {t}")

    print(f'failed:    {len(failed)}\nsucceeded: {len(succeeded)}')

if __name__ == "__main__":
    main()


