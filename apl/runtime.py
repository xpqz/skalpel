"""
Somewhat noddy clone of Dyalog's `cmpx` function found in the `dfns` workspace.
This provides a way to compare execution times of short expressions.

It's available in the repl under `]cmpx`.

skalpel> ]cmpx "+/⍳10000" "{⍺+⍵}/⍳10000"
+/⍳10000     → 0.0054 |    0% ⎕⎕
{⍺+⍵}/⍳10000 → 0.1154 | 2036% ⎕⎕⎕⎕⎕⎕⎕⎕⎕⎕⎕⎕⎕⎕⎕⎕⎕⎕⎕⎕⎕⎕⎕⎕⎕⎕⎕⎕⎕⎕⎕⎕⎕⎕⎕⎕⎕⎕⎕⎕

"""
import timeit


PRELUDE = """\
from apl.parser import Parser
from apl.skalpel import run
from apl.stack import Stack

"""

def cmpx(variants: list[str], number: int=10) -> str:
    results = []
    for code in variants:
        setup = f'{PRELUDE}\nparser = Parser(); ast = parser.parse("{code}"); c = ast.emit()'
        results.append(timeit.timeit('run(c, dict(), 0, Stack())', number=number, setup=setup))

    times = [r/number for r in results]
    pc = [round(100.*t/times[0]-100.) for t in times]
    barl = [round(40*t/max(times)) for t in times]

    # Find field widths so we can align the output nicely
    table = [
        (variants[i], f"{times[i]:.4f}", str(pc[i]))
        for i in range(len(variants))
    ]

    maxwidths = [0, 0, 0]
    for row in table:
        for field in range(3):
            if len(row[field])>maxwidths[field]:
                maxwidths[field] = len(row[field])

    data = []
    for i, row in enumerate(table):
        data.append(f"{row[0]:<{maxwidths[0]}} → {row[1]:>{maxwidths[1]}} | {row[2]:>{maxwidths[2]}}% {'⎕'*barl[i]}")
    
    return "\n".join(data)
    

    