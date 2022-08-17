from typing import Sequence

class Stack:
    def __init__(self) -> None:
        self.stack = [None for _ in range(256)]
        self.stackptr = -1

    def pop(self, n:int=1) -> list:
        v = self.stack[self.stackptr-n+1:self.stackptr+1]
        self.stackptr -= n
        return v

    def push(self, n:Sequence) -> None:
        self.stack[self.stackptr+1:self.stackptr+1] = n # Ok, so that's pretty cool
        self.stackptr += len(n)

    def peek(self) -> tuple:
        return (self.stackptr, self.stack[self.stackptr])

    def dump(self) -> list:
        return self.stack[:self.stackptr+1]