# xpqz/apl 

I wanted to implement an APL interpreter as the next phase of my APL journey. I'm not arrogant enough to think that I somehow can do better than the incumbents, but solely for my own amusement and as a means to gain insight into how things actually hang together 'under the hood'.

## Python?

I started building this in zig, and got a fair way, but learning both a for me new language and how to make an APL compiler made for slow progress. I made the decision to switch to python until I have a working prototype, and then look to zig after that. For this reason, I've tried to keep my python as non-magic as possible, and use type annotations to make a future transition to a c-like language less convoluted. 

## Unsorted useful

* Python [bitarray](https://pypi.org/project/bitarray/) (c-extension)
* [APL in R](https://bookdown.org/jandeleeuw6/apl/core.html)
* [RGS/apl](https://mathspp.com/blog/lsbasi-apl-part1)
* [dzaima/apl](https://github.com/dzaima/APL)
* [ngn/apl](https://github.com/abrudz/ngn-apl)

## 

```python
# optimise rank 2 case for monadic transpose
if rank == 2:
    idx = 0 # index into ravel of source
    for x in range(shape[0]):
        j = x # index into ravel of transpose
        for y in range(shape[1]):
            newdata[j] = deepcopy(omega.data[idx])
            idx += 1
            j += shape[0]
```