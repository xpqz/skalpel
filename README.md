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
* [prompt toolkit](https://python-prompt-toolkit.readthedocs.io)

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

## The Array

Everything in APL is an array, even scalars. An array's shape defines its dimensionality. A scalar is an array with empty shape and a rank 0:

    Array([], [5])

A vector has a single-element shape (rank 1), the sole element denoting the length of the vector:

    Array([5], [1, 2, 3, 4, 5]) # A length 5 vector

This idea generalises to higher ranks. What we normally think of as a matrix -- a 2D array -- has rank 2:

    Array([2, 3], [2, 3, 6, 4, 1, 6])

    ┌→────┐
    ↓2 3 6│
    │4 1 6│
    └~────┘

APLs arrays are stored row-major, so in a rank 2 array, the leading axis -- the first -- defines the rows, that is (y, x).

### Cells

In leading axis theory, each array consists of a set of cells of varying degree, from the rank and down to, and including zero. Thus, the rank 2 array above consists of cells of degree 2 (a single 2-cell, the array itself), rank 1 (two 1-cells; the rows), and, finally, 6 0-cells; the scalars. Each such cell is in itself an array, as per the definition above. 

2 3


We can express the general idea like this: any given array of rank R can be composed from a set of cells of rank S

      The result of asking for the n-cells of an array of rank r
        is an APLArray of rank (r-n) with shape equal to the first
        (r-n) elements of the shape of the original array.

