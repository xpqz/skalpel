"""
Rich boxing display of arrays. Aim is to be a passable approximation
of Dyalog's

    ]box on -s=max

It's a complex problem to get right. See the file

    BOXERRORS.txt

for known formatting errors.
"""
from functools import reduce
import apl.arr as arr

def encase_vector(source: arr.Array, res: list, nested: bool, empty: bool=False) -> list:
    """
    Surround a vector with a frame.
    """
    rows = []

    if not nested and type(source.data[0]) == str:
        rows = [source.data] # type: ignore
        vtype = '─'
    else:
        vtype = '∊' if nested else '~'

        height = max(map(len, res))
        data = []
        for e in res:
            padrows = []
            if len(e) < height:
                padrows = [' '*len(e[0]) for _ in range((height-len(e))//2)]
            if isinstance(e, str):
                data.append(padrows+[e]+padrows)
            else:
                data.append(padrows+e+padrows)
        
        for row in map(list, zip(*data)): # transpose
            rows.append(list(reduce(lambda x, y:list(x)+[' ']+list(y), row))) # type: ignore

    cols = max(map(len, rows))
    if empty:
        encased = [list("┌⊖"+'─'*(cols-1)+"┐")]
    else:
        encased = [list("┌→"+'─'*(cols-1)+"┐")]
    for r in rows:
        encased.append(['│']+r+['│'])
    encased.append(list(f"└{vtype}"+'─'*(cols-1)+"┘"))

    return encased

def encase_enclosure(res: list) -> list:
    cols = len(res[0])
    encased = [list("┌"+'─'*cols+"┐")]
    for r in res:
        encased.append(['│']+r+['│'])
    encased.append(list(f"└∊"+'─'*(cols-1)+"┘"))

    return encased

def encase(source: arr.Array, res: list, *, nested: bool, empty: bool, wrap: bool=True) -> list:
    """
    Surround a character-array representation of an array with a frame.
    """
    shape = source.shape
    if shape == []:
        return encase_enclosure(res)

    if len(shape) == 1:
        return encase_vector(source, res, nested, empty)
    if type(source.data[0]) == str:
        vtype = '─'
    else:
        vtype = '∊' if nested else '~'

    # Find the overall dimensions of the cells.
    heights = []  # The vertical size we need to centre each cell into, per row
    widths = [-1]*shape[1] # The horizontal width each cell must be, per col
    for y in range(shape[0]):
        elems = res[y*shape[1]:(y+1)*shape[1]]
        heights.append(max(map(len, elems)))
        for idx, e in enumerate(elems):
            widths[idx] = max(len(e[0]), widths[idx])

    # Add vertical padding around cells that are lower than the max height,
    # and then merge the stringified elemens of each row, also padding each
    # cell if it's narrower than the widest cell in its column.
    rows = []
    for y in range(shape[0]):
        data = []
        elems = res[y*shape[1]:(y+1)*shape[1]]
        for col, e in enumerate(elems):
            padrows = []
            if len(e) < heights[y]:
                padrows = [' '*len(e[0]) for _ in range((heights[y]-len(e))//2)]
            if isinstance(e, str):
                data.append(padrows+[f"{e:<{widths[col]}}"]+padrows)
            else:
                padded = [f"{''.join(r):<{widths[col]}}" for r in e]
                data.append(padrows+padded+padrows)
        
        # Merge each row horizontally
        for row in map(list, zip(*data)): # transpose
            # Merge, separated by space
            rows.append(list(reduce(lambda x, y:list(x)+[' ']+list(y), row))) # type: ignore

    # Merge the rows vertically, and wrap with the surrounding box, if requested
    if not wrap:
        cols = max(map(len, rows))
        encased = []
        for r in rows:
            if len(r)<cols: # Account for cell separators added
                r.extend([' ']*(cols-len(r)))
            encased.append([' ']+r+[' '])
    else:
        if empty:
            cols = 1
            rows = res
            encased = [list("┌⊖"+'─'*(cols-1)+"┐")]    
        else:
            cols = max(map(len, rows))
            encased = [list("┌→"+'─'*(cols-1)+"┐")]
        for r in rows:
            if len(r)<cols: # Account for cell separators added
                r.extend([' ']*(cols-len(r)))
            encased.append(['│']+r+['│'])
        encased.append(list(f"└{vtype}"+'─'*(cols-1)+"┘"))
        if len(shape) == 2:
            if empty:
                encased[1][0] = '⌽'
            else:
                encased[1][0] = '↓'

    return encased

def _format(a: arr.Array, frame: bool=True) -> list:
    """
    Dispatch rendering on rank of cells in a.
    """
    if a.shape == []: # I am enclosed
        elem = a.data[0]
        return encase(a, _format(elem), nested=True, empty=elem.shape==[0], wrap=True)

    res = []
    nested = False
    empty = False
    for c in arr.Array.coords(a.shape):
        try: # Ugh.. dealing with empties; not great
            cell = arr.enclose_if_simple(a.get(c))
        except IndexError:
            empty = True
            cell = a.prot()
        if cell.issimple() or empty: # simple scalar
            res.append([(str(cell.data[0]).replace('-', '¯'))])
        else:
            nested = True
            res.append(box(cell))

    return encase(a, res, nested=nested, empty=empty, wrap=frame)

def box(mat: arr.Array) -> list:
    """
    Convert an array to a string representation.

    This is the entry point for boxing of arrays suitable for display.
    """

    # Simple scalars don't box. Just convert any minus sign
    if mat.issimple():
        return [str(mat.data[0]).replace('-', '¯')]

    # If we're rank 1 or 2, we need no additional adornment beyond
    # the box, arrow(s) and nest indicator.
    if mat.rank < 3:
        return _format(mat)

    # Higher-ranked arrays render as 2-cells, with the preceeding
    # axes indicated by vertical 'bars' running down the left side.

    bars = len(mat.shape[:-2])
    cells = [_format(c, False) for c in mat.kcells(2)]

    # cols = len(cells[0][0])

    cols = max(map(lambda x: len(x[0]), cells))

    divider = f"{'│'*(bars+1)}{' '*cols}│"
    first = True

    output = ["┌"*bars+"┌→"+'─'*(cols-1)+"┐"]
    for cell in cells: # Loop over 2-cells
        for row in cell:
            row_str = ''.join(row)
            hpadd = ''
            if len(row_str) < cols:
                hpadd = ' '*(cols-len(row_str))
            if first:
                output.append(f"{'↓'*(bars+1)}{row_str}{hpadd}│")
                first = False
            else:
                output.append(f"{'│'*(bars+1)}{row_str}{hpadd}│")
        output.append(divider)
    if type(mat.data[0]) == str:
        output[-1] = "└"*bars+"└"+'─'*cols+"┘"
    else:
        output[-1] = "└"*bars+"└~"+'─'*(cols-1)+"┘"

    return output

def disp(a: arr.Array) -> None:
    aa = box(a)
    for r in aa:
        print(''.join(r))