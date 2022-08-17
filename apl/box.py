from apl.arr import Array

def _box_vector(vec: Array) -> None:
    data = ' '.join([f'{s.data[0]}' for s in vec.data])
    div = '─'*(len(data)-1)
    print(f"┌→{div}┐")
    print(f"│{data}│")
    print(f"└~{div}┘")

def _format_matrix(mat: Array, width: int) -> list[str]:
    res = []
    for row in range(mat.shape[0]):
        data = mat.index_cell([row]).data
        res.append((''.join([f'{s.data[0]:>{width}} ' for s in data]))[:-1])
    return res

def box(mat: Array) -> None:
    """
    Boxed display for non-nested arrays.
    """
    if mat.shape == []:
        print(mat.data[0])
        return

    strs = [str(i.data[0]) for i in mat.data]
    width = max(map(len, strs))

    if mat.rank == 1:
        _box_vector(mat)
        return 

    # Loop over 2-cells
    bars = len(mat.shape[:-2])
    cells = []
    for cell2 in mat.kcells(2).data:
        cells.append(_format_matrix(cell2, width))

    cols = len(cells[0][0])
    divider = f"{'│'*(bars+1)}{' '*cols}│"
    output = []
    output.append("┌"*bars+"┌→"+'─'*(cols-1)+"┐")
    first = True
    for cell in cells:
        for row in cell:
            if first:
                output.append(f"{'↓'*(bars+1)}{row}│")
                first = False
            else:
                output.append(f"{'│'*(bars+1)}{row}│")
        output.append(divider)
    output[-1] = "└"*bars+"└~"+'─'*(cols-1)+"┘"
    print("\n".join(output))