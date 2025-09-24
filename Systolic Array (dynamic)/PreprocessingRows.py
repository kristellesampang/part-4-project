import numpy as np

def RowRemoval(matrix):
    matrix = np.array(matrix)

    # detect any nonzero elemetns in the rows & cols
    nonzeroRows = [i for i in range(matrix.shape[0]) if np.any(matrix[i, :] != 0)]
    nonzeroCols = [j for j in range(matrix.shape[1]) if np.any(matrix[:, j] != 0)]

    # create the submatrix
    stripped = matrix[np.ix_(nonzeroRows, nonzeroCols)]

    return stripped, len(nonzeroRows),len(nonzeroCols), nonzeroRows, nonzeroCols


# example inputs
inputMatrix =   [[5,2,0,3],
                [0,0,0,0],
                [1,0,3,0],
                [0,0,0,0]]

dense, rows, cols, rmap, cmap = RowRemoval(inputMatrix)

print("Original matrix:")
print(np.array(inputMatrix))
print("Stripped matrix:")
print(dense)
print("Active rows:", rows, "Active cols:", cols)
print("Row indices kept:", rmap)
print("Col indices kept:", cmap)