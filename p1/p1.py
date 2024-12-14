import numpy


def interchange_rows(matrix, r1, r2):
    tmp = matrix[r1].copy()
    matrix[r1] = matrix[r2]
    matrix[r2] = tmp


# this method convert echelon form of a matrix to its reduced echelon form.(the input matrix has to be already in
# echelon form)
def reduced_echelon_form(matrix):
    row = len(matrix)
    if row == 0:
        return matrix
    col = len(matrix[0])
    pivot_row = 0
    pivot_col = 0
    for r in range(row - 1, -1, -1):
        for c in range(col - 1):
            if matrix[r, c] != 0:
                pivot_row = r
                pivot_col = c
                break
        matrix[0:pivot_row] -= matrix[pivot_row] * matrix[0:pivot_row, pivot_col:pivot_col + 1]
    return matrix


def null_space_basis(matrix, dim):
    # echelon form.
    ef = echelon_form(matrix)

    # reduced echelon form.
    ref = reduced_echelon_form(ef)

    # an array which stores the pivot variables (the variables which are not free).
    ans = []

    # in this loop we just determine the pivot positions and fill the ans array.
    for index in range(dim):
        for j in range(dim):
            if ref[index, j] != 0:
                ans.append(j + 1)
                break

    # the matrix which stores the basis vectors in its columns.
    bases = numpy.zeros((dim, dim - len(ans)), dtype=float)

    # in here we fill the bases.
    for index in range(dim):
        for j in range(dim):
            if ref[index, j] != 0:
                cnt = 0
                for k in range(j + 1, dim):
                    if ref[index, k] != 0:
                        if (k + 1) not in ans:
                            bases[j, cnt] = (-1) * ref[index, k]
                            cnt += 1
                break
    cnt = 0
    for k in range(dim):
        if (k + 1) not in ans:
            bases[k, cnt] = 1
            cnt += 1
    return bases


def echelon_form(matrix):
    row = len(matrix)
    if row == 0:
        return matrix
    col = len(matrix[0])
    if col == 0:
        return matrix
    is_zero = True
    index = 0
    for r in range(row):
        if matrix[r, 0] != 0:
            is_zero = False
            index = r
            break
    if is_zero:
        matrix2 = echelon_form(matrix[:, 1:])
        return numpy.hstack([matrix[:, :1], matrix2])
    if index > 0:
        interchange_rows(matrix, index, 0)
    matrix[0] = matrix[0] / matrix[0, 0]
    for r in range(1, row):
        coefficient = matrix[r, 0]
        for c in range(col):
            arvand = round(matrix[0, c] * coefficient, 15)  # I used round here for a very good reason.
            if round(matrix[r, c], 15) == arvand:  # long story short; I used round to prevent a bug which took me
                # very long to find it. (if u have more question contact me)
                matrix[r, c] = 0
            else:
                matrix[r, c] = matrix[r, c] - arvand
    matrix2 = echelon_form(matrix[1:, 1:])
    return numpy.vstack([matrix[:1], numpy.hstack([matrix[1:, :1], matrix2])])


# getting the input.
print('stochastic matrix')
n = int(input('dimension of stochastic matrix : '))
mat_M = numpy.empty((n, n), dtype=float)
mat_b = numpy.empty((n, 1), dtype=float)
for i in range(n):
    mat_M[i] = input(f'row {i + 1} : ').split(" ")
for i in range(n):
    mat_b[i][0] = 0

# in here we create M - I matrix and store it in mat_c.
mat_c = mat_M.copy()
for i in range(n):
    mat_c[i, i] -= 1
print("M - I :")
print(mat_c)
print("null space basis:")
# we pass the augmented matrix of [mat_c  mat_b] to null_space_basis function
# and it will return a matrix which its columns are basis vectors.
null_basis = null_space_basis(numpy.hstack((mat_c, mat_b)), n)
if len(null_basis[0]) == 0:
    print("null space has no basis!")
    print("(in other words) dim null space is zero (0).")
else:
    # cause the dimension of null space is not zero then we have at least 1 basis vector,
    # we create a vector by a linear combination of basis vector(s) (in here we just add them)
    # and then convert it to a normal vector.
    print(null_basis)
    res = numpy.zeros((n, 1), dtype=float)
    for i in range(len(null_basis[0])):
        for j in range(n):
            res[j, 0] += null_basis[j, i]
    summation = 0
    for j in range(n):
        summation += res[j, 0]
    for j in range(n):
        res[j, 0] = res[j, 0] / summation
    print("normal vector:")
    print(res)
