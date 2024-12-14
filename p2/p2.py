import numpy


def mat_multiply(mat, vector, dim):
    result = numpy.empty((dim, 1), dtype=float)

    for j in range(dim):
        summation = 0
        for k in range(dim):
            summation += mat[j, k] * vector[k, 0]
        result[j, 0] = summation
    return result


if __name__ == '__main__':
    myFile = open("D:\\Arvand\\uni\\term 4\\Linear Algebra\\HW\\HW4\\graph.txt", "r")
    cont = myFile.read().split("\n")
    myFile.close()
    n = 4
    mat_M = numpy.zeros((n, n), dtype=float)
    degree = numpy.zeros(n, dtype=int)
    ranks = numpy.empty((n, 1), dtype=float)
    for i in range(len(cont)):
        mat_M[int(cont[i][4]) - 1][int(cont[i][1]) - 1] = 1
        degree[int(cont[i][1]) - 1] += 1
    for i in range(n):
        ranks[i, 0] = 0.25
        mat_M[:, i] /= degree[i]
    print("Matrix M:")
    print(mat_M)
    print("ranks in t = 0:")
    print(ranks)
    for i in range(7):
        print(f"t = {i+1}")
        ranks = mat_multiply(mat_M, ranks, n)
        print(ranks)
