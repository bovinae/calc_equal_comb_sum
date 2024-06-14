from numpy import *
import numpy as np
import sys

dataFile = "data/data1.tsv"

def readData():
    lines = open(dataFile).readlines()
    [row, col] = lines[0].strip('\n').split('\t')
    row, col = int(row), int(col)
    A_row = 0
    A = zeros((row, col), dtype=double)
    for line in lines[1:]:
        list = line.strip('\n').split('\t')
        A[A_row:] = list[0:col]
        A_row += 1
    return row, col, A

if __name__ == '__main__':
    args = sys.argv[1:]
    lhs, rhs = [], []
    curr = lhs
    for arg in args:
        if arg == '+':
            continue
        elif arg == '=':
            curr = rhs
            continue
        curr.append(int(arg))

    print("input col number:", lhs, rhs)

    row, col, A = readData()
    lhsSum = A[:,lhs[0]]
    rhsSum = A[:,rhs[0]]
    for i in range(1, len(lhs)):
        lhsSum = np.add(lhsSum, A[:,lhs[i]])
    for i in range(1, len(rhs)):
        rhsSum = np.add(rhsSum, A[:,rhs[i]])
    lhsSum = np.round(lhsSum, decimals=6)
    rhsSum = np.round(rhsSum, decimals=6)

    if np.array_equal(lhsSum, rhsSum):
        print("equal")
    else:
        print("not equal")

    arr = lhs + rhs
    for i in range(len(arr) - 1):
        for j in range(i+1, len(arr)):
            print("{}, {}, {}".format(arr[i], arr[j], np.corrcoef(A[:,arr[i]], A[:,arr[j]])[0][1]))
