# from multiprocessing import  Process
from numpy import *
import numpy as np
# import psutil
import csv
import sys

dataFile = "data/data1.tsv"
corrFile = "data/corr1.csv"
corrThreshold = 0.4

def readData():
    lines = open(dataFile).readlines()
    [row, col] = lines[0].strip('\n').split('\t')
    row, col = int(row), int(col)
    A_row = 0
    A = zeros((row, col), dtype=float)
    for line in lines[1:]:
        list = line.strip('\n').split('\t')
        A[A_row:] = list[0:col]
        A_row += 1
    return row, col, A

def corr(offset, col, A):
    if offset >= col:
        return
    arr = []
    for i in range(offset, col-1):
        print('processing col: %d' % i)
        for j in range(i+1, col):
            coef = np.corrcoef(A[:,i], A[:,j])
            if coef[0][1] < corrThreshold and coef[0][1] > -corrThreshold:
                continue
            # print(i, j, coef[0][1])
            exist = False
            for s in arr:
                if i in s or j in s:
                    exist = True
                    s.add(i)
                    s.add(j)
                    break
            if not exist:
                s = set()
                s.add(i)
                s.add(j)
                arr.append(s)
    write(arr)

def write(arr):
    f = open(corrFile, 'w', encoding='utf-8')
    csv_writer = csv.writer(f)
    for s in arr:
        if len(s) > 2:
            csv_writer.writerow(s)
    f.close()

if __name__ == '__main__':
    if (len(sys.argv) == 2):
        dataFile = str(sys.argv[1])
    elif (len(sys.argv) == 3):
        dataFile = str(sys.argv[1])
        corrFile = str(sys.argv[2])
    print(dataFile, corrFile)

    _, col, A = readData()
    corr(0, col, A)

    # process_list = []
    # cpuNum=psutil.cpu_count(False)
    # offset = 0
    # while offset < col:
    #     print("processing offset: %d" % offset)
    #     for i in range(cpuNum):
    #         p = Process(target=corr, args=(offset+i, col, A))
    #         p.start()
    #         process_list.append(p)

    #     for i in process_list:
    #         i.join()
    #     offset = offset + cpuNum
