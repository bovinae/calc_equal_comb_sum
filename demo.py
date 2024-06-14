from pulp import *

def read_data(data_path):
    all_A = []
    with open(data_path) as lines:
        for line in lines:
            if len(line) < 100:
                continue
            row = line.strip().split("\t")
            all_A.append([float(x) for x in row])
    return all_A

def get_sub_matrix(all_A, index_list):
    AM = []
    for e in all_A:
        elist = []
        for idx in index_list:
            elist.append(e[idx])
        AM.append(elist)
    return AM


def get_optimal(A, index_list):
    m = len(A)
    n = len(A[0])
    prob = LpProblem("Problem", sense=LpMaximize)
    vals = [-1, 0, 1]
    cols = [i for i in range(n)]
    choices = LpVariable.dicts("Choice", (vals, cols), cat="Binary")
    # 目标函数
    prob += 0
    # 约束条件
    for c in cols:
        prob += lpSum([choices[v][c] for v in vals]) == 1
    for i in range(m):
        prob += lpSum(-choices[-1][j] * A[i][j] + choices[1][j] * A[i][j] for j in range(n)) <= 0.01
        prob += lpSum(-choices[-1][j] * A[i][j] + choices[1][j] * A[i][j] for j in range(n)) >= -0.01
    fw = open("demo_result.txt", "w")
    while True:
        prob.solve()
        if LpStatus[prob.status] == "Optimal":
            result_dict = {}
            for c in cols:
                for v in vals:
                    if value(choices[v][c]) == 1:
                        if v != 0:
                            result_dict[index_list[c]] = v
            if result_dict:
                fw.write("{}\n".format(json.dumps(result_dict)))
            # 当前解加入到约束条件中，避免下次循环出现重复解
            prob += lpSum([choices[v][c] for v in vals for c in cols
                           if value(choices[v][c]) == 1]) <= n - 1
            prob += lpSum([choices[-v][c] for v in vals for c in cols
                           if value(choices[v][c]) == 1]) <= n - 1
        else:
            break
    fw.close()


if __name__ == "__main__":
    # 数据路径
    data_path = "data/data1.tsv"
    cols_list = [1034, 1864, 1265, 1456, 1113, 1664,
                 1776, 1264, 1900, 1948, 1000, 1001,
                 1002, 1003, 1004, 1005]
    # 获取系数矩阵
    all_A = read_data(data_path)
    A = get_sub_matrix(all_A, cols_list)
    # 求解
    get_optimal(A, cols_list)
