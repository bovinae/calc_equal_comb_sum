#include <stdio.h>
#include <math.h>
#include <vector>
#include <unordered_map>
#include <sstream>
#include <map>
#include <sys/sysinfo.h>
#include <thread>
#include <atomic>
#include <unistd.h>
#include <algorithm>

#include "c.h"

using namespace std;

string resultFile = "result.txt";

typedef struct DataWithPos
{
  double data;
  int pos;
}DataWithPos;

typedef std::unordered_map<double, size_t> CombData;

int cmpfunc(const void *a, const void *b)
{
   double tmp = ((const DataWithPos*)a)->data - ((const DataWithPos*)b)->data;
   if (-0.000001 < tmp && tmp < 0.000001) return 0;
   else if (tmp > 0.000001) return 1;
   else return -1;
}

std::string toString(DataWithPos* d, vector<int>& v, vector<int>& sel) {
  printf("begin toString\n");
  std::stringstream ss;
  bool first = true;
  for (auto &&i : v) {
    if (first) {
      ss << sel[d[i].pos];
      first = false;
    }
    else {
      ss << " + " << sel[d[i].pos];
    }
  }
  
  printf("end toString: %s\n", ss.str().c_str());
  return ss.str();
}

void printVector(const char* prefix, vector<int>& v) {
  if (v.size() == 0) return;
  printf("%s: ", prefix);
  for (auto it = v.begin(); it != v.end(); it++) {
    printf("%d ", *it);
  }
  printf("\n");
}

bool checkResult(DataWithPos *d, double **mat, int row, vector<int>& a, vector<int>& b) {
  printf("begin checkResult, row=%d, aSize=%ld, bSize=%ld\n", row, a.size(), b.size());
  for (int i = 0; i < row; i++) {
    double sum1 = 0;
    for (auto &&j : a) {
      sum1 += mat[i][d[j].pos];
    }
    double sum2 = 0;
    for (auto &&j : b) {
      sum2 += mat[i][d[j].pos];
    }
    if (fabs(sum1 - sum2) > 0.01) return false;
  }
  return true;
}

double accuSum(DataWithPos *d, vector<int>& v) {
  double sum = 0;
  for (auto it = v.begin(); it != v.end(); it++) {
    sum += d[*it].data;
  }
  return sum;
}

double continueDegree(vector<int>& v) {
  int total = 0;
  for (int i = 0; i < v.size()-1 && v.size() > 1; i++) {
    if (v[i]-1 == v[i+1]) {
      // printf("%d %d\n", lhs[i], lhs[i+1]);
      total++;
    }
  }
  double ret = double(total+1) / double(v.size());
  // printVector("rhs", v);
  if (ret < 0.5) printf("%f\n", ret);
  return ret;
}

void recurse(DataWithPos* d, double **mat, int row, double target, vector<int>& lhs, unordered_map<int, bool>& lhsMap, vector<int>& rhs, int begin, vector<int>& sel) {
  // if (target <= 0) {
  //   printVector("lhs", lhs);
  //   printVector("rhs", rhs);
  // }
  // printVector("rhs", rhs);
  if (rhs.size() > 10 || continueDegree(rhs) < 0.5) return;
  // if (target < 0) return ;
  if (fabs(target) < 0.000001) {
    if (checkResult(d, mat, row, lhs, rhs)) {
      char out[1024];
      sprintf(out, "%s = %s\n", toString(d, lhs, sel).c_str(), toString(d, rhs, sel).c_str());
      writeFile(resultFile.c_str(), out);
    }
    return ;
  }

  for (int i = begin; i >= 0; i--) {
    if (lhsMap[i]) {
      continue;
    }
    rhs.push_back(i);
    recurse(d, mat, row, target-d[i].data, lhs, lhsMap, rhs, i-1, sel);
    rhs.pop_back();
  }
}

bool hasContinue(vector<int>& lhs) {
  for (int i = 0; i < lhs.size()-1; i++) {
    if (lhs[i]-1 == lhs[i+1]) {
      // printf("%d %d\n", lhs[i], lhs[i+1]);
      return true;
    }
  }
  return false;
}

void combination(DataWithPos* d, double **mat, int row, int col, int lhsNum, vector<int>& lhs, vector<int>& sel) {
  if (lhsNum == 0) {
    if (!hasContinue(lhs)) return;
    vector<int> rhs;
    unordered_map<int, bool> lhsMap;
    for (auto && j : lhs) lhsMap[j] = true;
    double target = accuSum(d, lhs);
    printVector("lhs", lhs);
    recurse(d, mat, row, target, lhs, lhsMap, rhs, lhs[0]-1, sel);
    return ;
  }

  int begin = col-1-lhs.size();
  if (lhs.size() > 0) begin = min(begin, lhs.back()-1);
  int end = 0;
  if (lhs.size() == 0) end = lhsNum;
  for (int i = begin; i >= end; i--) {
    if (lhs.size() == 0 && lhsNum > (i+1)/2) return ;
    lhs.push_back(i);
    combination(d, mat, row, col, lhsNum-1, lhs, sel);
    lhs.pop_back();
  }
}

void find(DataWithPos* d, double **mat, int row, int col, vector<int>& sel) {
  struct sysinfo si;
  sysinfo(&si);
  printf("total available ram: %ld\n", si.totalram);
  printf("get_nprocs_conf: %d\n", get_nprocs_conf());
  printf("get_nprocs: %d\n", get_nprocs());

  int max_threads = get_nprocs();
  max_threads = 1;
  std::atomic<int> thread_num{max_threads};
  for (int lhsNum = 5; lhsNum <= col/2; lhsNum++) {
    while(true) {
      if(thread_num > 0) {
        thread_num--;
        break;
      }
      sleep(10);
    }
    std::thread t([&, lhsNum]{
      printf("processing lhsNum: %d\n", lhsNum);
      vector<int> lhs;
      combination(d, mat, row, col, lhsNum, lhs, sel);
      thread_num++;
    });
    t.detach();
  }
  while(true) {
    if (thread_num == max_threads) break;
    sleep(3);
  }
}

int main(int argc, char** argv)
{  
  string fileName;
  if (argc == 2) {
    fileName = argv[1];
  } else if (argc == 3) {
    resultFile = argv[2];
    fileName = argv[1];
  }else {
    fileName = "data/data1.tsv";
  }

  vector<vector<int>> corr;
  string corrFileName = fileName;
  corrFileName = corrFileName.replace(corrFileName.rfind("data"), 4, "corr");
  corrFileName = corrFileName.replace(corrFileName.rfind("tsv"), 3, "csv");
  readCorrFile(corrFileName.c_str(), corr);
  vector<int> sel;
  for (auto &&line : corr) {
    for (auto &&cell : line) {
      sel.push_back(cell);
    }
  }
  // sort(sel.begin(), sel.end());

  int row, col;
  getRowCol(fileName.c_str(), row, col);
  col = sel.size();

  double** mat = (double**)malloc(row * sizeof(double*));
  for (size_t i = 0; i < row; i++) {
    mat[i] = (double*)malloc(col * sizeof(double));
  }

  clock_t begin = clock();
  readFile(fileName.c_str(), mat, sel);
  costtime(begin);

  double *sum = (double*)malloc(col * sizeof(double));
  for (int i = 0; i < col; i++) {
    double tmp = 0;
    for (int j = 0; j < row; j++) {
      tmp += mat[j][i];
    }
    sum[i] = tmp;
  }
  printf("%f, %f\n", sum[0], sum[col-1]);
  costtime(begin);

  DataWithPos* dataWithPos = (DataWithPos*)malloc(col * sizeof(DataWithPos));
  for (int i = 0; i < col; i++) {
    dataWithPos[i].data = mat[0][i];
    dataWithPos[i].pos = i;
  }
  // qsort(dataWithPos, col, sizeof(DataWithPos), cmpfunc);
  printf("%f, %f, %f\n", dataWithPos[0].data, dataWithPos[1].data, dataWithPos[col-1].data);
  // double base = dataWithPos[0].data;
  // for (int i = 0; i < col; i++) {
  //   dataWithPos[i].data /= base;
  //   printf("%f ", dataWithPos[i].data);
  // }
  // printf("\n");
  printf("%d, %d, %d\n", dataWithPos[0].pos, dataWithPos[1].pos, dataWithPos[col-1].pos);

  find(dataWithPos, mat, row, col, sel);

  for (size_t i = 0; i < row; i++) {
    free(mat[i]);
  }
  free(mat);

  return 0;
}

/*
14.32 prune to 4
14.36 col sum data1.tsv
14.35 col sum data2.tsv
14.34 col sum data3.tsv
*/