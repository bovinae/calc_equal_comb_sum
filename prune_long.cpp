#include <stdio.h>
#include <math.h>
#include <vector>
#include <unordered_map>
#include <sstream>
#include <map>
#include <sys/sysinfo.h>
#include <thread>

#include "c.h"

using namespace std;

typedef struct DataWithPos
{
  double data;
  int pos;
}DataWithPos;

typedef std::unordered_map<double, vector<int>> CombData;

int cmpfunc(const void *a, const void *b)
{
   double tmp = ((const DataWithPos*)a)->data - ((const DataWithPos*)b)->data;
   if (-0.000001 < tmp && tmp < 0.000001) return 0;
   else if (tmp > 0.000001) return 1;
   else return -1;
}

int getColIndex(size_t a) {
  int index = 0;
  while(a != 1) {
    index++;
    a >>= 1;
  }
  return index;
}

std::string toString(DataWithPos* d, vector<int>& v) {
  printf("begin toString\n");
  std::stringstream ss;
  bool first = true;
  for (auto &&i : v) {
    if (first) {
      ss << d[i].pos;
      first = false;
    }
    else {
      ss << " + " << d[i].pos;
    }
  }
  
  return ss.str();
}

double accuSum(DataWithPos *d, vector<int>& v) {
  double sum = 0;
  for (auto &&i : v) {
    sum += d[i].data;
  }
  return sum;
}

void removeSameCol(vector<int>& a, vector<int>& b) {
  // printf("begin removeSameCol\n");
  auto left = a.begin(), right = b.begin();
  while(left != a.end() && right != b.end()) {
    if (*left == *right) {
      left = a.erase(left);
      right = b.erase(right);
    } else if (*left < *right) {
      left++;
    } else {
      right++;
    }
  }
}

bool checkResult(DataWithPos *d, double **mat, int row, vector<int>& a, vector<int>& b) {
  // printf("begin checkResult, row=%d, aSize=%ld, bSize=%ld\n", row, a.size(), b.size());
  for (int i = 0; i < row; i++) {
    double sum1 = 0;
    for (auto &&j : a) {
      sum1 += mat[i][d[j].pos];
    }
    double sum2 = 0;
    for (auto &&j : b) {
      sum2 += mat[i][d[j].pos];
    }
    if (abs(sum1 - sum2) > 0.01) return false;
  }
  return true;
}

void find(DataWithPos* d, double **mat, int row, int col) {
  struct sysinfo si;
  sysinfo(&si);
  printf("total available ram: %ld\n", si.totalram);

  clock_t begin = clock();
  map<vector<int>, vector<int>> result;
  CombData m;
  std::vector<vector<int>> walk{vector<int>{}};
  for (int i = 0; i < col; i++) {
    printf("processing col: %d\n", i);
    size_t len = walk.size();
    printf("walk size: %ld\n", len);
    for(size_t j = 0; j < len; j++){
      auto tmp = walk[j];
      tmp.push_back(i);
      double sum = accuSum(d, tmp);
      if (m.find(sum) == m.end()) {
        if(tmp.size() <= 12) {
          walk.push_back(tmp);
          m[sum] = tmp;
        }
        continue;
      }
      auto tmp1 = m[sum];
      removeSameCol(tmp, tmp1);
      // if (result.find(tmp) == result.end() && result.find(tmp1) == result.end()) {
        if (checkResult(d, mat, row, tmp, tmp1)) {
          char out[1024];
          sprintf(out, "%s = %s\n", toString(d, tmp).c_str(), toString(d, tmp1).c_str());
          writeFile("result.txt", out);
          result[tmp] = tmp1;
        }
      // }
    }
  }
  costtime(begin);

  printf("no same column\n");
}

int main(int argc, char** argv)
{
  int row = 100, col = 10000;
  double** mat = (double**)malloc(row * sizeof(double*));
  for (size_t i = 0; i < row; i++) {
    mat[i] = (double*)malloc(col * sizeof(double));
  }

  clock_t begin = clock();
  if (argc == 2) {
    readFile(argv[1], mat);
  } else {
    readFile("data/data1.tsv", mat);
  }
  costtime(begin);

  double *sum = (double*)malloc(col * sizeof(double));
  for (int i = 0; i < col; i++) {
    double tmp = 0;
    for (int j = 0; j < row; j++) {
      tmp += mat[j][i];
    }
    sum[i] = tmp;
  }
  printf("%f, %f\n", sum[0], sum[9999]);
  costtime(begin);

  DataWithPos* dataWithPos = (DataWithPos*)malloc(col * sizeof(DataWithPos));
  for (int i = 0; i < col; i++) {
    dataWithPos[i].data = sum[i];
    dataWithPos[i].pos = i;
  }
  qsort(dataWithPos, col, sizeof(DataWithPos), cmpfunc);
  printf("%f, %f, %f\n", dataWithPos[0].data, dataWithPos[1].data, dataWithPos[9999].data);
  // double base = dataWithPos[0].data;
  // for (int i = 0; i < col; i++) {
  //   dataWithPos[i].data /= base;
  //   printf("%f ", dataWithPos[i].data);
  // }
  printf("\n");
  printf("%d, %d, %d\n", dataWithPos[0].pos, dataWithPos[1].pos, dataWithPos[9999].pos);

  find(dataWithPos, mat, row, col);

  for (size_t i = 0; i < row; i++) {
    free(mat[i]);
  }
  free(mat);

  return 0;
}
