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

typedef std::unordered_map<double, size_t> CombData;

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

std::string toString(DataWithPos* d, size_t a) {
  // printf("begin toString\n");
  std::stringstream ss;
  int cnt = 0;
  while (a != 0) {
    size_t tmp = a & (a - 1);
    if (cnt == 0) ss << getColIndex(a ^ tmp);
    else ss << " + " << getColIndex(a ^ tmp);
    a = tmp;
    cnt++;
  }
  return ss.str();
}

void removeSameCol(size_t& a, size_t& b) {
  // printf("begin removeSameCol\n");
  size_t tmp = a ^ b;
  a &= tmp;
  b &= tmp;
}

bool checkResult(DataWithPos *d, double **mat, int offset, int row, size_t a, size_t b) {
  // printf("begin checkResult: %ld, %ld\n", a, b);
  for (int i = 0; i < row; i++) {
    double sum1 = 0;
    while (a != 0) {
      size_t tmp = a & (a - 1);
      // printf("%d\n", getColIndex(a ^ tmp));
      sum1 += mat[i][d[offset + getColIndex(a ^ tmp)].pos];
      a = tmp;
    }
    double sum2 = 0;
    while (b != 0) {
      size_t tmp = b & (b - 1);
      // printf("%d\n", getColIndex(b ^ tmp));
      sum2 += mat[i][d[offset + getColIndex(b ^ tmp)].pos];
      b = tmp;
    }
    if (abs(sum1 - sum2) > 0.01) return false;
  }
  return true;
}

void find(DataWithPos* d, double **mat, int row, int col) {
  struct sysinfo si;
  sysinfo(&si);
  printf("total available ram: %ld\n", si.totalram);

  for (int k = 0; k < col; k += 24) {
    clock_t begin = clock();
    printf("processing col: %d\n", k);
    map<size_t, size_t> result;
    CombData m;
    m.reserve(16*1024*1024);
    std::vector<double> walk{0};
    walk.reserve(16*1024*1024);
    // printf("sizeof walk:%ld, %f\n", walk.size(), walk[0]);
    for (int i = k; i < k+24; i++) {
      // printf("processing col: %d\n", i);
      size_t len = walk.size();
      for(size_t j = 0; j < len; j++){
        double sum = walk[j] + d[i].data;
        if (m.find(sum) != m.end()) {
          auto tmp = len + j;
          auto tmp1 = m[sum];
          removeSameCol(tmp, tmp1);
          if (result.find(tmp) == result.end() && result.find(tmp1) == result.end()) {
            if (checkResult(d, mat, k, row, tmp, tmp1)) {
              char out[1024];
              sprintf(out, "%s = %s\n", toString(d, tmp).c_str(), toString(d, tmp1).c_str());
              writeFile("result.txt", out);
              result[tmp] = tmp1;
            }
          }
        }
        walk.push_back(sum);
        m[sum] = len + j;
      }
    }
    costtime(begin);
  }

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
  double base = dataWithPos[0].data;
  for (int i = 0; i < col; i++) {
    dataWithPos[i].data /= base;
    printf("%f ", dataWithPos[i].data);
  }
  printf("\n");
  printf("%d, %d, %d\n", dataWithPos[0].pos, dataWithPos[1].pos, dataWithPos[9999].pos);

  find(dataWithPos, mat, row, col);

  for (size_t i = 0; i < row; i++) {
    free(mat[i]);
  }
  free(mat);

  return 0;
}
