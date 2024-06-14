#include <stdio.h>
#include <math.h>
#include <vector>
#include <set>
#include <list>
#include <unordered_map>
#include <sstream>
#include <map>
// #include <sys/sysinfo.h>
#include <thread>
#include <atomic>
#include <unistd.h>
#include <fcntl.h>           /* For O_* constants */
#include <sys/stat.h>        /* For mode constants */
#include <mutex>

#include "c.h"

using namespace std;

string resultFile = "result.txt";

typedef std::unordered_map<double, vector<size_t>> CombData;

int getColIndex(size_t a) {
  int index = 0;
  while(a != 1) {
    index++;
    a >>= 1;
  }
  return index;
}

std::string toString(unordered_set<int>& hs) {
  std::stringstream ss;
  int cnt = 0;
  for (auto &&i : hs) {
    if (cnt == 0) ss << i;
    else ss << " + " << i;
    cnt++;
  }
  return ss.str();
}

bool checkResult(double **mat, int row, unordered_set<int>& lhs, unordered_set<int>& rhs) {
  if (lhs.size() < 2 || rhs.size() < 2) return false;
  // printf("begin checkResult: %ld, %ld\n", a, b);
  for (int i = 0; i < row; i++) {
    double sum1 = 0;
    for (auto &&j : lhs) {
      sum1 += mat[i][j];
    }
    double sum2 = 0;
    for (auto &&j : rhs) {
      sum2 += mat[i][j];
    }
    if (abs(sum1 - sum2) > 0.01) return false;
  }
  return true;
}

void find(double **mat, int row, int col, vector<vector<unordered_set<int>>>& corr) {
  for (int k = 0; k < corr.size(); k++) {
    clock_t begin = clock();
    printf("processing corr row: %d, row size:%ld\n", k, corr[k][0].size() + corr[k][1].size());
    if (checkResult(mat, row, corr[k][0], corr[k][1])) {
      char out[1024];
      string lhsStr = toString(corr[k][0]);
      string rhsStr = toString(corr[k][1]);
      sprintf(out, "%s = %s\n", lhsStr.c_str(), rhsStr.c_str());
      writeFile(resultFile.c_str(), out);
    }
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

  int row, col;
  getRowCol(fileName.c_str(), row, col);

  double** mat = (double**)malloc(row * sizeof(double*));
  for (size_t i = 0; i < row; i++) {
    mat[i] = (double*)malloc(col * sizeof(double));
  }

  clock_t begin = clock();
  readFile(fileName.c_str(), mat);
  costtime(begin);

  vector<vector<double>> mat_vec;
  mat_vec.reserve(col);
  for (size_t i = 0; i < col; i++) {
    vector<double> v;
    v.reserve(row);
    for (size_t j = 0; j < row; j++) {
      v.push_back(mat[j][i]);
    }
    mat_vec.push_back(v);
  }

  vector<vector<unordered_set<int>>> clusters;
  colCorrCluster(mat_vec, clusters);
  for (auto &&i : clusters) {
    int cnt = 0;
    for (auto &&j : i) {
      for (auto &&k : j) {
        printf("%d ", k);
      }
      if (cnt == 0) printf(" | ");
      cnt++;
    }
    printf("\n");
  }
  
  find(mat, row, col, clusters);

  for (size_t i = 0; i < row; i++) {
    free(mat[i]);
  }
  free(mat);

  return 0;
}
