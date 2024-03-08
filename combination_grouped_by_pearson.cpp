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

std::string toString(size_t a, const vector<int>& sel) {
  // printf("begin toString\n");
  std::stringstream ss;
  int cnt = 0;
  while (a != 0) {
    size_t tmp = a & (a - 1);
    if (cnt == 0) ss << sel[getColIndex(a ^ tmp)];
    else ss << " + " << sel[getColIndex(a ^ tmp)];
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

bool checkResult(double **mat, int row, size_t aa, size_t bb, const vector<int>& sel) {
  // printf("begin checkResult: %ld, %ld\n", a, b);
  for (int i = 0; i < row; i++) {
    size_t a = aa, b = bb;
    double sum1 = 0;
    while (a != 0) {
      size_t tmp = a & (a - 1);
      // printf("%d\n", getColIndex(a ^ tmp));
      sum1 += mat[i][sel[getColIndex(a ^ tmp)]];
      a = tmp;
    }
    double sum2 = 0;
    while (b != 0) {
      size_t tmp = b & (b - 1);
      // printf("%d\n", getColIndex(b ^ tmp));
      sum2 += mat[i][sel[getColIndex(b ^ tmp)]];
      b = tmp;
    }
    if (abs(sum1 - sum2) > 0.01) return false;
  }
  return true;
}

mutex mtx;
// set<string> result_set;
// map<string, string> result_map;
void find(double **mat, int row, int col, vector<vector<int>>& corr) {
  // struct sysinfo si;
  // sysinfo(&si);
  // printf("total available ram: %ld\n", si.totalram);

  int max_threads = 1; // get_nprocs()
  // max_threads = 1;
  std::atomic<int> thread_num{max_threads};
  for (int k = 0; k < corr.size(); k++) {
    while(true) {
      if(thread_num > 0) {
        thread_num--;
        break;
      }
      sleep(3);
    }
    thread t([=, &thread_num]{
      clock_t begin = clock();
      printf("processing corr row: %d, row size:%ld\n", k, corr[k].size());
      map<size_t, size_t> result;
      CombData m;
      m.reserve(int(pow(2, corr[k].size())));
      vector<double> walk(1, 0);
      walk.reserve(int(pow(2, corr[k].size())));
      for (int i = 0; i < corr[k].size(); i++) {
        size_t len = walk.size();
        for(size_t j = 0; j < len; j++){
          double sum = round((walk[j] + mat[0][corr[k][i]])*1000000)/1000000;
          if (m.find(sum) != m.end()) {
            auto tmp2 = len + j;
            for (auto tmp1 : m[sum]) {
              auto tmp = tmp2;
              // auto tmp1 = m[sum];
              removeSameCol(tmp, tmp1);
              if (result.find(tmp) == result.end() && result.find(tmp1) == result.end()) {
                if (checkResult(mat, row, tmp, tmp1, corr[k])) {
                  char out[1024];
                  string lhsStr = toString(tmp, corr[k]);
                  string rhsStr = toString(tmp1, corr[k]);
                  mtx.lock();
                  // if (result_set.find(lhsStr) == result_set.end() || result_set.find(rhsStr) == result_set.end()) {
                  //   string lhsStrTmp = lhsStr;
                  //   for (auto it = result_map.begin(); it != result_map.end(); it++) {
                  //     size_t pos = lhsStrTmp.find(it->first);
                  //     if (pos != string::npos) {
                  //       lhsStrTmp = lhsStrTmp.replace(pos, it->first.length(), it->second);
                  //     } else {
                  //       pos = lhsStrTmp.find(it->second);
                  //       if (pos != string::npos) {
                  //         lhsStrTmp = lhsStrTmp.replace(pos, it->second.length(), it->first);
                  //       }
                  //     }
                  //   }
                  //   if (lhsStrTmp != rhsStr) {
                      sprintf(out, "%s = %s\n", lhsStr.c_str(), rhsStr.c_str());
                      writeFile(resultFile.c_str(), out);
                      // result_set.insert(lhsStr);
                      // result_set.insert(rhsStr);
                      // result_map[lhsStr] = rhsStr;
                  //   }
                  // }
                  mtx.unlock();
                  result[tmp] = tmp1;
                }
              }
            }
          }
          walk.push_back(sum);
          m[sum].push_back(len + j);
        }
      }
      CombData m_empty;
      swap(m_empty, m);
      vector<double> walk_empty;
      swap(walk_empty, walk);
      thread_num++;
      costtime(begin);
    });
    t.detach();
  }
  while (true) {
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
  // string corrFileName = fileName;
  // corrFileName = corrFileName.replace(corrFileName.rfind("data"), 4, "corr");
  // corrFileName = corrFileName.replace(corrFileName.rfind("tsv"), 3, "csv");
  // readCorrFile(corrFileName.c_str(), corr);

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

  std::vector<std::unordered_set<int>> clusters;
  colCorrCluster(mat_vec, corr);

  find(mat, row, col, corr);

  for (size_t i = 0; i < row; i++) {
    free(mat[i]);
  }
  free(mat);

  return 0;
}
