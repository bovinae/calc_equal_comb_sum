#include <stdio.h>
#include <math.h>
#include <sys/time.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unordered_map>
#include <numeric>
#include <iostream>
#include <unordered_set>

using namespace std;

void split(char *buf, double** data, int row);

void costtime(clock_t begin) {
  clock_t end = clock();
  double time_consumption = (double)(end - begin) / CLOCKS_PER_SEC;
  printf("%fs\n", time_consumption);
}

double** readFile1(const char* fileName, int& row, int& col) {
    FILE *fp;

    fp = fopen(fileName, "r");
    if (fp == NULL) {
        printf("open file failed!\n");
        return nullptr;
    }

    int line = -1;
    char buf[120*1024];
    double** data;
    while (fgets(buf, sizeof(buf), fp) != NULL) {
	    line++;
      if (line == 0) {
        char sep[] = "\t";
        char* s1 = NULL;
        int cnt = 0;
        for (s1 = strtok(buf, sep); s1 != NULL; s1 = strtok(NULL, sep)) {
          if (cnt == 0) row = atoi(s1);
          else if (cnt == 1) col = atoi(s1);
          cnt++;
        }
        printf("row=%d, col=%d\n", row, col);
        data = (double**)malloc(row * sizeof(double*));
        for (size_t i = 0; i < row; i++) {
          data[i] = (double*)malloc(col * sizeof(double));
        }
        continue;
      }
	    if (line > row) break;
	    split(buf, data, line-1);
    }

    fclose(fp);

    return data;
}

int getRowCol(const char* fileName, int& row, int& col) {
    FILE *fp;

    fp = fopen(fileName, "r");
    if (fp == NULL) {
        printf("open file failed!\n");
        return -1;
    }

    char buf[64];
    if (fgets(buf, sizeof(buf), fp) == NULL) return -1;
    char sep[] = "\t";
    char* s1 = NULL;
    int cnt = 0;
    for (s1 = strtok(buf, sep); s1 != NULL; s1 = strtok(NULL, sep)) {
      if (cnt == 0) row = atoi(s1);
      else if (cnt == 1) col = atoi(s1);
      cnt++;
    }
    printf("row=%d, col=%d\n", row, col);

    fclose(fp);

    return 0;
}

void split(char *buf, double** data, int row, vector<int>& sel);

int readFile(const char* fileName, double** data, vector<int>& sel) {
    FILE *fp;

    fp = fopen(fileName, "r");
    if (fp == NULL) {
        printf("open file failed!\n");
        return -1;
    }

    int row = -1;
    char buf[120*1024];
    
    while (fgets(buf, sizeof(buf), fp) != NULL) {
	    row++;
	    if (row <= 0 || row > 100) continue;
	    split(buf, data, row-1, sel);
    }

    fclose(fp);

    return 0;
}

int readCorrFile(const char* fileName, vector<vector<int>>& data) {
    FILE *fp;

    fp = fopen(fileName, "r");
    if (fp == NULL) {
        printf("open file failed!\n");
        return -1;
    }

    char buf[120*1024];
    while (fgets(buf, sizeof(buf), fp) != NULL) {
	    char sep[] = ",";
      char* s1 = NULL;
      vector<int> line;
      for (s1 = strtok(buf, sep); s1 != NULL; s1 = strtok(NULL, sep)) {
        int col = atoi(s1);
        if (col == 0) break;
        line.push_back(col);
      }
      if (line.size() == 0) break;
      data.push_back(line);
    }

    fclose(fp);

    return 0;
}

void split(char *buf, double** data, int row, vector<int>& sel) {
  unordered_map<int, double> m;
  for (auto &&i : sel) {
    m[i] = 0.0;
  }

  char sep[] = "\t";
  char* s1 = NULL;
  int col = 0;
  for (s1 = strtok(buf, sep); s1 != NULL; s1 = strtok(NULL, sep), col++) {
    if (m.find(col) != m.end()) {
      m[col] = strtod(s1, NULL);
    }
  }
  for (int i = 0; i < sel.size(); i++) {
    data[row][i] = m[sel[i]];
  }
}

int readFile(const char* fileName, double** data) {
    FILE *fp;

    fp = fopen(fileName, "r");
    if (fp == NULL) {
        printf("open file failed!\n");
        return -1;
    }

    int row = -1;
    char buf[120*1024];
    while (fgets(buf, sizeof(buf), fp) != NULL) {
	    row++;
	    if (row <= 0 || row > 100) continue;
	    split(buf, data, row-1);
    }

    fclose(fp);

    return 0;
}

void split(char *buf, double** data, int row) {
  char sep[] = "\t";
  char* s1 = NULL;
  int col = 0;
  for (s1 = strtok(buf, sep); s1 != NULL; s1 = strtok(NULL, sep)) {
    // if (col == 9999) {
    //     printf("%s\n", s1);
    // }
    // printf("%f ", strtod(s1, NULL));
    data[row][col++] = strtod(s1, NULL);
  }
  // printf("\n");
}

int writeFile(const char* fileName, const char* str) {
    FILE *fp;

    fp = fopen(fileName, "aw");
    if (fp == NULL) {
        printf("open file failed!\n");
        return -1;
    }

    int ret = fputs(str, fp);
    if (ret < 0) printf("fputs failed: %d\n", ret);

    fflush(fp);
    fclose(fp);

    return 0;
}

template <class T1, class T2>
double pearson(std::vector<T1> &inst1, std::vector<T2> &inst2) {
  if(inst1.size() != inst2.size()) {
    std::cout<<"the size of the vectors is not the same\n";
    return 0;
  }

  size_t n = inst1.size();
  double pearson = n * inner_product(inst1.begin(), inst1.end(), inst2.begin(), 0.0) - accumulate(inst1.begin(), inst1.end(), 0.0) * accumulate(inst2.begin(), inst2.end(), 0.0);
  double temp1 = n * inner_product(inst1.begin(), inst1.end(), inst1.begin(), 0.0) - pow(accumulate(inst1.begin(), inst1.end(), 0.0), 2.0);
  double temp2 = n * inner_product(inst2.begin(), inst2.end(), inst2.begin(), 0.0) - pow(accumulate(inst2.begin(), inst2.end(), 0.0), 2.0);
  temp1 = sqrt(temp1);
  temp2 = sqrt(temp2);
  pearson = pearson / (temp1 * temp2);

  return pearson;
}

template <class T>
void colCorrCluster(std::vector<std::vector<T>> &mat, vector<vector<int>>& corr) {
  if(mat.size() == 0) {
    std::cout<<"the size of the mat is 0\n";
    return ;
  }

  std::vector<std::unordered_set<int>> clusters;
  double corrThreshold = 0.4;
  for (int i = 0; i < mat.size()-1; i++) {
    cout << "processing col: " << i << endl;
    for (int j = i+1; j < mat.size(); j++) {
      double corr = pearson(mat[i], mat[j]);
      if (corr < corrThreshold && corr > -corrThreshold) continue;
      bool exists = false;
      for (auto && cluster : clusters) {
        if (cluster.find(i) != cluster.end() || cluster.find(j) != cluster.end()) {
          cluster.insert(i);
          cluster.insert(j);
          exists = true;
          break;
        }
      }
      if (!exists) {
        std::unordered_set<int> tmp;
        tmp.insert(i);
        tmp.insert(j);
        clusters.push_back(tmp);
      }
    }
  }

  for (auto &&cluster : clusters) {
    if (cluster.size() <= 2) continue;
    vector<int> tmp;
    tmp.reserve(cluster.size());
    for (auto && i : cluster) {
      tmp.push_back(i);
    }
    corr.push_back(tmp);
  }
}
