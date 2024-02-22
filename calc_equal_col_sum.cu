#include <stdio.h>
#include <math.h>
#include <vector>
#include <sstream>
#include <map>
#include <sys/sysinfo.h>
#include <thread>
#include <atomic>
#include <unistd.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include "c.h"

using namespace std;

string resultFile = "result.txt";

typedef struct DataWithPos
{
  double data;
  int pos;
}DataWithPos;

int cmpfunc(const void *a, const void *b)
{
   double tmp = ((const DataWithPos*)a)->data - ((const DataWithPos*)b)->data;
   if (-0.000001 < tmp && tmp < 0.000001) return 0;
   else if (tmp > 0.000001) return 1;
   else return -1;
}

template<typename T>
struct my_vector {
    int cap;
    int len;
    T* mv;

    my_vector() : cap(5000), len(0) {
      cudaMallocManaged(&mv, 5000 * sizeof(T));
    }
    explicit my_vector(size_t _cap) : cap(_cap), len(0) {
      cudaMallocManaged(&mv, _cap * sizeof(T));
    }
    explicit my_vector(vector<int>& v) : cap(v.size()), len(v.size()) {
      cudaMallocManaged(&mv, v.size() * sizeof(T));
      for (int i = 0; i < v.size(); i++) {
        mv[i] = v[i];
      }
    }
    explicit my_vector(my_vector<int>& v) : cap(v.cap), len(v.len) {
      cudaMallocManaged(&mv, v.cap * sizeof(T));
      for (int i = 0; i < v.len; i++) {
        mv[i] = v.mv[i];
      }
    }
    ~my_vector() {
      cudaFree(mv);
    }
};

__host__ __device__
void printVector(const char* prefix, thrust::device_vector<int>& v) {
  if (v.size() == 0) return;
  printf("%s: ", prefix);
  for (int i = 0; i < v.size(); i++) {
    printf("%d ", int(v[i]));
  }
  printf("\n");
}

__host__ __device__
void printVector(const char* prefix, my_vector<int>& v) {
  if (v.len == 0) return;
  printf("%s: ", prefix);
  for (int i = 0; i < v.len; i++) {
    printf("%d ", int(v.mv[i]));
  }
  printf("\n");
}

__host__ __device__
void printVector(const char* prefix, int* lhs, int lhsSize) {
  if (lhsSize == 0) return;
  printf("%s: ", prefix);
  for (int i = 0; i < lhsSize; i++) {
    printf("%d ", int(lhs[i]));
  }
  printf("\n");
}

__host__ __device__
void printVector(int* lhs, int lhsSize, my_vector<int>& rhs) {
  if (lhsSize == 0 || rhs.len == 0) return;
  for (int i = 0; i < lhsSize; i++) {
    if (i == 0) printf("%d", int(lhs[i]));
    else printf(" + %d", int(lhs[i]));
  }
  printf(" = ");
  for (int i = 0; i < rhs.len; i++) {
    if (i == 0) printf("%d", int(rhs.mv[i]));
    else printf(" + %d", int(rhs.mv[i]));
  }
  printf("\n");
}

__device__
bool checkResult(DataWithPos *d, double **mat, int row, int* a, int lhsSize, my_vector<int>& b) {
  // printf("begin checkResult, row=%d\n", row);
  for (int i = 0; i < row; i++) {
    double sum1 = 0;
    for (int j = 0; j < lhsSize; j++) {
      sum1 += mat[i][d[a[j]].pos];
    }
    double sum2 = 0;
    for (int j = 0; j < b.len; j++) {
      sum2 += mat[i][d[b.mv[j]].pos];
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

__device__
void recurse(DataWithPos* d, double **mat, int row, int target, int* lhs, int lhsSize, char* lhsMap, my_vector<int>& rhs, int begin) {
  // printf("begin recurse\n");
  // if (target <= 0) {
  //   printVector("lhs", lhs);
  //   printVector("rhs", rhs);
  // }
  if (target < 0) return ;
  if (target == 0) {
    if (checkResult(d, mat, row, lhs, lhsSize, rhs)) {
      // char out[1024];
      // printf("%s = %s\n", toString(d, lhs).c_str(), toString(d, rhs).c_str());
      printVector(lhs, lhsSize, rhs);
      // writeFile(resultFile.c_str(), out);
    }
    return ;
  }

  for (int i = begin; i >= 0; i--) {
    if (lhsMap[i] == 1) {
      continue;
    }
    rhs.mv[rhs.len] = i;
    rhs.len++;
    recurse(d, mat, row, target-d[i].data, lhs, lhsSize, lhsMap, rhs, i-1);
    rhs.len--;
  }
}

__global__ 
void recurse_kernel(DataWithPos* d, double **mat, int row, int target, int* lhs, int lhsSize, char* lhsMap, my_vector<int>* rhs, int begin)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < 0 || i > begin) return;
  if (lhsMap[i] == 1) {
    return;
  }
  rhs[i].mv[rhs[i].len] = i;
  rhs[i].len++;
  // printf("len: %d, back: %d\n", rhs[i].len, rhs[i].mv[rhs[i].len-1]);
  recurse(d, mat, row, target-d[i].data, lhs, lhsSize, lhsMap, rhs[i], i-1);
  rhs[i].len--;
}

void parallel(DataWithPos* d, double **mat, int row, int target, int* lhs, int lhsSize, char* lhsMap, int begin) {
  dim3 blockSize(512);
  dim3 gridSize((begin + 1 + blockSize.x - 1) / blockSize.x);
  my_vector<int>* tmp = new my_vector<int>[begin+1];
  my_vector<int>* many_rhs;
  cudaMallocManaged(&many_rhs, (begin+1) * sizeof(my_vector<int>));
  for (int i = 0; i <= begin+1; i++) {
    many_rhs[i] = tmp[i];
  }
  recurse_kernel<<<gridSize, blockSize>>>(d, mat, row, target, lhs, lhsSize, lhsMap, many_rhs, begin);
  cudaDeviceSynchronize();
  cudaFree(many_rhs);
  delete []tmp;
}

void combination(DataWithPos* d, double **mat, int row, int col, int lhsNum, vector<int>& lhs) {
  if (lhsNum == 0) {
    char* lhsMap;
    cudaMallocManaged(&lhsMap, col * sizeof(char));
    int *lhsDev;
    cudaMallocManaged(&lhsDev, lhs.size() * sizeof(int));
    cudaMemset(lhsMap, 0, col);
    for (int i = 0; i < lhs.size(); i++) {
      lhsMap[lhs[i]] = 1;
      lhsDev[i] = lhs[i];
    }
    double target = accuSum(d, lhs);
    // printVector("lhs", lhs);
    // my_vector<int> rhs(col/2);
    parallel(d, mat, row, target, lhsDev, lhs.size(), lhsMap, lhs[0]-1);
    cudaFree(lhsDev);
    cudaFree(lhsMap);
    return ;
  }

  int begin = col-1-lhs.size();
  if (lhs.size() > 0) begin = min(begin, lhs.back()-1);
  int end = 0;
  if (lhs.size() == 0) end = lhsNum;
  for (int i = begin; i >= end; i--) {
    if (lhs.size() == 0 && lhsNum > (i+1)/2) return ;
    lhs.push_back(i);
    combination(d, mat, row, col, lhsNum-1, lhs);
    lhs.pop_back();
  }
}

// __global__ 
// void combination_kernel(DataWithPos* d, double **mat, int row, int col)
// {
//   int lhsNum = blockIdx.x * blockDim.x + threadIdx.x;
//   my_vector<int> lhs(col/2);
//   printf("processing lhsNum: %d\n", lhsNum);
//   combination(d, mat, row, col, lhsNum, lhs);
// }

void find(DataWithPos* d, double **mat, int row, int col) {
  struct sysinfo si;
  sysinfo(&si);
  printf("total available ram: %ld\n", si.totalram);
  printf("get_nprocs_conf: %d\n", get_nprocs_conf());
  printf("get_nprocs: %d\n", get_nprocs());

  for (int lhsNum = 1; lhsNum <= col/2; lhsNum++) {
    printf("processing lhsNum: %d\n", lhsNum);
    vector<int> lhs;
    combination(d, mat, row, col, lhsNum, lhs);
  }
}

int main(int argc, char** argv)
{
  clock_t begin = clock();
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
  
  double **mat, *sum;
  cudaMallocManaged(&mat, row * sizeof(double*));
  for (size_t i = 0; i < row; i++) {
    cudaMallocManaged(&mat[i], col * sizeof(double));
  }
  cudaMallocManaged(&sum, col * sizeof(double));

  readFile(fileName.c_str(), mat);

  costtime(begin);

  for (int i = 0; i < col; i++) {
    double tmp = 0;
    for (int j = 0; j < row; j++) {
      tmp += mat[j][i];
    }
    sum[i] = tmp;
  }
  printf("%f, %f\n", sum[0], sum[col-1]);
  costtime(begin);

  DataWithPos* dataWithPos;
  cudaMallocManaged(&dataWithPos, col * sizeof(DataWithPos));
  for (int i = 0; i < col; i++) {
    dataWithPos[i].data = sum[i];
    dataWithPos[i].pos = i;
  }
  qsort(dataWithPos, col, sizeof(DataWithPos), cmpfunc);
  printf("%f, %f, %f\n", dataWithPos[0].data, dataWithPos[1].data, dataWithPos[col-1].data);
  printf("%d, %d, %d\n", dataWithPos[0].pos, dataWithPos[1].pos, dataWithPos[col-1].pos);

  find(dataWithPos, mat, row, col);

  cudaFree(dataWithPos);
  cudaFree(sum);
  for (size_t i = 0; i < row; i++) {
    cudaFree(mat[i]);
  }
  cudaFree(mat);

  return 0;
}
