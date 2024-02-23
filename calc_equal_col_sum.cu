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

void printCudaMem(void);

#define cudaMallocManagedWrap(a, b) \
do { \
  cudaError_t code = cudaMallocManaged((a), (b)); \
  if (code != cudaSuccess) { \
    printf("cudaMallocManaged failed: %d\n", code); \
    printCudaMem(); \
    exit(-1); \
  } \
}while(0);

unsigned int hash(unsigned int x) {
  x = ((x >> 16) ^ x) * 0x45d9f3b;
  x = ((x >> 16) ^ x) * 0x45d9f3b;
  x = (x >> 16) ^ x;
  return x;
}

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

    my_vector() : cap(10000), len(0) {
      cudaMallocManagedWrap(&mv, 10000 * sizeof(T));
    }
    explicit my_vector(size_t _cap) : cap(_cap), len(0) {
      cudaMallocManagedWrap(&mv, _cap * sizeof(T));
    }
    explicit my_vector(vector<int>& v) : cap(v.size()), len(v.size()) {
      cudaMallocManagedWrap(&mv, v.size() * sizeof(T));
      for (int i = 0; i < v.size(); i++) {
        mv[i] = v[i];
      }
    }
    explicit my_vector(my_vector<int>& v) : cap(v.cap), len(v.len) {
      cudaMallocManagedWrap(&mv, v.cap * sizeof(T));
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

void printVector(const char* prefix, vector<int>& v) {
  if (v.size() == 0) return;
  printf("%s: ", prefix);
  for (int i = 0; i < v.size(); i++) {
    printf("%d ", int(v[i]));
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
bool isInLhs(int* lhs, int lhsSize, int i) {
  for (int j = 0; j < lhsSize; j++) {
    if (i == lhs[j]) {
      return true;
    }
  }
  return false;
}

__device__
void recurse(int thd, DataWithPos* d, double **mat, int row, double& target, int* lhs, int lhsSize, char* lhsMap, my_vector<int>& rhs, int begin, int& recurseDepth) {
  if (recurseDepth >= 4) return ;
  // printf("begin recurse\n");
  // if (target <= 0) {
  //   printVector("lhs", lhs);
  //   printVector("rhs", rhs);
  // }
  if (target < 0) return ;
  if (fabs(target) < 0.000001) {
    if (checkResult(d, mat, row, lhs, lhsSize, rhs)) {
      // char out[1024];
      // printf("%s = %s\n", toString(d, lhs).c_str(), toString(d, rhs).c_str());
      printVector(lhs, lhsSize, rhs);
      // writeFile(resultFile.c_str(), out);
    }
    return ;
  }

  recurseDepth++;
  // printf("thd:%d, recurseDepth: %d\n", thd, recurseDepth);
  for (int i = begin; i >= 0; i--) {
    if (isInLhs(lhs, lhsSize, i)) continue;
    // if (lhsMap[i] == 1) {
    //   continue;
    // }
    rhs.mv[rhs.len] = i;
    rhs.len++;
    target -= d[i].data;
    recurse(thd, d, mat, row, target, lhs, lhsSize, lhsMap, rhs, i-1, recurseDepth);
    if (recurseDepth < 4) {
      target += d[i].data;
      rhs.len--;
    }
  }
}

__global__ 
void recurse_kernel(DataWithPos* d, double **mat, int row, double target, int* lhs, int lhsSize, char* lhsMap, my_vector<int>* rhs, int begin)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < 0 || i > begin) return;
  if (isInLhs(lhs, lhsSize, i)) return;
  // if (lhsMap[i] == 1) {
  //   return;
  // }
  // printf("thd: %d\n", i);
  rhs[i].mv[rhs[i].len] = i;
  rhs[i].len++;
  // printf("len: %d, back: %d\n", rhs[i].len, rhs[i].mv[rhs[i].len-1]);
  int recurseDepth;
  double recurseTarget = target-d[i].data;
  do {
    recurseDepth = 0;
    recurse(i, d, mat, row, recurseTarget, lhs, lhsSize, lhsMap, rhs[i], rhs[i].mv[rhs[i].len-1] - 1, recurseDepth);
    // printf("recurseDepth: %d\n", recurseDepth);
  } while (recurseDepth >= 4);
  rhs[i].len--;
}

void parallel(DataWithPos* d, double **mat, int row, double target, int* lhs, int lhsSize, char* lhsMap, int begin, my_vector<int> *many_rhs) {
  dim3 blockSize(512);
  dim3 gridSize((begin + 1 + blockSize.x - 1) / blockSize.x);
  recurse_kernel<<<gridSize, blockSize>>>(d, mat, row, target, lhs, lhsSize, lhsMap, many_rhs, begin);
  cudaDeviceSynchronize();
}

void combination(DataWithPos* d, double **mat, int row, int col, int lhsNum, vector<int>& lhs, int* lhsDev, char* lhsMap, my_vector<int> *many_rhs) {
  if (lhsNum == 0) {
    // printVector("lhs", lhs);
    // printCudaMem();
    // cudaMemset(lhsMap, 0, col);
    for (int i = 0; i < lhs.size(); i++) {
      if (lhs[i] < 0 || lhs[i] >= col) {
        printf("invalid lhs value: %d\n", lhs[i]);
        break;
      }
      // lhsMap[lhs[i]] = 1;
      lhsDev[i] = lhs[i];
    }
    double target = accuSum(d, lhs);
    for (int i = 0; i < col; i++) {
      many_rhs[i].len = 0;
    }
    parallel(d, mat, row, target, lhsDev, lhs.size(), lhsMap, lhs[0]-1, many_rhs);
    return ;
  }

  int begin = col-1-lhs.size();
  if (lhs.size() > 0) begin = min(begin, lhs.back()-1);
  int end = 0;
  if (lhs.size() == 0) end = lhsNum;
  for (int i = begin; i >= end; i--) { // todo: begin 4215
    if (lhs.size() == 0 && lhsNum > (i+1)/2) return ;
    lhs.push_back(i);
    combination(d, mat, row, col, lhsNum-1, lhs, lhsDev, lhsMap, many_rhs);
    lhs.pop_back();
  }
}

void find(DataWithPos* d, double **mat, int row, int col) {
  struct sysinfo si;
  sysinfo(&si);
  printf("total available ram: %ld\n", si.totalram);
  printf("get_nprocs_conf: %d\n", get_nprocs_conf());
  printf("get_nprocs: %d\n", get_nprocs());

  int *lhsDev;
  cudaMallocManagedWrap(&lhsDev, (col/2) * sizeof(int));
  char* lhsMap = NULL;
  // cudaMallocManagedWrap(&lhsMap, col * sizeof(char));
  my_vector<int>* tmp = new my_vector<int>[col];
  my_vector<int>* many_rhs;
  cudaMallocManagedWrap(&many_rhs, col * sizeof(my_vector<int>));
  for (int i = 0; i <= col; i++) {
    many_rhs[i] = tmp[i];
  }
  for (int lhsNum = 1; lhsNum <= col/2; lhsNum++) {
    printf("processing lhsNum: %d\n", lhsNum);
    vector<int> lhs;
    combination(d, mat, row, col, lhsNum, lhs, lhsDev, lhsMap, many_rhs);
  }
  cudaFree(many_rhs);
  delete []tmp;
  // cudaFree(lhsMap);
  cudaFree(lhsDev);
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
  cudaMallocManagedWrap(&mat, row * sizeof(double*));
  for (size_t i = 0; i < row; i++) {
    cudaMallocManagedWrap(&mat[i], col * sizeof(double));
  }
  cudaMallocManagedWrap(&sum, col * sizeof(double));

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
  cudaMallocManagedWrap(&dataWithPos, col * sizeof(DataWithPos));
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

void printCudaMem(void) {
  // show memory usage of GPU
  size_t free_byte ;
  size_t total_byte ;
  cudaError_t cuda_status = cudaMemGetInfo( &free_byte, &total_byte ) ;
  if ( cudaSuccess != cuda_status ){
      printf("Error: cudaMemGetInfo fails, %s \n", cudaGetErrorString(cuda_status) );
      exit(1);
  }
  double free_db = (double)free_byte ;
  double total_db = (double)total_byte ;
  double used_db = total_db - free_db ;
  printf("GPU memory usage: used = %f, free = %f MB, total = %f MB\n", used_db/1024.0/1024.0, free_db/1024.0/1024.0, total_db/1024.0/1024.0);
}
