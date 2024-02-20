#include <stdio.h>
#include <math.h>
#include <sys/time.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

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

    fp = fopen(fileName, "w");
    if (fp == NULL) {
        printf("open file failed!\n");
        return -1;
    }

    int ret = fputs(str, fp);
    if (ret < 0) printf("fputs failed: %d\n", ret);

    fclose(fp);

    return 0;
}
