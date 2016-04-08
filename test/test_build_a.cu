#include<stdio.h>
#include<string.h>
#include<stdlib.h>
#include "cuda.h"
#include "build_a.h"

#define BLOCK_SIZE 1024 //@@ Number of threads per block
#define DEBUG 0         //@@ toggle for debug messages
#define REPS 1		//Max number of runs for timing analysis

/* Choose which kernel to run:
  1) Naive implementation
  2) improved branch performance (non-divergent)
  3) Non Divergent  total w/ GPU summing
  4) Sequential Addressing
*/
#define KERNEL_SELECT 1

char *inputFile;
char *outputFile;

#define CHECK(e) _errorCheck(e, __FILE__, __LINE__)

void _errorCheck(cudaError_t e, const char* file, int line){
	if(e != cudaSuccess){
		printf("Failed to run statement: %s %d \n", file, line);
		exit(1);
	}
}


int main(int argc, char **argv) {

	int* h_a;
	int* d_a;

	const unsigned int nx = 64;
	const unsigned int ny = 64;

	CHECK(cudaMalloc((void**) &d_a, 4 * nx * ny * sizeof(int)));
	h_a = (int*) malloc(4 * nx * ny * sizeof(int));

	unsigned int grid_x = nx * ny / 1024;

	device_build_a<<<grid_x, 1024>>>(d_a, nx, ny);
	CHECK(cudaGetLastError());

	CHECK(cudaMemcpy(h_a, d_a, 4 * nx * ny * sizeof(int), cudaMemcpyDeviceToHost));

	for (unsigned int i = 0; i < nx * ny; ++i)
		printf("%d %d -4 %d %d\n", h_a[4 * i + 0], h_a[4 * i + 1], h_a[4 * i + 2], h_a[4 * i + 3]);
}
