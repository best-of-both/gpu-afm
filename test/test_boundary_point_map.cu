#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "cuda.h"
#include "mapping_matrix.cuh"

int main(int argc, char **argv) {
	const int s = 10;
	const int n = s * s;
	unsigned char *h_bd = (unsigned char *) malloc(n * sizeof(unsigned char));
	for (int i = 0; i < n; ++i){
		h_bd[i] = (unsigned char) (i % s != 0 && i % s != s - 1 && i / s != 0 && i / s != s - 1);
		printf("%d ", h_bd[i]);
	}
	printf("\n");
	unsigned char *d_bd;
	unsigned int *d_rows, *d_num_els;
	unsigned int *rows, num_els;

	cudaMalloc((void **) &d_bd, n * sizeof(unsigned char));
	cudaMalloc((void **) &d_rows, n * sizeof(unsigned int));
	cudaMalloc((void **) &d_num_els, n * sizeof(unsigned int));
	cudaMemcpy(d_bd, h_bd, n * sizeof(unsigned char), cudaMemcpyHostToDevice);

	dt::map_boundary_points<<<1, n>>>(d_bd, d_num_els, d_rows, n);
	CHECK(cudaGetLastError());

	cudaMemcpy(&num_els, d_num_els, sizeof(unsigned int), cudaMemcpyDeviceToHost);
	rows = (unsigned int *) malloc(num_els * sizeof(unsigned int));
	cudaMemcpy(rows, d_rows, num_els * sizeof(unsigned int), cudaMemcpyDeviceToHost);
	printf("%d\n", num_els);
	for (int i = 0; i < num_els; ++i)
		printf("%d ", rows[i]);
	printf("\n");
	return 0;
}
