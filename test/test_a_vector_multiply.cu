#include<stdio.h>
#include<string.h>
#include<stdlib.h>
#include "cuda.h"
#include "vector.cuh"
#include "helmholtz.cuh"

int main(int argc, char **argv) {
	float *ys = (float *) malloc(64 * 64 * sizeof(float));

	for (int i = 0; i < 64 * 64; ++i)
		ys[i] = 2.0;

	dt::helmholtz A(64, 64, 1, 1);
	dt::vector x(64 * 64, ys), y = A * x;
	CHECK(cudaGetLastError());

	y.get(ys);

	for (int i = 0; i < 64 * 64; ++i)
		printf("%f\n", ys[i]);

	free(ys);
}
