#ifndef DT_MATRIX_H
#define DT_MATRIX_H

#include "stdio.h"
#include "vector.cuh"

#define CHECK(e) _errorCheck(e, __FILE__, __LINE__)

void _errorCheck(cudaError_t e, const char* file, int line){
	if(e != cudaSuccess){
		printf("Failed to run statement (%s:%d): %s \n",
		       file, line, cudaGetErrorString(e));
		exit(1);
	}
}

namespace dt {

	template<typename T>
	__global__ void
	matrix_vector_multiply(vector l, T a, vector r)
	{
		unsigned int thread = blockDim.x * blockIdx.x + threadIdx.x;
		if (thread < l.size())
			l[thread] = a.vector_multiply(r);
	}

	template<typename T>
	__host__ void
	launch_mult_kernel(vector& l, T& a, vector& r, dim3 grid, dim3 block)
	{
		matrix_vector_multiply<<<grid, block>>>(l, a, r);
		CHECK(cudaGetLastError());
	}

}

#endif
