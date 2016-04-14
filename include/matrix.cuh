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

	template<typename T, unsigned int N, unsigned int M>
	__global__ void
	matrix_vector_multiply(vector<M> l, T a, vector<N> r)
	{
		unsigned int thread = blockDim.x * blockIdx.x + threadIdx.x;
		if (thread < M)
			l[thread] = a.vector_multiply(r);
	}

	template<typename V1, typename T, typename V2>
	__host__ void
	launch_mult_kernel(V1& l, T& a, V2& r, dim3 grid, dim3 block)
	{
		matrix_vector_multiply<<<grid, block>>>(l, a, r);
		CHECK(cudaGetLastError());
	}

}

#endif
