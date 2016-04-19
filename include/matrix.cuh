#ifndef DT_MATRIX_H
#define DT_MATRIX_H

#include "vector.cuh"

namespace dt {

	template<typename T>
	__global__ void
	matrix_vector_multiply(vector l, T a, vector r)
	{
		a.vector_multiply(l, r);
	}

	template<typename T>
	__host__ void
	launch_mult_kernel(vector& l, T& a, vector& r, dim3 grid, dim3 block)
	{
		matrix_vector_multiply<<<grid, block>>>(l, a, r);
		//CHECK(cudaGetLastError());
	}

}

#endif
