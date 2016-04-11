#ifndef DT_MATRIX_H
#define DT_MATRIX_H

#include "stdio.h"
#include "vector.h"

#define CHECK(e) _errorCheck(e, __FILE__, __LINE__)

void _errorCheck(cudaError_t e, const char* file, int line){
	if(e != cudaSuccess){
		printf("Failed to run statement (%s:%d): %s \n",
		       file, line, cudaGetErrorString(e));
		exit(1);
	}
}

namespace dt {

	template<unsigned int nx, unsigned int ny, unsigned int M = nx * ny>
	class matrix;

	template<unsigned int nx, unsigned int ny, unsigned int M>
	__global__ void
	matrix_vector_multiply(vector<nx * ny>, matrix<nx, ny, M>, vector<M>);
	
	template<unsigned int nx, unsigned int ny, unsigned int M>
	class matrix {
		protected:
			int* d_values;
			__device__ float vector_multiply(vector<M>& vec);
			friend void matrix_vector_multiply<>(vector<nx * ny>, matrix<nx, ny, M>, vector<M>);
		public:
			__host__ __device__ matrix() {}

	};

	template<unsigned int nx, unsigned int ny, unsigned int M>
	__device__ float
	matrix<nx, ny, M>::vector_multiply(vector<M>& vec)
	{
		unsigned int thread = blockDim.x * blockIdx.x + threadIdx.x;
		unsigned int col = thread % nx;
		unsigned int row = thread / nx;
		
		float down = row > 0 ? vec[thread - nx] : 0,
		      left = col > 0 ? vec[thread - 1] : 0,
		      center = vec[thread],
		      right = col < nx - 1 ? vec[thread + 1] : 0,
		      up = row < ny - 1 ? vec[thread + nx] : 0;
		return down + left + right + up - 4 * center;
	}

	template<unsigned int nx, unsigned int ny, unsigned int M>
	__global__ void
	matrix_vector_multiply(vector<nx * ny> r, matrix<nx, ny, M> A, vector<M> v)
	{
		unsigned int thread = blockDim.x * blockIdx.x + threadIdx.x;
		if (thread < nx * ny)
			r[thread] = A.vector_multiply(v);
	}

	template<unsigned int nx, unsigned int ny, unsigned int M>
	__host__ vector<nx * ny>
	operator*(matrix<nx, ny, M>& A, vector<M>& v)
	{
		vector<nx * ny> r;
		matrix_vector_multiply<<<nx * ny / 1024, 1024>>>(r, A, v);
		CHECK(cudaGetLastError());
		return r;
	}

}

#endif
