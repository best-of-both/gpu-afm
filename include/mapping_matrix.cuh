#ifndef MAPPING_MATRIX_CUH
#define MAPPING_MATRIX_CUH

// stuff for P matrix

#include "vector.cuh"
#include "matrix.cuh"

namespace dt {

	__global__ void
	map_boundary_points(unsigned char* is_boundary,
						unsigned int* num_els, unsigned int* rows, unsigned int rowc)
	{

		// r:  0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15
		// r:  0  1  2  3  5  5  6  7  9  9 10 11 12 13 14 15
		// r:  0  1  2  3  5  6  6  7  9 10 10 11 12 13 14 15
		// r:  5  6  2  3  5  6  6  7  9 10 10 11 12 13 14 15
		// r:  5  6  9 10  5  6  6  7  9 10 10 11 12 13 14 15

		// e:  0  0  0  0  0  1  1  0  0  1  1  0  0  0  0  0
		// e:  0  0  0  0  1  1  1  0  1  1  1  0  0  0  0  0
		// e:  0  0  0  0  2  1  1  0  2  1  1  0  0  0  0  0
		// e:  2  0  0  0  2  1  1  0  2  1  1  0  0  0  0  0
		// e:  4  0  0  0  2  1  1  0  2  1  1  0  0  0  0  0
		unsigned int thread = blockDim.x * blockIdx.x + threadIdx.x;
		num_els[thread] = (unsigned int) is_boundary[thread];
		rows[thread] = thread;

		__syncthreads();

		thread <<= 1;
		for (unsigned int stride = 1; thread < rowc && stride < rowc; thread <<= 1, stride <<= 1) {
			unsigned int num_left = num_els[thread],
			             num_right = thread + stride < rowc ? num_els[thread + stride] : 0;
			memcpy(&rows[thread + num_left], &rows[thread + stride], num_right * sizeof(unsigned int));
			num_els[thread] = num_left + num_right;
			__syncthreads();
		}
	}
	

	/*class mapping_matrix {
		private:
			unsigned int rows, cols, *mapping;
		public:
			__host__ __device__ mapping_matrix();
	};*/

}

#endif
