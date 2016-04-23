#include "typedefs.h"
#include "vector.cuh"
#include "mapping.cuh"

namespace dt{

	__device__ index_type* p_row_indices;

	__device__ size_type
	initialize_mapping(size_type rows, unsigned char* n_f)
	{
		index_type thread = blockDim.x * blockIdx.x + threadIdx.x;
		if (thread == 0)
			p_row_indices = (index_type*) malloc(rows * sizeof(index_type));
		if (thread < rows) {
			p_row_indices[thread] = thread;
			__syncthreads();

			for (size_type stride = 1; thread <<= 1 < rows && stride < rows; stride <<= 1) {
				index_type num_left = (index_type) n_f[thread],
						   num_right = thread + stride < rows ? n_f[thread + stride] : 0;
				memcpy(&p_row_indices[thread + num_left], &p_row_indices[thread + stride],
					   num_right * sizeof(index_type));
				n_f[thread] = num_left + num_right;
				__syncthreads();
			}
		}
		return n_f[0];
	}

	__device__
	mapping::mapping(size_type rows, unsigned char* n_f) :
		matrix(rows, initialize_mapping(rows, n_f)) {}

	__device__
	mapping::~mapping()
	{
		free(p_row_indices);
	}

	__device__ data_type
	mapping::operator*(vector& in)
	{
		index_type thread = blockDim.x * blockIdx.x + threadIdx.x;
		if (thread < cols())
			return in[p_row_indices[thread]];
		return 0;
	}
	
}
