#include "typedefs.h"
#include "vector.cuh"
#include "mapping.cuh"
#include <stdio.h>

namespace dt{

	__device__ index_type
	compute_index(size_type rows, unsigned int* n_f)
	{
		index_type thread = blockDim.x * blockIdx.x + threadIdx.x;
		index_type offset = 1;
		index_type result = 0;
		size_type stride = 1;

		while (stride < rows) {
			stride <<= 1;
		   	if (offset + thread * stride < rows)
				n_f[offset + thread * stride] += n_f[offset + thread * stride - (stride >> 1)];
			__syncthreads();
			offset += stride;
		}

		offset = 0;
		stride >>= 1;
		while (stride > 0) {
			bool right = (thread > offset + stride - 1);
			if (offset + stride - 1 < rows)
				result += n_f[offset + stride - 1] * right;
			offset += stride * right;
			stride >>= 1;
		}

		return result;
	}

	__device__
	mapping::mapping(size_type rows, size_type cols, bool is_forcing, index_type forcing_index) :
		matrix(rows, cols), is_forcing(is_forcing), forcing_index(forcing_index) {}

	__device__ data_type
	mapping::operator*(vector& in)
	{
		return is_forcing * in[forcing_index];
	}
	
}
