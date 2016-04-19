#include <iostream>
#include <stdio.h>
#include "mapping_matrix.cuh"
#include "forcing.cuh"

namespace dt {

	__global__ void
	map_forcing_points(forcing_data data, unsigned int* num_els, unsigned int* rows)
	{
		unsigned int thread = blockDim.x * blockIdx.x + threadIdx.x;
		const unsigned int rowc = data.size();
		num_els[thread] = (unsigned int) data[thread];
		rows[thread] = thread;
		__syncthreads();

		thread <<= 1;
		for (unsigned int stride = 1; thread < rowc && stride < rowc; thread <<= 1, stride <<= 1) {
			unsigned int num_left = num_els[thread],
			             num_right = thread + stride < rowc ? num_els[thread + stride] : 0;
			memcpy(&rows[thread + num_left], &rows[thread + stride],
			       num_right * sizeof(unsigned int));
			num_els[thread] = num_left + num_right;
			__syncthreads();
		}
	}


	__host__
	mapping_matrix::mapping_matrix(forcing_data data)
	{
		rows = data.size();
		unsigned int *num_els;
		cudaMalloc((void **) &num_els, rows * sizeof(unsigned int));
		cudaMalloc((void **) &mapping, rows * sizeof(unsigned int));
		map_forcing_points<<<rows / 1024, 1024>>>(data, num_els, mapping);
		cudaMemcpy(&cols, num_els, sizeof(unsigned int), cudaMemcpyDeviceToHost);
		cudaFree(num_els);
	}

	__device__ void
	mapping_matrix::vector_multiply(vector& out, vector& in)
	{
		const unsigned int thread = blockIdx.x * blockDim.x + threadIdx.x;
		if (thread < cols)
			out[mapping[thread]] = in[thread];
	}

	__host__ vector
	mapping_matrix::operator*(vector r)
	{
		std::cout << "called" << std::endl;

		vector l(rows);
		matrix_vector_multiply<<<1, cols>>>(l, *this, r);
		return l;
	}

}

