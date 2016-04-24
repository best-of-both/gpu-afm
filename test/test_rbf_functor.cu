#include <iostream>
#include "typedefs.h"
#include "mq.cuh"
#include "debug.cuh"

__global__ void
gpu_main(data_type* out, unsigned int offset)
{
	unsigned int index = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned int thread = gridDim.x * blockDim.x * offset + index;

	rbfs::mq phi(0.8);

	out[index] = diff(phi(0.1 * thread));
}

int
main(void)
{
	unsigned int n = 1024;
	data_type *h_data, *d_data;

	h_data = (data_type *) malloc(n * n * sizeof(data_type));
	CHECK(cudaMalloc((void **) &d_data, n * n * sizeof(data_type)));
	gpu_main<<<n, n>>>(d_data, 0);
	CHECK(cudaGetLastError());
	CHECK(cudaMemcpy(h_data, d_data, n * n * sizeof(data_type), cudaMemcpyDeviceToHost));

/*	for (unsigned int i = 0; i < n * n; ++i) {
		std::cout << h_data[i];
		if (i % n == n - 1) {
			std::cout << std::endl;
			continue;
		}
		std::cout << " ";
	}*/
	std::cout << "computed " << n * n << " derivatives." << std::endl;
}
