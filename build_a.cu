#include "stdio.h"

__global__ void
device_build_a(int* a, unsigned int nx, unsigned int ny)
{
	/*
	   Store A as an nx * ny x 4 matrix:
	    * first column:  1 if row > 0 otherwise 0, matrix column is row - nx
	    * second column: 1 if col > 0 otherwise 0, matrix column is row - 1
	    * third column:  1 if col < nx-1 otherwise 0, matrix column is row + 1
	    * fourth column: 1 if row < ny-1 otherwise 0, matrix column is row + nx

	   The diagonal of A is always -4, so we do not need to store that.
	 */

	unsigned int thread = blockDim.x * blockIdx.x + threadIdx.x;
	if (thread < nx * ny)
	{
		unsigned int col = thread % nx;
		unsigned int row = thread / nx;

		a[4 * thread    ] = (int) (row > 0u);
		a[4 * thread + 1] = (int) (col > 0u);
		a[4 * thread + 2] = (int) (col < nx-1);
		a[4 * thread + 3] = (int) (row < ny-1);
	}
}
