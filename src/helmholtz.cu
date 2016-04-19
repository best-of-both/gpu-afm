#include "vector.cuh"
#include "matrix.cuh"
#include "helmholtz.cuh"

namespace dt {

	__device__ void
	helmholtz::vector_multiply(vector& out, vector& in)
	{
		unsigned int thread = blockDim.x * blockIdx.x + threadIdx.x;
		if (thread < nx * ny) {
			unsigned int col = thread % nx;
			unsigned int row = thread / nx;
			
			float down = row > 0 ? in[thread - nx] : 0,
				  left = col > 0 ? in[thread - 1] : 0,
				  center = in[thread],
				  right = col < nx - 1 ? in[thread + 1] : 0,
				  up = row < ny - 1 ? in[thread + nx] : 0;
			out[thread] center + scale * n * n * (down + left + right + up - 4 * center);
		}
	}

	__host__ vector
	helmholtz::operator*(vector& vec)
	{
		vector result(nx * ny);
		dim3 grid(nx * ny / 1024);
		dim3 block(1024);
		
		launch_mult_kernel(result, *this, vec, grid, block);
		return result;
	}

}
