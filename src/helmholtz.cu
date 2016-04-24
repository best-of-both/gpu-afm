#include "vector.cuh"
#include "helmholtz.cuh"

namespace dt {

	__device__ data_type
	helmholtz::operator*(vector& in)
	{
		index_type thread = blockDim.x * blockIdx.x + threadIdx.x;
		if (thread < nx * ny) {
			unsigned int col = thread % nx;
			unsigned int row = thread / nx;

			float down = row > 0 ? in[thread - nx] : 0,
			      left = col > 0 ? in[thread - 1] : 0,
			      center = in[thread],
			      right = col < nx - 1 ? in[thread + 1] : 0,
			      up = row < ny - 1 ? in[thread + nx] : 0;
			return center + scale * n * n * (down + left + right + up - 4 * center);
		}
		return 0;
	}

}
