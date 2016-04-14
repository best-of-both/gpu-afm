#ifndef DT_HELMHOLTZ_H
#define DT_HELMHOLTZ_H

#include "vector.cuh"
#include "matrix.cuh"

namespace dt {

	class helmholtz {
		private:
			unsigned int nx, ny, n;
			float scale;
		protected:
		public:
			// XXX wish to move this to `protected` vis.
			__device__ float vector_multiply(vector& vec);
			__host__ __device__ helmholtz(unsigned int nx, unsigned int ny, unsigned int n, float scale) :
				nx(nx), ny(ny), n(n), scale(scale) {}
			__host__ vector operator*(vector&);
		// XXX for some reason this is now broken when compiling with nvcc...
		//friend __global__ void matrix_vector_multiply<>(vector, helmholtz, vector);
	};

	__device__ float
	helmholtz::vector_multiply(vector& vec)
	{
		unsigned int thread = blockDim.x * blockIdx.x + threadIdx.x;
		unsigned int col = thread % nx;
		unsigned int row = thread / nx;
		
		float down = row > 0 ? vec[thread - nx] : 0,
		      left = col > 0 ? vec[thread - 1] : 0,
		      center = vec[thread],
		      right = col < nx - 1 ? vec[thread + 1] : 0,
		      up = row < ny - 1 ? vec[thread + nx] : 0;
		return center + scale * n * n * (down + left + right + up - 4 * center);
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

#endif
