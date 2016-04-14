#ifndef DT_HELMHOLTZ_H
#define DT_HELMHOLTZ_H

#include "vector.cuh"
#include "matrix.cuh"

namespace dt {

	template<unsigned int nx, unsigned int ny, unsigned int n>
	class helmholtz {
		private:
			float scale;
		protected:
			__device__ float vector_multiply(vector<nx * ny>& vec);
			friend void matrix_vector_multiply<>(vector<nx * ny>, helmholtz, vector<nx * ny>);
		public:
			__host__ __device__ helmholtz(float scale) :
				scale(scale) {}
			__host__ vector<nx * ny> operator*(vector<nx * ny>&);
	};

	template<unsigned int nx, unsigned int ny, unsigned int n>
	__device__ float
	helmholtz<nx, ny, n>::vector_multiply(vector<nx * ny>& vec)
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

	template<unsigned int nx, unsigned int ny, unsigned int n>
	__host__ vector<nx * ny>
	helmholtz<nx, ny, n>::operator*(vector<nx * ny>& vec)
	{
		vector<nx * ny> result;
		dim3 grid(nx * ny / 1024);
		dim3 block(1024);
		
		launch_mult_kernel(result, *this, vec, grid, block);
		return result;
	}

}

#endif
