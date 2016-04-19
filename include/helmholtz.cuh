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
			__device__ void vector_multiply(vector&, vector&);
			__host__ __device__ helmholtz(unsigned int nx, unsigned int ny, unsigned int n, float scale) :
				nx(nx), ny(ny), n(n), scale(scale) {}
			__host__ vector operator*(vector&);
		// XXX for some reason this is now broken when compiling with nvcc...
		//friend __global__ void matrix_vector_multiply<>(vector, helmholtz, vector);
	};

}

#endif
